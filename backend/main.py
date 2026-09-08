# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 6.1 (Corregido: Retrieval de Equipos + Fuzzy Matching + Conteo + Fix Enseñar + Blindaje ERP/Inferencia)
#
# PROVEEDOR DE IA:
# - Google Gemini Native API
#
# MODELO:
# - gemini-3.5-flash-lite
#
# FUNCIONES:
# - Chat con Penaguillo
# - Chat con historial
# - Búsqueda contextual
# - Enseñar texto (Con Títulos dinámicos)
# - Enseñar imágenes
# - Enseñar PDF
# - PDF con texto seleccionable -> PyMuPDF
# - PDF escaneado -> Gemini Vision
# - Imágenes -> Gemini Vision
# - Persistencia local
# - Google Drive como almacenamiento permanente
# - penaguillo.json como fuente maestra
# - Backups
# - Escritura atómica
# - Búsqueda local por relevancia inteligente
# - Búsqueda difusa (tolerancia a errores ortográficos)
# - Deduplicación inteligente
# ============================================================

import base64
import difflib  # <-- IMPORTANTE: Librería para búsqueda difusa
import hashlib
import json
import os
import re
import sys
import time
import uuid

from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import requests

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# GOOGLE DRIVE
# ============================================================

try:

    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import AuthorizedSession

    GOOGLE_AVAILABLE = True

except ImportError:

    GOOGLE_AVAILABLE = False


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

if getattr(sys, "frozen", False):

    BASE_DIR = (
        Path(sys.executable)
        .resolve()
        .parent
    )

else:

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parent
    )


# ============================================================
# DIRECTORIOS
# ============================================================

if os.getenv("RENDER") == "true":

    CONOCIMIENTO_DIR = (
        BASE_DIR
        / "storage"
        / "conocimiento"
    )

else:

    CONOCIMIENTO_DIR = (
        BASE_DIR
        / "conocimiento"
    )


ARCHIVOS_DIR = (
    CONOCIMIENTO_DIR
    / "archivos"
)

PDF_DIR = (
    ARCHIVOS_DIR
    / "pdf"
)

IMAGENES_DIR = (
    ARCHIVOS_DIR
    / "imagenes"
)

BACKUP_DIR = (
    CONOCIMIENTO_DIR
    / "backups"
)

ARCHIVO_CONOCIMIENTO = (
    CONOCIMIENTO_DIR
    / "penaguillo.json"
)

PROMPT_FILE = (
    BASE_DIR
    / "conocimiento"
    / "prompt.txt"
)


# ============================================================
# CREAR CARPETAS
# ============================================================

CONOCIMIENTO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARCHIVOS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGENES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ENV
# ============================================================

ENV_FILE = (
    BASE_DIR
    / ".env"
)

load_dotenv(
    ENV_FILE
)


# ============================================================
# GEMINI NATIVO
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)


if not GEMINI_API_KEY:

    print(
        "⚠️ ADVERTENCIA: "
        "GEMINI_API_KEY no está configurada."
    )

else:

    print(
        "🔑 GEMINI_API_KEY: configurada"
    )


# ============================================================
# MODELOS
# ============================================================

CHAT_MODEL = GEMINI_MODEL

VISION_MODEL = GEMINI_MODEL


# ============================================================
# CONFIGURACIÓN DE TOKENS
# ============================================================

MAX_OUTPUT_TOKENS = 1200


# ============================================================
# CONFIGURACIÓN DEL RETRIEVAL LOCAL
# ============================================================

RELEVANCIA_TOP_K = 8

# CAMBIADO DE 16 a 32 PARA SOPORTAR EQUIPOS GRANDES
MAX_KB_CONOCIMIENTO_CHAT = 32

MAX_CHARS_CONOCIMIENTO_CHAT = (
    MAX_KB_CONOCIMIENTO_CHAT * 1024
)


# ============================================================
# CONFIGURACIÓN DEL HISTORIAL
# ============================================================

MAX_MENSAJES_HISTORIAL = 6


# ============================================================
# CONFIGURACIÓN DE BÚSQUEDA CONTEXTUAL
# ============================================================

MAX_MENSAJES_RETRIEVAL = 4

MAX_CHARS_CONSULTA_RETRIEVAL = 4000


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS_ES = {

    "a", "al", "algo", "algunas", "algunos",
    "ante", "antes", "como", "con", "contra",
    "cual", "cuales", "cuando", "de", "del",
    "desde", "donde", "dos", "el", "ella",
    "ellas", "ello", "ellos", "en", "entre",
    "era", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estas", "este", "esto",
    "estos", "fue", "ha", "hay", "la", "las",
    "le", "les", "lo", "los", "más", "me",
    "mi", "mis", "muy", "no", "nos", "o",
    "para", "pero", "por", "que", "qué", "se",
    "sea", "si", "sí", "sin", "sobre", "son",
    "su", "sus", "también", "te", "tener",
    "ti", "tu", "tus", "un", "una", "unas",
    "uno", "unos", "y", "ya", "yo",

}


# ============================================================
# ERROR CONTROLADO DE GEMINI
# ============================================================

class GeminiError(RuntimeError):

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: int | None = None,
    ):

        super().__init__(
            message
        )

        self.status_code = status_code

        self.retry_after = retry_after


# ============================================================
# GEMINI — EXTRAER RETRY-AFTER
# ============================================================

def extraer_retry_after(
    respuesta: requests.Response,
    mensaje_error: str,
) -> int | None:

    valor_header = respuesta.headers.get(
        "Retry-After"
    )

    if valor_header:

        try:

            segundos = int(
                valor_header
            )

            if segundos >= 0:

                return segundos

        except (
            TypeError,
            ValueError,
        ):

            pass


    patrones = [

        r"retry in ([0-9]+(?:\.[0-9]+)?)s",

        r"retryDelay.*?([0-9]+)s",

        r"seconds.*?([0-9]+)",

    ]


    texto = (
        mensaje_error
        or ""
    )


    for patron in patrones:

        coincidencia = re.search(
            patron,
            texto,
            re.IGNORECASE,
        )

        if coincidencia:

            try:

                return max(
                    1,
                    int(
                        float(
                            coincidencia.group(1)
                        )
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):

                pass


    return None


# ============================================================
# GEMINI — GENERAR RESPUESTA
# ============================================================

def generar_con_gemini(
    *,
    model: str,
    messages: list,
    max_retries: int = 2,
):

    if not GEMINI_API_KEY:

        raise GeminiError(
            "GEMINI_API_KEY no está configurada."
        )


    # ========================================================
    # URL GEMINI NATIVE
    # ========================================================

    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{model}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )


    headers = {

        "Content-Type": "application/json",

    }


    system_instruction = None

    gemini_contents = []


    # ========================================================
    # CONVERTIR MENSAJES A GEMINI
    # ========================================================

    for msg in messages:

        role = msg.get("role")
        content = msg.get("content")

        if role == "system":

            if isinstance(content, str):

                system_instruction = {
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                }

            continue


        gemini_role = (
            "model"
            if role == "assistant"
            else "user"
        )

        parts = []

        if isinstance(content, str):

            if content.strip():

                parts.append(
                    {
                        "text": content
                    }
                )

        elif isinstance(content, list):

            for item in content:

                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                if item_type == "text":

                    texto = item.get(
                        "text",
                        "",
                    )

                    if texto:

                        parts.append(
                            {
                                "text": texto
                            }
                        )

                elif item_type == "image_url":

                    image_data = item.get(
                        "image_url",
                        {}
                    )

                    url_img = image_data.get(
                        "url",
                        ""
                    )

                    if not isinstance(
                        url_img,
                        str,
                    ):
                        continue

                    if not url_img.startswith(
                        "data:"
                    ):
                        continue

                    try:

                        encabezado, b64_data = (
                            url_img.split(
                                ",",
                                1,
                            )
                        )

                        if ":" not in encabezado:

                            raise ValueError(
                                "Encabezado MIME inválido."
                            )

                        mime_type = (
                            encabezado
                            .split(
                                ":",
                                1,
                            )[1]
                            .split(
                                ";",
                                1,
                            )[0]
                            .strip()
                        )

                        if not mime_type:

                            raise ValueError(
                                "MIME type vacío."
                            )

                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": b64_data,
                                }
                            }
                        )

                    except (
                        ValueError,
                        IndexError,
                    ) as error:

                        print(
                            "⚠️ Imagen Base64 "
                            f"con formato inválido: {error}"
                        )


        if parts:

            gemini_contents.append(
                {
                    "role": gemini_role,
                    "parts": parts,
                }
            )

    # ========================================================
    # PAYLOAD
    # ========================================================

    payload = {

        "contents": gemini_contents,

        "generationConfig": {

            "maxOutputTokens": (
                MAX_OUTPUT_TOKENS
            ),

            "temperature": 0.3,

        }

    }


    if system_instruction:

        payload[
            "systemInstruction"
        ] = system_instruction


    ultimo_error = None


    # ========================================================
    # PETICIÓN
    # ========================================================

    for intento in range(
        1,
        max_retries + 1,
    ):

        try:

            inicio = time.time()


            print(
                f"🤖 Gemini -> "
                f"modelo={model}, "
                f"intento={intento}/{max_retries}, "
                f"max_tokens={MAX_OUTPUT_TOKENS}"
            )


            respuesta = requests.post(

                url,

                headers=headers,

                json=payload,

                timeout=120,

            )


            duracion = (
                time.time()
                - inicio
            )


            print(
                f"🌐 Gemini HTTP: "
                f"{respuesta.status_code}"
            )


            print(
                f"⏱️ Gemini respondió "
                f"en {duracion:.2f}s"
            )


            if respuesta.status_code == 200:

                try:

                    datos = respuesta.json()

                except ValueError as error:

                    raise GeminiError(

                        "Gemini devolvió "
                        "una respuesta que no es JSON.",

                        status_code=200,

                    ) from error


                return datos


            mensaje_error = (
                respuesta.text[:5000]
            )


            retry_after = (
                extraer_retry_after(
                    respuesta,
                    mensaje_error,
                )
            )


            ultimo_error = GeminiError(

                "Gemini HTTP "
                f"{respuesta.status_code}: "
                f"{mensaje_error}",

                status_code=(
                    respuesta.status_code
                ),

                retry_after=retry_after,

            )


            print(
                "⚠️ Gemini falló:"
            )


            print(
                mensaje_error
            )


            # Errores de configuración.
            if respuesta.status_code in (
                400,
                403,
                404,
            ):

                raise ultimo_error


            es_temporal = (
                respuesta.status_code
                in (
                    429,
                    500,
                    502,
                    503,
                    504,
                )
            )


            if (
                not es_temporal
                or intento >= max_retries
            ):

                raise ultimo_error


            # =================================================
            # ESPERA
            # =================================================

            if retry_after is not None:

                espera = min(
                    retry_after,
                    15,
                )

            else:

                espera = 2 ** intento


            print(
                f"⏳ Error temporal. "
                f"Reintentando en "
                f"{espera}s..."
            )


            time.sleep(
                espera
            )


        except requests.RequestException as error:

            ultimo_error = GeminiError(

                "No fue posible conectar "
                f"con Gemini: {error}",

            )


            print(
                "⚠️ Error de conexión "
                "con Gemini:"
            )


            print(
                error
            )


            if intento >= max_retries:

                raise ultimo_error


            espera = 2 ** intento


            print(
                f"⏳ Reintentando en "
                f"{espera}s..."
            )


            time.sleep(
                espera
            )


    if ultimo_error:

        raise ultimo_error


    raise GeminiError(
        "Gemini no pudo generar una respuesta."
    )


# ============================================================
# GEMINI — EXTRAER CONTENIDO
# ============================================================

def extraer_contenido_gemini(
    respuesta: dict,
) -> str:

    if not respuesta:

        raise RuntimeError(
            "Gemini no devolvió respuesta."
        )


    if "error" in respuesta:

        error_data = respuesta["error"]

        raise RuntimeError(
            "Gemini devolvió un error: "
            f"{error_data}"
        )


    candidatos = respuesta.get(
        "candidates",
        [],
    )


    if not candidatos:

        raise RuntimeError(
            "Gemini no devolvió ningún candidato."
        )


    primer_candidato = candidatos[0]


    contenido = primer_candidato.get(
        "content",
        {},
    )


    partes = contenido.get(
        "parts",
        [],
    )


    texto_final = []


    for parte in partes:

        if "text" in parte:

            texto = parte.get(
                "text",
                "",
            )

            if texto:

                texto_final.append(
                    texto
                )


    resultado = "\n".join(
        texto_final
    ).strip()


    if resultado:

        return resultado


    raise RuntimeError(
        "Gemini no devolvió "
        "contenido de texto."
    )


# ============================================================
# GOOGLE DRIVE — VARIABLES
# ============================================================

GOOGLE_SERVICE_ACCOUNT_EMAIL = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_EMAIL"
)

GOOGLE_PRIVATE_KEY = os.getenv(
    "GOOGLE_PRIVATE_KEY"
)

GOOGLE_PROJECT_ID = os.getenv(
    "GOOGLE_PROJECT_ID"
)

GOOGLE_DRIVE_FOLDER_ID = os.getenv(
    "GOOGLE_DRIVE_FOLDER_ID"
)


# ============================================================
# GOOGLE DRIVE
# ============================================================

DRIVE_SCOPES = [

    "https://www.googleapis.com/auth/drive"

]


drive_service = None

drive_session = None

DRIVE_ROOT_FOLDER = None

DRIVE_SHARED_ID = None

DRIVE_KNOWLEDGE_FOLDER = None

DRIVE_FILES_FOLDER = None

DRIVE_PDF_FOLDER = None

DRIVE_IMAGES_FOLDER = None

DRIVE_BACKUPS_FOLDER = None


# ============================================================
# UTILIDADES
# ============================================================

def ahora_iso() -> str:

    return datetime.now().isoformat()


def generar_id() -> str:

    return str(
        uuid.uuid4()
    )


def nombre_seguro(
    nombre: str,
) -> str:

    nombre = Path(
        nombre
    ).name


    nombre = re.sub(

        r"[^a-zA-Z0-9._-]",

        "_",

        nombre,

    )


    return (

        nombre

        or

        f"archivo_{generar_id()}"

    )


# ============================================================
# GOOGLE DRIVE — BUSCAR ARCHIVO
# ============================================================

def buscar_archivo_drive(
    nombre: str,
    folder_id: str,
    solo_json: bool = False,
):

    if not drive_service:

        return None


    nombre_escapado = (
        nombre.replace(
            "'",
            "\\'",
        )
    )


    query = (

        f"name = '{nombre_escapado}'"

        f" and '{folder_id}' in parents"

        " and trashed = false"

    )


    if solo_json:

        query += (
            " and mimeType = 'application/json'"
        )


    try:

        parametros = {

            "q": query,

            "spaces": "drive",

            "includeItemsFromAllDrives": True,

            "supportsAllDrives": True,

            "fields": (
                "files("
                "id,"
                "name,"
                "mimeType,"
                "size,"
                "modifiedTime,"
                "parents,"
                "shortcutDetails"
                ")"
            ),

            "pageSize": 100,

            "orderBy": (
                "modifiedTime desc"
            ),

        }


        if DRIVE_SHARED_ID:

            parametros["corpora"] = "drive"

            parametros["driveId"] = (
                DRIVE_SHARED_ID
            )


        resultado = (
            drive_service
            .files()
            .list(
                **parametros
            )
            .execute()
        )


        archivos = resultado.get(
            "files",
            [],
        )


        print(
            "🔎 Búsqueda Drive:"
        )


        print(
            f"   Nombre: {nombre}"
        )


        print(
            f"   Carpeta: {folder_id}"
        )


        print(
            f"   Solo JSON: {solo_json}"
        )


        print(
            "   Resultados encontrados: "
            f"{len(archivos)}"
        )


        if archivos:

            archivo = archivos[0]


            if archivo.get(
                "mimeType"
            ) == (
                "application/vnd.google-apps.shortcut"
            ):

                print(
                    "🛑 El archivo encontrado "
                    "es un shortcut."
                )

                return None


            if solo_json:

                if archivo.get(
                    "mimeType"
                ) != "application/json":

                    print(
                        "🛑 El archivo encontrado "
                        "no es application/json."
                    )

                    return None


            print(
                "🎯 Archivo seleccionado:"
            )


            print(
                f"   ID = "
                f"{archivo.get('id')}"
            )


            print(
                f"   Nombre = "
                f"{archivo.get('name')}"
            )


            print(
                f"   MIME = "
                f"{archivo.get('mimeType')}"
            )


            print(
                f"   Tamaño = "
                f"{archivo.get('size', 'N/D')} bytes"
            )


            return archivo


        print(
            "ℹ️ No se encontró el archivo."
        )


        return None


    except Exception as error:

        print(
            "❌ Error buscando archivo "
            "en Drive:"
        )


        print(
            error
        )


        return None


# ============================================================
# GOOGLE DRIVE — CREAR CARPETA
# ============================================================

def obtener_o_crear_carpeta(
    nombre: str,
    parent_id: str,
):

    existente = buscar_archivo_drive(
        nombre,
        parent_id,
    )


    if existente:

        mime_type = existente.get(
            "mimeType"
        )


        if mime_type == (
            "application/vnd.google-apps.folder"
        ):

            print(
                f"📁 Carpeta existente: "
                f"{nombre}"
            )

            return existente["id"]


    metadata = {

        "name": nombre,

        "mimeType": (
            "application/vnd.google-apps.folder"
        ),

        "parents": [
            parent_id
        ],

    }


    archivo = (
        drive_service
        .files()
        .create(

            body=metadata,

            fields=(
                "id,"
                "name,"
                "mimeType,"
                "parents"
            ),

            supportsAllDrives=True,

        )
        .execute()
    )


    print(
        f"📁 Carpeta Drive creada: {nombre}"
    )


    return archivo["id"]


# ============================================================
# GOOGLE DRIVE — INICIALIZAR
# ============================================================

def inicializar_google_drive():

    global drive_service
    global drive_session

    global DRIVE_ROOT_FOLDER
    global DRIVE_SHARED_ID

    global DRIVE_KNOWLEDGE_FOLDER
    global DRIVE_FILES_FOLDER
    global DRIVE_PDF_FOLDER
    global DRIVE_IMAGES_FOLDER
    global DRIVE_BACKUPS_FOLDER


    if os.getenv("RENDER") != "true":

        print(
            "ℹ️ Modo local: "
            "Google Drive no será utilizado."
        )

        return


    if not GOOGLE_AVAILABLE:

        print(
            "❌ Las librerías de Google Drive "
            "no están instaladas."
        )

        return


    if not GOOGLE_SERVICE_ACCOUNT_EMAIL:

        print(
            "❌ Falta "
            "GOOGLE_SERVICE_ACCOUNT_EMAIL"
        )

        return


    if not GOOGLE_PRIVATE_KEY:

        print(
            "❌ Falta GOOGLE_PRIVATE_KEY"
        )

        return


    if not GOOGLE_PROJECT_ID:

        print(
            "❌ Falta GOOGLE_PROJECT_ID"
        )

        return


    if not GOOGLE_DRIVE_FOLDER_ID:

        print(
            "❌ Falta "
            "GOOGLE_DRIVE_FOLDER_ID"
        )

        return


    try:

        private_key = (
            GOOGLE_PRIVATE_KEY
            .replace(
                "\\n",
                "\n",
            )
        )


        credentials_info = {

            "type": "service_account",

            "project_id": (
                GOOGLE_PROJECT_ID
            ),

            "private_key_id": os.getenv(
                "GOOGLE_PRIVATE_KEY_ID",
                "",
            ),

            "private_key": private_key,

            "client_email": (
                GOOGLE_SERVICE_ACCOUNT_EMAIL
            ),

            "client_id": os.getenv(
                "GOOGLE_CLIENT_ID",
                "",
            ),

            "auth_uri": (
                "https://accounts.google.com/o/oauth2/auth"
            ),

            "token_uri": (
                "https://oauth2.googleapis.com/token"
            ),

            "auth_provider_x509_cert_url": (
                "https://www.googleapis.com/oauth2/v1/certs"
            ),

            "client_x509_cert_url": os.getenv(
                "GOOGLE_CLIENT_X509_CERT_URL",
                "",
            ),

        }


        credentials = (
            service_account
            .Credentials
            .from_service_account_info(

                credentials_info,

                scopes=DRIVE_SCOPES,

            )
        )


        drive_service = build(

            "drive",

            "v3",

            credentials=credentials,

            cache_discovery=False,

        )


        drive_session = AuthorizedSession(
            credentials
        )


        DRIVE_ROOT_FOLDER = (
            GOOGLE_DRIVE_FOLDER_ID
        )


        info_carpeta = (
            drive_service
            .files()
            .get(

                fileId=DRIVE_ROOT_FOLDER,

                fields=(
                    "id,"
                    "name,"
                    "mimeType,"
                    "driveId,"
                    "parents"
                ),

                supportsAllDrives=True,

            )

            .execute()
        )


        mime_raiz = (
            info_carpeta.get(
                "mimeType"
            )
        )


        if mime_raiz != (
            "application/vnd.google-apps.folder"
        ):

            raise RuntimeError(
                "GOOGLE_DRIVE_FOLDER_ID "
                "no corresponde a una carpeta."
            )


        DRIVE_SHARED_ID = (
            info_carpeta.get(
                "driveId"
            )
        )


        print(
            "✅ Google Drive conectado."
        )


        print(
            "📁 Carpeta raíz: "
            f"{info_carpeta.get('name')}"
        )


        print(
            "🆔 Folder ID: "
            f"{DRIVE_ROOT_FOLDER}"
        )


        if DRIVE_SHARED_ID:

            print(
                "🗂️ Shared Drive detectado: "
                f"{DRIVE_SHARED_ID}"
            )

        else:

            print(
                "⚠️ La carpeta no reportó "
                "un Shared Drive ID."
            )


        DRIVE_KNOWLEDGE_FOLDER = (
            obtener_o_crear_carpeta(
                "conocimiento",
                DRIVE_ROOT_FOLDER,
            )
        )


        DRIVE_FILES_FOLDER = (
            obtener_o_crear_carpeta(
                "archivos",
                DRIVE_ROOT_FOLDER,
            )
        )


        DRIVE_PDF_FOLDER = (
            obtener_o_crear_carpeta(
                "pdf",
                DRIVE_FILES_FOLDER,
            )
        )


        DRIVE_IMAGES_FOLDER = (
            obtener_o_crear_carpeta(
                "imagenes",
                DRIVE_FILES_FOLDER,
            )
        )


        DRIVE_BACKUPS_FOLDER = (
            obtener_o_crear_carpeta(
                "backups",
                DRIVE_KNOWLEDGE_FOLDER,
            )
        )


        print(
            "📁 Estructura de Google Drive lista."
        )


        print(
            "📂 Carpeta conocimiento ID: "
            f"{DRIVE_KNOWLEDGE_FOLDER}"
        )


        sincronizar_conocimiento_desde_drive()


    except Exception as error:

        drive_service = None

        drive_session = None

        print(
            "❌ Error inicializando "
            "Google Drive:"
        )

        print(
            error
        )


# ============================================================
# GOOGLE DRIVE — SUBIR / ACTUALIZAR
# ============================================================

def subir_archivo_drive(
    ruta: Path,
    folder_id: str,
    nombre: str | None = None,
):

    if not drive_service:

        return None


    if not ruta.exists():

        print(
            f"⚠️ No existe archivo: {ruta}"
        )

        return None


    nombre_drive = (
        nombre
        or ruta.name
    )


    es_json_maestro = (
        nombre_drive.lower()
        == "penaguillo.json"
    )


    existente = buscar_archivo_drive(

        nombre_drive,

        folder_id,

        solo_json=es_json_maestro,

    )


    extension = (
        ruta.suffix.lower()
    )


    mime_map = {

        ".json": "application/json",

        ".pdf": "application/pdf",

        ".jpg": "image/jpeg",

        ".jpeg": "image/jpeg",

        ".png": "image/png",

        ".webp": "image/webp",

        ".txt": "text/plain",

    }


    mime_type = mime_map.get(

        extension,

        "application/octet-stream",

    )


    try:

        tamaño_local = ruta.stat().st_size


        print(
            "☁️ Preparando archivo para Drive:"
        )


        print(
            f"   📄 Nombre: {nombre_drive}"
        )


        print(
            f"   📏 Tamaño local: "
            f"{tamaño_local} bytes"
        )


        media = MediaFileUpload(

            str(ruta),

            mimetype=mime_type,

            resumable=True,

        )


        if existente:

            archivo = (
                drive_service
                .files()
                .update(

                    fileId=existente["id"],

                    media_body=media,

                    fields=(
                        "id,"
                        "name,"
                        "mimeType,"
                        "size,"
                        "modifiedTime,"
                        "parents"
                    ),

                    supportsAllDrives=True,

                )
                .execute()
            )


            print(
                "☁️ Archivo actualizado en Drive: "
                f"{nombre_drive}"
            )


            return archivo["id"]


        metadata = {

            "name": nombre_drive,

            "mimeType": mime_type,

            "parents": [
                folder_id
            ],

        }


        archivo = (
            drive_service
            .files()
            .create(

                body=metadata,

                media_body=media,

                fields=(
                    "id,"
                    "name,"
                    "mimeType,"
                    "size,"
                    "modifiedTime,"
                    "parents"
                ),

                supportsAllDrives=True,

            )
            .execute()
        )


        print(
            "☁️ Archivo subido a Drive: "
            f"{nombre_drive}"
        )


        return archivo["id"]


    except Exception as error:

        print(
            "❌ Error subiendo archivo "
            "a Drive:"
        )

        print(
            error
        )

        return None


# ============================================================
# GOOGLE DRIVE — DESCARGAR
# ============================================================

def descargar_archivo_drive(
    file_id: str,
    destino: Path,
):

    if not drive_service:

        return False


    if not drive_session:

        print(
            "❌ No existe sesión HTTP "
            "autenticada."
        )

        return False


    try:

        metadata = (
            drive_service
            .files()
            .get(

                fileId=file_id,

                fields=(
                    "id,"
                    "name,"
                    "mimeType,"
                    "size,"
                    "modifiedTime,"
                    "parents,"
                    "shortcutDetails"
                ),

                supportsAllDrives=True,

            )

            .execute()
        )


        print(
            "📋 Archivo Drive:"
        )


        print(
            f"   Nombre: "
            f"{metadata.get('name')}"
        )


        print(
            f"   MIME: "
            f"{metadata.get('mimeType')}"
        )


        print(
            f"   Tamaño Drive: "
            f"{metadata.get('size', 'N/D')} bytes"
        )


        if metadata.get(
            "mimeType"
        ) == (
            "application/vnd.google-apps.shortcut"
        ):

            print(
                "🛑 El archivo es un shortcut."
            )

            return False


        if (
            metadata.get("name")
            == "penaguillo.json"
        ):

            if metadata.get(
                "mimeType"
            ) != "application/json":

                print(
                    "🛑 penaguillo.json no tiene "
                    "MIME application/json."
                )

                return False


        download_url = (

            "https://www.googleapis.com/"
            "drive/v3/files/"
            f"{file_id}"

        )


        response = drive_session.get(

            download_url,

            params={

                "alt": "media",

                "supportsAllDrives": "true",

            },

            timeout=120,

        )


        print(
            "🌐 HTTP status: "
            f"{response.status_code}"
        )


        if response.status_code != 200:

            print(
                response.text[:1000]
            )

            return False


        datos = response.content


        print(
            "📏 Bytes descargados: "
            f"{len(datos)}"
        )


        tamaño_drive = metadata.get(
            "size"
        )


        if tamaño_drive is not None:

            try:

                if int(
                    tamaño_drive
                ) == len(datos):

                    print(
                        "✅ Tamaño descargado "
                        "coincide con Drive."
                    )

                else:

                    print(
                        "⚠️ Tamaños diferentes:"
                    )

                    print(
                        f"   Drive: "
                        f"{tamaño_drive}"
                    )

                    print(
                        f"   Local: "
                        f"{len(datos)}"
                    )

            except (
                TypeError,
                ValueError,
            ):

                pass


        if len(datos) == 0:

            print(
                "🛑 Drive devolvió "
                "un archivo vacío."
            )

            return False


        if (
            metadata.get("name")
            == "penaguillo.json"
        ):

            try:

                contenido_texto = (
                    datos
                    .decode("utf-8")
                )


                json_data = json.loads(
                    contenido_texto
                )


                if not isinstance(
                    json_data,
                    list,
                ):

                    print(
                        "🛑 El JSON maestro "
                        "no contiene una lista."
                    )

                    return False


                if not all(
                    isinstance(
                        item,
                        dict,
                    )
                    for item in json_data
                ):

                    print(
                        "🛑 El JSON maestro "
                        "contiene registros inválidos."
                    )

                    return False


                print(
                    "✅ JSON maestro descargado "
                    "y validado."
                )


                print(
                    "📚 Registros: "
                    f"{len(json_data)}"
                )


            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:

                print(
                    "🛑 JSON maestro inválido:"
                )

                print(
                    error
                )

                return False


        destino.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        temporal_destino = (

            destino.parent

            / (
                f".{destino.name}."
                f"{generar_id()}"
                ".tmp"
            )

        )


        try:

            with open(

                temporal_destino,

                "wb",

            ) as archivo:

                archivo.write(
                    datos
                )

                archivo.flush()

                os.fsync(
                    archivo.fileno()
                )


            os.replace(
                temporal_destino,
                destino,
            )


        finally:

            if temporal_destino.exists():

                try:
                    temporal_destino.unlink()
                except OSError:
                    pass


        print(
            "⬇️ Archivo descargado: "
            f"{destino.name}"
        )


        return True


    except Exception as error:

        print(
            "❌ Error descargando archivo "
            "desde Drive:"
        )

        print(
            error
        )

        return False


# ============================================================
# VALIDAR JSON
# ============================================================

def validar_json_conocimiento(
    ruta: Path,
) -> tuple[bool, list[dict[str, Any]]]:

    if not ruta.exists():

        return False, []


    try:

        tamaño = ruta.stat().st_size


        if tamaño == 0:

            return False, []


        with open(

            ruta,

            "r",

            encoding="utf-8",

        ) as archivo:

            data = json.load(
                archivo
            )


        if not isinstance(
            data,
            list,
        ):

            return False, []


        if not all(
            isinstance(
                item,
                dict,
            )
            for item in data
        ):

            return False, []


        return True, data


    except Exception as error:

        print(
            f"❌ Error validando JSON: "
            f"{error}"
        )

        return False, []


# ============================================================
# DRIVE — SINCRONIZAR DESDE DRIVE
# ============================================================

def sincronizar_conocimiento_desde_drive():

    if not drive_service:

        return


    if not DRIVE_KNOWLEDGE_FOLDER:

        return


    try:

        archivo_drive = (
            buscar_archivo_drive(

                "penaguillo.json",

                DRIVE_KNOWLEDGE_FOLDER,

                solo_json=True,

            )
        )


        if archivo_drive:

            print(
                "☁️ penaguillo.json encontrado "
                "en Google Drive."
            )


            temporal = (

                CONOCIMIENTO_DIR

                / "penaguillo_drive.tmp"

            )


            if temporal.exists():

                try:
                    temporal.unlink()
                except OSError:
                    pass


            descargado = (
                descargar_archivo_drive(

                    archivo_drive["id"],

                    temporal,

                )
            )


            if not descargado:

                print(
                    "🛡️ El conocimiento local "
                    "NO será reemplazado."
                )

                return


            valido, data = (
                validar_json_conocimiento(
                    temporal
                )
            )


            if valido:

                archivo_temporal = (

                    CONOCIMIENTO_DIR

                    / (
                        f"penaguillo_"
                        f"{generar_id()}"
                        ".tmp"
                    )

                )


                try:

                    with open(

                        archivo_temporal,

                        "w",

                        encoding="utf-8",

                    ) as archivo:

                        json.dump(

                            data,

                            archivo,

                            ensure_ascii=False,

                            indent=2,

                        )


                        archivo.flush()

                        os.fsync(
                            archivo.fileno()
                        )


                    os.replace(

                        archivo_temporal,

                        ARCHIVO_CONOCIMIENTO,

                    )


                    print(
                        "✅ Conocimiento descargado "
                        "desde Google Drive."
                    )


                    print(
                        "📚 Registros cargados: "
                        f"{len(data)}"
                    )


                finally:

                    if archivo_temporal.exists():

                        try:
                            archivo_temporal.unlink()
                        except OSError:
                            pass


                    if temporal.exists():

                        try:
                            temporal.unlink()
                        except OSError:
                            pass


                return


            print(
                "🛑 penaguillo.json de Drive "
                "está inválido."
            )


            if temporal.exists():

                try:
                    temporal.unlink()
                except OSError:
                    pass


            return


        print(
            "ℹ️ No existe penaguillo.json "
            "en Google Drive."
        )


        if ARCHIVO_CONOCIMIENTO.exists():

            valido, _ = (
                validar_json_conocimiento(
                    ARCHIVO_CONOCIMIENTO
                )
            )


            if valido:

                sincronizar_conocimiento_a_drive()

                return


        print(
            "ℹ️ No existe conocimiento válido."
        )


    except Exception as error:

        print(
            "❌ Error sincronizando "
            "conocimiento desde Drive:"
        )

        print(
            error
        )


# ============================================================
# DRIVE — SUBIR CONOCIMIENTO
# ============================================================

def sincronizar_conocimiento_a_drive():

    if not drive_service:
        return


    if not DRIVE_KNOWLEDGE_FOLDER:
        return


    if not ARCHIVO_CONOCIMIENTO.exists():
        return


    valido, data = (
        validar_json_conocimiento(
            ARCHIVO_CONOCIMIENTO
        )
    )


    if not valido:

        print(
            "🛑 BLOQUEADO: "
            "no se sube un JSON inválido."
        )

        return


    print(
        "☁️ Sincronizando "
        f"{len(data)} registros a Drive..."
    )


    subir_archivo_drive(

        ARCHIVO_CONOCIMIENTO,

        DRIVE_KNOWLEDGE_FOLDER,

        "penaguillo.json",

    )


# ============================================================
# DRIVE — BACKUP
# ============================================================

def sincronizar_backup_a_drive(
    ruta_backup: Path,
):

    if not drive_service:
        return


    if not DRIVE_BACKUPS_FOLDER:
        return


    subir_archivo_drive(

        ruta_backup,

        DRIVE_BACKUPS_FOLDER,

        ruta_backup.name,

    )


# ============================================================
# DRIVE — PDF
# ============================================================

def sincronizar_pdf_a_drive(
    ruta_pdf: Path,
):

    if not drive_service:
        return


    if not DRIVE_PDF_FOLDER:
        return


    subir_archivo_drive(

        ruta_pdf,

        DRIVE_PDF_FOLDER,

        ruta_pdf.name,

    )


# ============================================================
# DRIVE — IMAGEN
# ============================================================

def sincronizar_imagen_a_drive(
    ruta_imagen: Path,
):

    if not drive_service:
        return


    if not DRIVE_IMAGES_FOLDER:
        return


    subir_archivo_drive(

        ruta_imagen,

        DRIVE_IMAGES_FOLDER,

        ruta_imagen.name,

    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title="Penaguillo IA",

    version="6.1.0",

    description=(
        "Backend del asistente inteligente Penaguillo"
    ),

)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://localhost:5173",

        "http://127.0.0.1:5173",

        "https://penaguillo-1.onrender.com",

    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)


# ============================================================
# ARCHIVOS ESTÁTICOS
# ============================================================

app.mount(

    "/archivos",

    StaticFiles(

        directory=str(
            ARCHIVOS_DIR
        )

    ),

    name="archivos",

)


# ============================================================
# LÍMITES
# ============================================================

MAX_PDF_SIZE = (
    50 * 1024 * 1024
)

MAX_IMAGE_SIZE = (
    15 * 1024 * 1024
)


EXTENSIONES_IMAGEN = {

    ".jpg",
    ".jpeg",
    ".png",
    ".webp",

}


EXTENSIONES_PDF = {

    ".pdf",

}


# ============================================================
# MODELOS
# ============================================================

class ChatMessage(BaseModel):

    role: str

    content: str


class ChatRequest(BaseModel):

    message: str

    history: list[ChatMessage] = []


class EnsenarRequest(BaseModel):

    conocimiento: str


class EliminarRequest(BaseModel):

    id: str


# ============================================================
# NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(
    texto: str,
) -> str:

    if not texto:

        return ""


    texto = str(
        texto
    ).lower()


    reemplazos = {

        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",

    }


    for original, nuevo in reemplazos.items():

        texto = texto.replace(
            original,
            nuevo,
        )


    texto = re.sub(

        r"[^a-z0-9\s]",

        " ",

        texto,

    )


    texto = re.sub(

        r"\s+",

        " ",

        texto,

    )


    return texto.strip()


# ============================================================
# EXTRAER PALABRAS IMPORTANTES
# ============================================================

def extraer_palabras_importantes(
    texto: str,
) -> list[str]:

    texto_normalizado = (
        normalizar_texto(texto)
    )


    palabras = (
        texto_normalizado
        .split()
    )


    resultado = []


    for palabra in palabras:

        if len(palabra) < 3:

            continue


        if palabra in STOPWORDS_ES:

            continue


        if palabra not in resultado:

            resultado.append(
                palabra
            )


    return resultado


# ============================================================
# CONSTRUIR CONSULTA DE RETRIEVAL
# ============================================================

def construir_consulta_retrieval(
    mensaje: str,
    history: list[ChatMessage],
) -> str:
    """Construye una consulta centrada en la pregunta actual."""
    mensaje_actual = str(mensaje or "").strip()

    if not mensaje_actual:
        return ""

    palabras_actuales = extraer_palabras_importantes(mensaje_actual)

    historial_usuario = []
    if history:
        historial_usuario = [
            str(msg.content).strip()
            for msg in history
            if msg.role == "user"
            and str(msg.content).strip()
        ]

    anteriores = historial_usuario[-MAX_MENSAJES_RETRIEVAL:]

    es_amplia = _es_solicitud_amplia(mensaje_actual)
    es_corta = len(palabras_actuales) <= 2 or len(mensaje_actual) <= 12

    # Verificamos si la pregunta actual menciona el área/entidad 
    # o si solo son palabras de intención (ej. "quienes", "conforman", "grupo")
    palabras_intent_amplia = {
        "toda", "todo", "datos", "data", "informacion", "info",
        "completa", "completo", "disponible", "ficha", "perfil",
        "dame", "dime", "sabes", "tienes",
        "equipo", "grupo", "integrantes", "quienes", "conforman",
        "personal", "area", "miembros", "cual", "cuales"
    }
    palabras_entidad = [p for p in palabras_actuales if p not in palabras_intent_amplia]

    if es_corta and anteriores:
        partes = anteriores + [mensaje_actual]
    elif es_amplia:
        # Si pide un equipo pero no dice cuál (palabras_entidad vacío), jalamos el historial
        if not palabras_entidad and anteriores:
            partes = [anteriores[-1], mensaje_actual]
        else:
            partes = [mensaje_actual]
    else:
        partes = [mensaje_actual]
        if anteriores:
            ultimo = anteriores[-1]
            if normalizar_texto(ultimo) != normalizar_texto(mensaje_actual):
                palabras_ultimo = extraer_palabras_importantes(ultimo)
                palabras_compartidas = set(palabras_actuales) & set(palabras_ultimo)
                if palabras_compartidas:
                    partes.insert(0, ultimo)

    consulta = " ".join(partes).strip()

    if len(consulta) > MAX_CHARS_CONSULTA_RETRIEVAL:
        espacio = MAX_CHARS_CONSULTA_RETRIEVAL - len(mensaje_actual) - 1
        if espacio > 0:
            contexto = " ".join(partes[:-1])
            consulta = contexto[-espacio:] + " " + mensaje_actual
        else:
            consulta = mensaje_actual[:MAX_CHARS_CONSULTA_RETRIEVAL]

    print("🧠 Consulta de búsqueda contextual:")
    print(f"   {consulta}")
    print("🎯 Pregunta actual priorizada:")
    print(f"   {mensaje_actual}")

    return consulta


# ============================================================
# RELEVANCIA DE UN REGISTRO
# ============================================================

def _es_solicitud_amplia(pregunta: str) -> bool:
    """Detecta solicitudes de toda la información o miembros de un grupo/equipo."""
    texto = normalizar_texto(str(pregunta or ""))

    patrones = (
        "toda la informacion",
        "toda la info",
        "toda la data",
        "todos los datos",
        "toda la informacion disponible",
        "datos completos",
        "informacion completa",
        "ficha completa",
        "perfil completo",
        "dame todo",
        "dime todo",
        "todo sobre",
        "que sabes de",
        "que informacion tienes de",
        "que datos tienes de",
        # PATRONES PARA EQUIPOS Y GRUPOS
        "equipo",
        "grupo",
        "integrantes",
        "conforman",
        "hacen parte",
        "personal de",
        "area de",
        "miembros",
    )

    return any(patron in texto for patron in patrones)


def _coincidencias_palabras(texto: str, palabras: list[str]) -> int:
    """Cuenta palabras presentes, permitiendo pequeños errores tipográficos (fuzzy matching)."""
    if not texto or not palabras:
        return 0

    total = 0
    # Extraemos palabras únicas del texto para comparar
    palabras_texto = set(re.findall(r'\b\w+\b', texto))

    for palabra in palabras:
        # 1. Intento de coincidencia exacta
        if re.search(rf"\b{re.escape(palabra)}\b", texto):
            total += 1
        else:
            # 2. Búsqueda difusa (tolerancia a errores tipográficos)
            # cutoff=0.8 significa que debe haber un 80% de similitud
            similares = difflib.get_close_matches(palabra, palabras_texto, n=1, cutoff=0.8)
            if similares:
                total += 1
                
    return total


def calcular_relevancia(
    pregunta: str,
    item: dict[str, Any],
    pregunta_actual: str | None = None,
) -> float:
    """Calcula relevancia dando mucha más importancia a la pregunta actual."""
    palabras = extraer_palabras_importantes(pregunta)
    actual = pregunta_actual if pregunta_actual else pregunta
    palabras_actuales = extraer_palabras_importantes(actual)

    if not palabras_actuales:
        palabras_actuales = palabras

    if not palabras:
        return 0.0

    titulo = normalizar_texto(str(item.get("titulo", "")))
    contenido = normalizar_texto(str(item.get("contenido", "")))
    descripcion = normalizar_texto(str(item.get("descripcion", "")))
    tipo = normalizar_texto(str(item.get("tipo", "")))

    texto_completo = " ".join((titulo, contenido, descripcion, tipo))
    pregunta_normalizada = normalizar_texto(actual)

    puntuacion = 0.0
    es_amplia = _es_solicitud_amplia(actual)

    # ------------------------------------------------------------
    # 1. Coincidencias de la pregunta ACTUAL
    # ------------------------------------------------------------
    coincidencias_actuales = _coincidencias_palabras(
        texto_completo,
        palabras_actuales,
    )

    coincidencias_titulo = 0
    coincidencias_descripcion = 0
    coincidencias_contenido = 0

    # Usamos la lógica difusa también para los contadores específicos si no encuentra coincidencia exacta
    palabras_titulo_set = set(re.findall(r'\b\w+\b', titulo))
    palabras_desc_set = set(re.findall(r'\b\w+\b', descripcion))
    palabras_cont_set = set(re.findall(r'\b\w+\b', contenido))

    for palabra in palabras_actuales:
        patron = rf"\b{re.escape(palabra)}\b"
        
        # Titulo
        if re.search(patron, titulo):
            coincidencias_titulo += len(re.findall(patron, titulo))
        elif difflib.get_close_matches(palabra, palabras_titulo_set, n=1, cutoff=0.8):
            coincidencias_titulo += 1
            
        # Descripcion
        if re.search(patron, descripcion):
            coincidencias_descripcion += len(re.findall(patron, descripcion))
        elif difflib.get_close_matches(palabra, palabras_desc_set, n=1, cutoff=0.8):
            coincidencias_descripcion += 1
            
        # Contenido
        if re.search(patron, contenido):
            coincidencias_contenido += len(re.findall(patron, contenido))
        elif difflib.get_close_matches(palabra, palabras_cont_set, n=1, cutoff=0.8):
            coincidencias_contenido += 1

    puntuacion += coincidencias_titulo * 24
    puntuacion += coincidencias_descripcion * 10
    
    # 🔥 Aumentamos el límite de 18 a 40 para que el contenido gane peso
    puntuacion += min(coincidencias_contenido * 4, 40)

    # 🔥 BONIFICACIÓN: Los textos enseñados a mano son "la verdad absoluta", les damos 20 puntos extra.
    if tipo == "texto":
        puntuacion += 20.0

    # ------------------------------------------------------------
    # 2. Coincidencia de frases completas
    # ------------------------------------------------------------
    palabras_frase = [p for p in palabras_actuales if len(p) >= 3]
    if len(palabras_frase) >= 2:
        frase = " ".join(palabras_frase)
        if frase and frase in texto_completo:
            puntuacion += 55

    # ------------------------------------------------------------
    # 3. Cobertura de la pregunta actual
    # ------------------------------------------------------------
    if palabras_actuales:
        cobertura = coincidencias_actuales / len(palabras_actuales)

        if cobertura >= 0.50:
            puntuacion += 12
        if cobertura >= 0.75:
            puntuacion += 20
        if cobertura >= 1.0:
            puntuacion += 35

    # ------------------------------------------------------------
    # 4. Solicitudes Amplias (Toda la data, Equipos, Grupos)
    # ------------------------------------------------------------
    if es_amplia:
        palabras_intent = {
            "toda", "todo", "datos", "data", "informacion", "info",
            "completa", "completo", "disponible", "ficha", "perfil",
            "dame", "dime", "sabes", "tienes",
            "equipo", "grupo", "integrantes", "quienes", "conforman",
            "personal", "area", "miembros", "cual", "cuales"
        }

        entidad = [
            p for p in palabras_actuales if p not in palabras_intent
        ][:4]

        coincidencias_entidad = _coincidencias_palabras(
            texto_completo,
            entidad,
        )

        if coincidencias_entidad >= 1:
            puntuacion += 8
        if coincidencias_entidad >= 2:
            puntuacion += 25
        if coincidencias_entidad >= 3:
            puntuacion += 35

        if _coincidencias_palabras(titulo, entidad) >= 1:
            puntuacion += 15
        if _coincidencias_palabras(descripcion, entidad) >= 1:
            puntuacion += 8

    # ------------------------------------------------------------
    # 5. Historial: solo como apoyo
    # ------------------------------------------------------------
    palabras_contexto = [
        palabra for palabra in palabras
        if palabra not in palabras_actuales
    ]

    for palabra in palabras_contexto:
        patron = rf"\b{re.escape(palabra)}\b"

        if re.search(patron, titulo) or difflib.get_close_matches(palabra, palabras_titulo_set, n=1, cutoff=0.8):
            puntuacion += 4
        if re.search(patron, descripcion) or difflib.get_close_matches(palabra, palabras_desc_set, n=1, cutoff=0.8):
            puntuacion += 2

        # Contamos exacta para el multiplicador
        coincidencias = len(re.findall(patron, texto_completo))
        # Si no hay exacta pero sí fuzzy general, sumamos al menos 1
        if coincidencias == 0 and difflib.get_close_matches(palabra, set(re.findall(r'\b\w+\b', texto_completo)), n=1, cutoff=0.8):
            coincidencias = 1
            
        puntuacion += min(coincidencias * 0.5, 3)

    # ------------------------------------------------------------
    # 6. Una pregunta amplia debe devolver información, no solo
    #    el registro que tenga el mayor score.
    # ------------------------------------------------------------
    if es_amplia and coincidencias_actuales >= 1:
        puntuacion += 6

    return puntuacion


# ============================================================
# CREAR CLAVE REAL DE DUPLICADO
# ============================================================

def clave_unica_conocimiento(
    item: dict[str, Any],
) -> str:

    titulo = normalizar_texto(
        str(
            item.get(
                "titulo",
                "",
            )
        )
    )


    contenido = normalizar_texto(
        str(
            item.get(
                "contenido",
                "",
            )
        )
    )


    descripcion = normalizar_texto(
        str(
            item.get(
                "descripcion",
                "",
            )
        )
    )


    tipo = normalizar_texto(
        str(
            item.get(
                "tipo",
                "",
            )
        )
    )


    material = (

        f"{tipo}|"

        f"{titulo}|"

        f"{contenido}|"

        f"{descripcion}"

    )


    huella = hashlib.sha256(

        material.encode(
            "utf-8"
        )

    ).hexdigest()


    return (
        f"hash:{huella}"
    )


# ============================================================
# BUSCAR CONOCIMIENTO RELEVANTE
# ============================================================

def buscar_conocimiento_relevante(
    pregunta: str,
    conocimientos: list[dict[str, Any]],
    top_k: int = RELEVANCIA_TOP_K,
    pregunta_actual: str | None = None,
) -> list[dict[str, Any]]:
    """Busca y selecciona conocimiento, ampliando resultados para preguntas completas o de equipos."""
    if not conocimientos:
        return []

    actual = pregunta_actual if pregunta_actual else pregunta
    es_amplia = _es_solicitud_amplia(actual)

    resultados = []
    claves_vistas = set()

    palabras_actuales = extraer_palabras_importantes(actual)
    palabras_entidad = [
        p for p in palabras_actuales
        if p not in {
            "toda", "todo", "datos", "data", "informacion", "info",
            "completa", "completo", "disponible", "ficha", "perfil",
            "dame", "dime", "sabes", "tienes",
            "equipo", "grupo", "integrantes", "quienes", "conforman",
            "personal", "area", "miembros", "cual", "cuales"
        }
    ]

    for indice, item in enumerate(conocimientos):
        clave = clave_unica_conocimiento(item)

        if clave in claves_vistas:
            continue

        claves_vistas.add(clave)

        puntuacion = calcular_relevancia(
            pregunta,
            item,
            actual,
        )

        # --------------------------------------------------------
        # Refuerzo de entidad para solicitudes amplias (Equipos/Info)
        # --------------------------------------------------------
        if es_amplia and palabras_entidad:
            texto_item = normalizar_texto(
                " ".join(
                    (
                        str(item.get("titulo", "")),
                        str(item.get("contenido", "")),
                        str(item.get("descripcion", "")),
                        str(item.get("tipo", "")),
                    )
                )
            )

            coincidencias_entidad = _coincidencias_palabras(
                texto_item,
                palabras_entidad,
            )

            if coincidencias_entidad >= 1:
                puntuacion += 10
            if coincidencias_entidad >= 2:
                puntuacion += 20
            if coincidencias_entidad >= 3:
                puntuacion += 30

            titulo = normalizar_texto(str(item.get("titulo", "")))
            if _coincidencias_palabras(titulo, palabras_entidad) >= 1:
                puntuacion += 20

        # Permisivo cuando se busca equipos o perfiles completos
        umbral = 2 if es_amplia else 3

        if puntuacion >= umbral:
            resultados.append((puntuacion, indice, item))

    resultados.sort(
        key=lambda elemento: (
            elemento[0],
            -elemento[1],
        ),
        reverse=True,
    )

    # Aumentado a 20 para soportar equipos más grandes
    limite_resultados = (
        min(len(resultados), max(top_k, 20))
        if es_amplia
        else top_k
    )

    seleccionados = [
        item
        for _, _, item in resultados[:limite_resultados]
    ]

    print("🔎 Búsqueda de conocimiento:")
    print(f"   Registros totales: {len(conocimientos)}")
    print(f"   Registros únicos evaluados: {len(claves_vistas)}")
    print(f"   Solicitud amplia/equipo: {'SÍ' if es_amplia else 'NO'}")
    print(f"   Registros relevantes: {len(seleccionados)}")

    if len(conocimientos) != len(claves_vistas):
        print(
            "♻️ Duplicados ignorados durante retrieval: "
            f"{len(conocimientos) - len(claves_vistas)}"
        )

    if resultados:
        for puntuacion, _, item in resultados[:limite_resultados]:
            print(
                "   📌 "
                f"{item.get('titulo', '')} "
                f"(score={puntuacion:.1f})"
            )
    else:
        print("   ℹ️ No se encontraron coincidencias relevantes.")

    return seleccionados


# ============================================================
# CONSTRUIR CONTEXTO RELEVANTE
# ============================================================

def construir_contexto_relevante(
    pregunta: str,
    conocimientos: list[dict[str, Any]],
    pregunta_actual: str | None = None,
) -> str:

    relevantes = (
        buscar_conocimiento_relevante(

            pregunta,

            conocimientos,

            pregunta_actual=(
                pregunta_actual
            ),

        )
    )


    if not relevantes:

        return (
            "No se encontraron registros "
            "específicos de la base de conocimiento "
            "relacionados con esta pregunta."
        )


    bloques = []

    caracteres_actuales = 0


    for indice, item in enumerate(
        relevantes,
        start=1,
    ):

        tipo = item.get(
            "tipo",
            "desconocido",
        )


        titulo = item.get(
            "titulo",
            "",
        )


        contenido = item.get(
            "contenido",
            "",
        )


        descripcion = item.get(
            "descripcion",
            "",
        )


        bloque = f"""
REGISTRO RELEVANTE {indice}

ID:
{item.get("id", "")}

TIPO:
{tipo}

TÍTULO:
{titulo}

CONTENIDO:
{contenido}

DESCRIPCIÓN:
{descripcion}
""".strip()


        nuevo_total = (

            caracteres_actuales

            + len(
                bloque
            )

        )


        if (
            nuevo_total
            > MAX_CHARS_CONOCIMIENTO_CHAT
        ):

            restante = (

                MAX_CHARS_CONOCIMIENTO_CHAT

                - caracteres_actuales

            )


            if restante > 500:

                bloque_recortado = (

                    bloque[:restante]

                    + "\n[CONTEXTO RECORTADO]"

                )


                bloques.append(
                    bloque_recortado
                )


            break


        bloques.append(
            bloque
        )


        caracteres_actuales = (
            nuevo_total
        )


    contexto = (

        "\n\n"
        "==============================\n\n"

    ).join(
        bloques
    )


    return contexto


# ============================================================
# SYSTEM PROMPT
# ============================================================

def cargar_system_prompt() -> str:

    if not PROMPT_FILE.exists():

        raise RuntimeError(

            "No se encontró "
            f"prompt.txt: {PROMPT_FILE}"

        )


    try:

        with open(

            PROMPT_FILE,

            "r",

            encoding="utf-8",

        ) as archivo:

            contenido = (
                archivo
                .read()
                .strip()
            )


        if not contenido:

            raise RuntimeError(
                "prompt.txt está vacío."
            )


        return contenido


    except OSError as error:

        raise RuntimeError(

            "No fue posible leer "
            f"prompt.txt: {error}"

        ) from error


SYSTEM_PROMPT_BASE = (
    cargar_system_prompt()
)


# ============================================================
# CARGAR CONOCIMIENTO
# ============================================================

def cargar_conocimiento() -> list[dict[str, Any]]:

    if not ARCHIVO_CONOCIMIENTO.exists():

        print(
            "ℹ️ penaguillo.json "
            "no existe localmente."
        )

        return []


    try:

        with open(

            ARCHIVO_CONOCIMIENTO,

            "r",

            encoding="utf-8",

        ) as archivo:

            data = json.load(
                archivo
            )


    except json.JSONDecodeError as error:

        raise RuntimeError(

            "penaguillo.json está corrupto "
            "o contiene JSON inválido."

        ) from error


    except OSError as error:

        raise RuntimeError(

            "No fue posible leer "
            f"penaguillo.json: {error}"

        ) from error


    if not isinstance(
        data,
        list,
    ):

        raise RuntimeError(

            "penaguillo.json debe contener "
            "una lista."

        )


    if not all(

        isinstance(
            item,
            dict,
        )

        for item in data

    ):

        raise RuntimeError(

            "penaguillo.json contiene "
            "registros inválidos."

        )


    print(
        "📚 Conocimiento local cargado: "
        f"{len(data)} registros"
    )


    return data


# ============================================================
# BACKUP
# ============================================================

def crear_backup() -> str | None:

    if not ARCHIVO_CONOCIMIENTO.exists():

        return None


    try:

        valido, data = (
            validar_json_conocimiento(
                ARCHIVO_CONOCIMIENTO
            )
        )


        if not valido:

            return None


        timestamp = (

            datetime.now()

            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )

        )


        backup_path = (

            BACKUP_DIR

            / f"penaguillo_{timestamp}.json"

        )


        archivo_temporal = (

            BACKUP_DIR

            / f"backup_{generar_id()}.tmp"

        )


        try:

            with open(

                archivo_temporal,

                "w",

                encoding="utf-8",

            ) as archivo:

                json.dump(

                    data,

                    archivo,

                    ensure_ascii=False,

                    indent=2,

                )


                archivo.flush()

                os.fsync(
                    archivo.fileno()
                )


            os.replace(

                archivo_temporal,

                backup_path,

            )


            print(
                "🛡️ Backup creado: "
                f"{backup_path.name}"
            )


            sincronizar_backup_a_drive(
                backup_path
            )


            return str(
                backup_path
            )


        finally:

            if archivo_temporal.exists():

                try:
                    archivo_temporal.unlink()
                except OSError:
                    pass


    except OSError as error:

        raise RuntimeError(

            "No se pudo crear backup: "
            f"{error}"

        ) from error


# ============================================================
# GUARDAR CONOCIMIENTO
# ============================================================

def guardar_conocimiento(

    conocimientos: list[dict[str, Any]],

) -> None:

    if not isinstance(
        conocimientos,
        list,
    ):

        raise RuntimeError(
            "El conocimiento debe ser una lista."
        )


    archivo_temporal = (

        CONOCIMIENTO_DIR

        / f"penaguillo_{generar_id()}.tmp"

    )


    try:

        with open(

            archivo_temporal,

            "w",

            encoding="utf-8",

        ) as archivo:

            json.dump(

                conocimientos,

                archivo,

                ensure_ascii=False,

                indent=2,

            )


            archivo.flush()

            os.fsync(
                archivo.fileno()
            )


        os.replace(

            archivo_temporal,

            ARCHIVO_CONOCIMIENTO,

        )


        print(

            "💾 Conocimiento guardado. "

            "Total registros: "

            f"{len(conocimientos)}"

        )


        sincronizar_conocimiento_a_drive()


        if len(conocimientos) > 0:

            crear_backup()


    except OSError as error:

        if archivo_temporal.exists():

            try:
                archivo_temporal.unlink()
            except OSError:
                pass


        raise RuntimeError(

            "No se pudo guardar "
            f"penaguillo.json: {error}"

        ) from error


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    conocimientos = (
        cargar_conocimiento()
    )


    return {

        "ok": True,

        "app": "Penaguillo IA",

        "version": "6.1.0",

        "provider": "Google Gemini Native",

        "chat_model": CHAT_MODEL,

        "vision_model": VISION_MODEL,

        "conocimientos": len(
            conocimientos
        ),

        "retrieval": True,

        "retrieval_top_k": (
            RELEVANCIA_TOP_K
        ),

        "max_context_chars": (
            MAX_CHARS_CONOCIMIENTO_CHAT
        ),

        "max_context_kb": (
            MAX_KB_CONOCIMIENTO_CHAT
        ),

        "max_history_messages": (
            MAX_MENSAJES_HISTORIAL
        ),

        "max_retrieval_messages": (
            MAX_MENSAJES_RETRIEVAL
        ),

        "max_output_tokens": (
            MAX_OUTPUT_TOKENS
        ),

        "duplicate_retrieval_filter": True,

        "current_question_priority": True,

        "gemini_api": (
            bool(GEMINI_API_KEY)
        ),

        "google_drive": (
            drive_service is not None
        ),

        "google_drive_session": (
            drive_session is not None
        ),

        "shared_drive": (
            DRIVE_SHARED_ID is not None
        ),

    }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    data: ChatRequest,
):

    mensaje = (
        data.message
        .strip()
    )


    if not mensaje:

        raise HTTPException(

            status_code=400,

            detail=(
                "El mensaje no puede estar vacío."
            ),

        )


    if not GEMINI_API_KEY:

        raise HTTPException(

            status_code=500,

            detail=(
                "GEMINI_API_KEY "
                "no está configurada."
            ),

        )


    try:

        tiempo_inicio_chat = time.time()


        # ====================================================
        # CARGAR CONOCIMIENTO
        # ====================================================

        conocimientos = (
            cargar_conocimiento()
        )


        # ====================================================
        # RETRIEVAL
        # ====================================================

        consulta_retrieval = (
            construir_consulta_retrieval(

                mensaje,

                data.history,

            )
        )


        conocimiento_relevante = (
            construir_contexto_relevante(

                consulta_retrieval,

                conocimientos,

                pregunta_actual=mensaje,

            )
        )


        # ====================================================
        # SYSTEM PROMPT (CON REGLAS DE NO-INFERENCIA DE EQUIPOS)
        # ====================================================

        system_prompt = (

            SYSTEM_PROMPT_BASE

            + "\n\n"

            + "==============================\n"

            + "INSTRUCCIONES SOBRE LA BASE "
            "DE CONOCIMIENTO\n"

            + "==============================\n"

            + f"DATO IMPORTANTE: Actualmente tienes exactamente {len(conocimientos)} registros en tu base de datos local.\n\n"

            + """

La información que aparece a continuación
es solamente el subconjunto de registros
que el sistema local considera relacionados
con la conversación y la pregunta del usuario.

Debes tomar tú la decisión final sobre qué
información utilizar para responder.

La pregunta ACTUAL del usuario tiene prioridad.

No asumas que todos los registros son relevantes.

Si la información proporcionada no permite
responder con seguridad, dilo claramente.

No inventes información.

Si la pregunta no necesita conocimiento
específico de Penaguillo, puedes responder
normalmente utilizando tus capacidades.

Utiliza el historial de conversación proporcionado
por el sistema para comprender referencias,
pronombres y preguntas de seguimiento.

Si el usuario dice cosas como:

- "esa máquina"
- "el casino"
- "ese teléfono"
- "allí"
- "ellos"
- "esa empresa"
- "¿y cuál?"
- "¿y dónde?"
- "¿y el número?"

debes intentar identificar a qué se refiere
utilizando el contexto de la conversación.

No obligues al usuario a repetir información
que ya proporcionó anteriormente.

Si el contexto de conversación permite saber
a qué se refiere, continúa la conversación
normalmente.

IMPORTANTE:

La pregunta actual que debes responder es
la última pregunta enviada por el usuario.

No reemplaces la pregunta actual por una
palabra aislada del mensaje.

"""

            + "\n\n"

            + "==============================\n"

            + "CONOCIMIENTO RELEVANTE\n"

            + "==============================\n"

            + conocimiento_relevante

            + "\n\n==============================\n"
            + "REGLAS STRICTAS DE NO-INFERENCIA Y EQUIPOS\n"
            + "==============================\n"
            + "1. Queda ABSOLUTAMENTE PROHIBIDO inferir o asumir qué área, equipo o persona maneja un sistema (como ERP SAP, servidores, CRM, etc.) si esa relación exacta no aparece explícitamente en el CONOCIMIENTO RELEVANTE.\n"
            + "2. Si el contexto NO menciona explícitamente qué persona o equipo atiende un sistema o área, debes responder claramente: 'No tengo un responsable confirmado para esa solicitud en la información que manejo.'\n"
            + "3. Si el usuario pregunta por un 'equipo', 'grupo' o 'área' (como Modernización Tecnológica, Optimización, etc.), debes listar ÚNICAMENTE a las personas que el contexto asigne formalmente a ese equipo. Jamás agregues personas de otros registros solo porque el usuario preguntó por un tema genérico antes.\n"
            + "4. Queda PROHIBIDO alterar, acortar o modificar correos electrónicos o teléfonos. Cópialos exactamente iguales a como aparecen en el contexto (mantiene dominios completos como .co y .com).\n"

        )


        print(
            "📤 Enviando pregunta a Gemini."
        )


        print(
            "📚 Contexto seleccionado: "
            f"{len(conocimiento_relevante)} caracteres"
        )


        # ====================================================
        # MENSAJES API
        # ====================================================

        mensajes_api = [

            {

                "role": "system",

                "content": system_prompt,

            }

        ]


        # ====================================================
        # HISTORIAL
        # ====================================================

        if data.history:

            historial_reciente = (
                data.history[
                    -MAX_MENSAJES_HISTORIAL:
                ]
            )


            print(
                "💬 Historial recibido: "
                f"{len(data.history)} mensajes"
            )


            print(
                "💬 Historial enviado a "
                "Gemini: "
                f"{len(historial_reciente)} mensajes"
            )


            for msg in historial_reciente:

                contenido_historial = (

                    str(
                        msg.content
                    )

                    .strip()

                )


                if not contenido_historial:

                    continue


                if msg.role not in (
                    "user",
                    "assistant",
                ):

                    continue


                mensajes_api.append(

                    {

                        "role": msg.role,

                        "content": (
                            contenido_historial
                        ),

                    }

                )


        # ====================================================
        # MENSAJE ACTUAL
        # ====================================================

        mensajes_api.append(

            {

                "role": "user",

                "content": mensaje,

            }

        )


        print(
            "💬 Mensajes totales enviados "
            "a Gemini: "
            f"{len(mensajes_api)}"
        )

        print("🎯 MENSAJE FINAL ENVIADO A GEMINI:")
        print(f"   {mensaje}")


        # ====================================================
        # GEMINI
        # ====================================================

        respuesta = generar_con_gemini(

            model=CHAT_MODEL,

            messages=mensajes_api,

        )


        contenido = (
            extraer_contenido_gemini(
                respuesta
            )
        )


        duracion_total = (
            time.time()
            - tiempo_inicio_chat
        )


        print(
            f"⏱️ Tiempo total /chat: "
            f"{duracion_total:.2f}s"
        )


        return {

            "ok": True,

            "response": contenido or "",

        }


    except GeminiError as error:

        print(
            f"❌ Error Gemini en /chat: "
            f"{error}"
        )


        if (
            error.status_code == 429
            and error.retry_after is not None
        ):

            raise HTTPException(

                status_code=429,

                headers={

                    "Retry-After": str(
                        error.retry_after
                    )

                },

                detail=str(
                    error
                ),

            )


        if error.status_code == 429:

            raise HTTPException(

                status_code=429,

                detail=str(
                    error
                ),

            )


        raise HTTPException(

            status_code=502,

            detail=str(
                error
            ),

        )


    except RuntimeError as error:

        print(
            f"❌ Error en /chat: {error}"
        )


        raise HTTPException(

            status_code=500,

            detail=str(error),

        )


    except Exception as error:

        print(
            f"❌ Error en /chat: {error}"
        )


        texto_error = (
            str(error).upper()
        )


        if (

            "429" in texto_error

            or "RATE LIMIT" in texto_error

            or "RESOURCE_EXHAUSTED" in texto_error

        ):

            raise HTTPException(

                status_code=429,

                detail=(

                    "Gemini alcanzó "
                    "un límite temporal. "
                    "Intenta nuevamente."

                ),

            )


        if (

            "503" in texto_error

            or "UNAVAILABLE" in texto_error

        ):

            raise HTTPException(

                status_code=503,

                detail=(

                    "El proveedor de IA "
                    "está temporalmente "
                    "no disponible."

                ),

            )


        raise HTTPException(

            status_code=500,

            detail=(

                "Error consultando Penaguillo: "

                f"{error}"

            ),

        )


# ============================================================
# ENSEÑAR TEXTO (CON TÍTULO Y DESCRIPCIÓN DINÁMICOS)
# ============================================================

@app.post("/ensenar")
def ensenar(
    data: EnsenarRequest,
):
    texto = data.conocimiento.strip()

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="El conocimiento no puede estar vacío.",
        )

    try:
        conocimientos = cargar_conocimiento()

        # Extraemos las primeras palabras para que el título tenga contexto real
        palabras = texto.split()
        titulo_dinamico = " ".join(palabras[:8])
        if len(palabras) > 8:
            titulo_dinamico += "..."

        nuevo = {
            "id": generar_id(),
            "tipo": "texto",
            "titulo": titulo_dinamico,  # <-- Ya no guarda "Conocimiento manual"
            "contenido": texto,
            "descripcion": f"Información importante sobre: {titulo_dinamico}",  # <-- Ya no queda vacío
            "fecha": ahora_iso(),
        }

        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)

        return {
            "ok": True,
            "mensaje": "Conocimiento guardado correctamente.",
            "conocimiento": nuevo,
            "total": len(conocimientos),
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# IMAGEN BASE64
# ============================================================

def imagen_a_base64(
    ruta: Path,
) -> str:

    with open(
        ruta,
        "rb",
    ) as archivo:

        contenido = archivo.read()


    return base64.b64encode(
        contenido
    ).decode(
        "utf-8"
    )


# ============================================================
# MIME IMAGEN
# ============================================================

def mime_imagen(
    ruta: Path,
) -> str:

    extension = (
        ruta.suffix
        .lower()
    )


    mapa = {

        ".jpg": "image/jpeg",

        ".jpeg": "image/jpeg",

        ".png": "image/png",

        ".webp": "image/webp",

    }


    return mapa.get(

        extension,

        "image/jpeg",

    )


# ============================================================
# VISION — IMAGEN
# ============================================================

def analizar_imagen_con_vision(
    ruta: Path,
    contexto: str = "",
) -> str:

    if not GEMINI_API_KEY:

        raise RuntimeError(

            "GEMINI_API_KEY "
            "no está configurada."

        )


    imagen_bytes = ruta.read_bytes()


    mime = mime_imagen(
        ruta
    )


    imagen_base64 = (
        base64.b64encode(
            imagen_bytes
        ).decode(
            "utf-8"
        )
    )


    prompt = f"""

Analiza cuidadosamente esta imagen para alimentar
la base de conocimiento de Penaguillo.

Tu respuesta debe ser útil para una empresa de
maquinaria agrícola y procesamiento de café.

Identifica cuando sea posible:

- Qué aparece en la imagen.
- Equipos o máquinas.
- Nombre o referencia visible.
- Textos visibles.
- Etiquetas.
- Números.
- Componentes.
- Diagramas.
- Tablas.
- Características técnicas visibles.
- Procesos.
- Información relevante para Penagos.

Si hay texto en la imagen,
transcríbelo claramente.

NO inventes información que no puedas observar.

Contexto proporcionado por el usuario:

{contexto}

Devuelve una descripción estructurada y detallada.

"""


    try:

        respuesta = generar_con_gemini(

            model=VISION_MODEL,

            messages=[

                {

                    "role": "user",

                    "content": [

                        {

                            "type": "text",

                            "text": prompt,

                        },

                        {

                            "type": "image_url",

                            "image_url": {

                                "url":
                                f"data:{mime};base64,"
                                f"{imagen_base64}",

                            },

                        },

                    ],

                }

            ],

        )


        return (
            extraer_contenido_gemini(
                respuesta
            )
        )


    except Exception as error:

        print(
            f"❌ Error Gemini Vision: "
            f"{error}"
        )


        raise RuntimeError(

            "No fue posible analizar "
            f"la imagen: {error}"

        ) from error


# ============================================================
# ENSEÑAR IMAGEN
# ============================================================

@app.post("/ensenar-imagen")
async def ensenar_imagen(

    file: UploadFile = File(...),

):

    nombre_original = (

        file.filename

        or "imagen"

    )


    extension = (

        Path(nombre_original)

        .suffix

        .lower()

    )


    if extension not in EXTENSIONES_IMAGEN:

        raise HTTPException(

            status_code=400,

            detail=(

                "Formato de imagen no permitido. "

                "Usa JPG, JPEG, PNG o WEBP."

            ),

        )


    contenido = await file.read()


    if len(contenido) > MAX_IMAGE_SIZE:

        raise HTTPException(

            status_code=400,

            detail=(
                "La imagen supera 15 MB."
            ),

        )


    nombre = (

        f"{uuid.uuid4().hex}_"

        f"{nombre_seguro(nombre_original)}"

    )


    ruta = (

        IMAGENES_DIR

        / nombre

    )


    try:

        with open(

            ruta,

            "wb",

        ) as archivo:

            archivo.write(
                contenido
            )


        descripcion = (
            analizar_imagen_con_vision(
                ruta
            )
        )


        conocimientos = (
            cargar_conocimiento()
        )


        nuevo = {

            "id": generar_id(),

            "tipo": "imagen",

            "titulo": nombre_original,

            "contenido": descripcion,

            "descripcion": descripcion,

            "archivo": (
                f"/archivos/imagenes/{nombre}"
            ),

            "nombre_archivo": nombre_original,

            "fecha": ahora_iso(),

        }


        conocimientos.append(
            nuevo
        )


        guardar_conocimiento(
            conocimientos
        )


        sincronizar_imagen_a_drive(
            ruta
        )


        return {

            "ok": True,

            "mensaje": (
                "Imagen aprendida "
                "correctamente."
            ),

            "conocimiento": nuevo,

            "total": len(
                conocimientos
            ),

        }


    except RuntimeError as error:

        if ruta.exists():

            try:
                ruta.unlink()
            except OSError:
                pass


        raise HTTPException(

            status_code=500,

            detail=str(error),

        )


    except Exception as error:

        print(
            f"❌ Error /ensenar-imagen: "
            f"{error}"
        )


        if ruta.exists():

            try:
                ruta.unlink()
            except OSError:
                pass


        raise HTTPException(

            status_code=500,

            detail=(

                "Error procesando imagen: "

                f"{error}"

            ),

        )


# ============================================================
# PDF — CONVERTIR PÁGINA A BYTES PNG
# ============================================================

def pdf_a_imagen_bytes(
    pagina: fitz.Page,
) -> bytes:

    matriz = fitz.Matrix(
        1.5,
        1.5,
    )


    pixmap = pagina.get_pixmap(

        matrix=matriz,

        alpha=False,

    )


    return pixmap.tobytes(
        "png"
    )


# ============================================================
# PDF — VISION CON GEMINI
# ============================================================

def analizar_pagina_pdf_con_vision(
    pagina: fitz.Page,
    numero_pagina: int,
) -> str:

    if not GEMINI_API_KEY:

        raise RuntimeError(

            "GEMINI_API_KEY "
            "no está configurada."

        )


    imagen_bytes = (
        pdf_a_imagen_bytes(
            pagina
        )
    )


    imagen_base64 = (
        base64.b64encode(
            imagen_bytes
        ).decode(
            "utf-8"
        )
    )


    prompt = f"""

Analiza esta página de un PDF para alimentar
la base de conocimiento de Penaguillo.

Es la página {numero_pagina}.

Extrae cuidadosamente:

- Todo texto legible.
- Títulos.
- Subtítulos.
- Tablas.
- Números.
- Referencias de productos.
- Especificaciones técnicas.
- Diagramas.
- Procesos.
- Datos importantes.
- Información comercial.
- Información relacionada con maquinaria agrícola,
café o Penagos.

Si existe una tabla,
intenta conservar su estructura.

No inventes información.

Devuelve todo lo útil de esta página
en texto estructurado.

"""


    try:

        respuesta = generar_con_gemini(

            model=VISION_MODEL,

            messages=[

                {

                    "role": "user",

                    "content": [

                        {

                            "type": "text",

                            "text": prompt,

                        },

                        {

                            "type": "image_url",

                            "image_url": {

                                "url":
                                "data:image/png;base64,"
                                f"{imagen_base64}",

                            },

                        },

                    ],

                }

            ],

        )


        return (
            extraer_contenido_gemini(
                respuesta
            )
        )


    except Exception as error:

        raise RuntimeError(

            f"Error analizando página "
            f"{numero_pagina}: {error}"

        ) from error


# ============================================================
# EXTRAER TEXTO PDF
# ============================================================

def extraer_texto_pdf(
    documento: fitz.Document,
) -> str:

    paginas = []


    for pagina in documento:

        texto = pagina.get_text(

            "text",

            sort=True,

        )


        if texto.strip():

            paginas.append(
                texto.strip()
            )


    return "\n\n".join(
        paginas
    )


# ============================================================
# ENSEÑAR PDF
# ============================================================

@app.post("/ensenar-pdf")
async def ensenar_pdf(

    file: UploadFile = File(...),

):

    nombre_original = (

        file.filename

        or "documento.pdf"

    )


    extension = (

        Path(nombre_original)

        .suffix

        .lower()

    )


    if extension not in EXTENSIONES_PDF:

        raise HTTPException(

            status_code=400,

            detail=(
                "El archivo debe ser un PDF."
            ),

        )


    contenido = await file.read()


    if len(contenido) > MAX_PDF_SIZE:

        raise HTTPException(

            status_code=400,

            detail=(

                "El PDF supera "
                "el límite de 50 MB."

            ),

        )


    nombre = (

        f"{uuid.uuid4().hex}_"

        f"{nombre_seguro(nombre_original)}"

    )


    ruta = (
        PDF_DIR
        / nombre
    )


    documento = None


    try:

        with open(

            ruta,

            "wb",

        ) as archivo:

            archivo.write(
                contenido
            )


        documento = fitz.open(
            str(ruta)
        )


        numero_paginas = (
            documento.page_count
        )


        texto_extraido = (
            extraer_texto_pdf(
                documento
            )
        )


        es_escaneado = (

            len(

                re.sub(

                    r"\s+",

                    "",

                    texto_extraido,

                )

            )

            < 50

        )


        contenido_final = ""


        if not es_escaneado:

            contenido_final = (
                texto_extraido
            )


        else:

            paginas_vision = []


            for indice, pagina in enumerate(

                documento,

                start=1,

            ):

                print(

                    "🔎 Analizando PDF — "

                    f"página "

                    f"{indice}/{numero_paginas}"

                )


                texto_pagina = (

                    analizar_pagina_pdf_con_vision(

                        pagina,

                        indice,

                    )

                )


                if texto_pagina.strip():

                    paginas_vision.append(

                        (

                            "==============================\n"

                            f"PÁGINA {indice}\n"

                            "==============================\n\n"

                            f"{texto_pagina}"

                        )

                    )


            contenido_final = (

                "\n\n".join(
                    paginas_vision
                )

            )


        documento.close()

        documento = None


        if not contenido_final.strip():

            raise RuntimeError(

                "No fue posible extraer "
                "información del PDF."

            )


        conocimientos = (
            cargar_conocimiento()
        )


        nuevo = {

            "id": generar_id(),

            "tipo": "pdf",

            "titulo": nombre_original,

            "contenido": contenido_final,

            "descripcion": (

                "Documento PDF procesado "
                "por Penaguillo."

            ),

            "archivo": (

                f"/archivos/pdf/{nombre}"

            ),

            "nombre_archivo": nombre_original,

            "numero_paginas": (
                numero_paginas
            ),

            "metodo": (

                "vision"

                if es_escaneado

                else "texto"

            ),

            "fecha": ahora_iso(),

        }


        conocimientos.append(
            nuevo
        )


        guardar_conocimiento(
            conocimientos
        )


        sincronizar_pdf_a_drive(
            ruta
        )


        return {

            "ok": True,

            "mensaje": (
                "PDF aprendido "
                "correctamente."
            ),

            "conocimiento": nuevo,

            "total": len(
                conocimientos
            ),

            "numero_paginas": (
                numero_paginas
            ),

            "metodo": (

                "vision"

                if es_escaneado

                else "texto"

            ),

        }


    except RuntimeError as error:

        if documento is not None:

            try:
                documento.close()
            except Exception:
                pass


        if ruta.exists():

            try:
                ruta.unlink()
            except OSError:
                pass


        raise HTTPException(

            status_code=500,

            detail=str(error),

        )


    except Exception as error:

        if documento is not None:

            try:
                documento.close()
            except Exception:
                pass


        if ruta.exists():

            try:
                ruta.unlink()
            except OSError:
                pass


        print(

            f"❌ Error /ensenar-pdf: "
            f"{error}"

        )


        raise HTTPException(

            status_code=500,

            detail=(

                "Error procesando PDF: "

                f"{error}"

            ),

        )


# ============================================================
# OBTENER CONOCIMIENTO
# ============================================================

@app.get("/conocimiento")
def obtener_conocimiento():

    try:

        conocimientos = (
            cargar_conocimiento()
        )


        return {

            "ok": True,

            "total": len(
                conocimientos
            ),

            "conocimientos": (
                conocimientos
            ),

        }


    except RuntimeError as error:

        raise HTTPException(

            status_code=500,

            detail=str(error),

        )


# ============================================================
# ELIMINAR CONOCIMIENTO
# ============================================================

@app.delete("/conocimiento")
def eliminar_conocimiento(

    data: EliminarRequest,

):

    try:

        conocimientos = (
            cargar_conocimiento()
        )


        encontrado = None


        for item in conocimientos:

            if str(

                item.get(
                    "id",
                    "",
                )

            ) == str(
                data.id
            ):

                encontrado = item

                break


        if encontrado is None:

            raise HTTPException(

                status_code=404,

                detail=(

                    "No se encontró "
                    "ese conocimiento."

                ),

            )


        nuevos_conocimientos = [

            item

            for item in conocimientos

            if str(

                item.get(
                    "id",
                    "",
                )

            )

            != str(
                data.id
            )

        ]


        guardar_conocimiento(
            nuevos_conocimientos
        )


        archivo_relativo = (
            encontrado.get(
                "archivo"
            )
        )


        if archivo_relativo:

            archivo_relativo = (

                archivo_relativo

                .replace(
                    "/archivos/",
                    "",
                    1,
                )

                .lstrip("/")

            )


            ruta_archivo = (

                ARCHIVOS_DIR
                / archivo_relativo

            )


            if ruta_archivo.exists():

                try:

                    ruta_archivo.unlink()

                except OSError as error:

                    print(

                        "⚠️ No se pudo "
                        "eliminar archivo: "
                        f"{error}"

                    )


        return {

            "ok": True,

            "mensaje": (

                "Conocimiento eliminado "
                "correctamente."

            ),

            "eliminado": encontrado,

            "total": len(
                nuevos_conocimientos
            ),

        }


    except HTTPException:

        raise


    except RuntimeError as error:

        raise HTTPException(

            status_code=500,

            detail=str(error),

        )


# ============================================================
# BACKUPS
# ============================================================

@app.get("/backups")
def listar_backups():

    archivos = sorted(

        BACKUP_DIR.glob(
            "penaguillo_*.json"
        ),

        key=lambda archivo:
            archivo.stat().st_mtime,

        reverse=True,

    )


    return {

        "ok": True,

        "total": len(
            archivos
        ),

        "backups": [

            {

                "nombre": archivo.name,

                "fecha": (

                    datetime

                    .fromtimestamp(

                        archivo.stat().st_mtime

                    )

                    .isoformat()

                ),

            }

            for archivo in archivos

        ]

    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        "🚀 Iniciando Penaguillo IA..."
    )


    print(
        f"📂 BASE_DIR: {BASE_DIR}"
    )


    print(
        "📂 CONOCIMIENTO_DIR: "
        f"{CONOCIMIENTO_DIR}"
    )


    print(
        "🌐 RENDER: "
        f"{os.getenv('RENDER')}"
    )


    print(
        "🤖 PROVEEDOR IA: "
        "Gemini Nativo"
    )


    print(
        "🤖 CHAT_MODEL: "
        f"{CHAT_MODEL}"
    )


    print(
        "👁️ VISION_MODEL: "
        f"{VISION_MODEL}"
    )


    print(
        "🔎 RETRIEVAL TOP K: "
        f"{RELEVANCIA_TOP_K}"
    )


    print(
        "💬 MAX HISTORIAL GEMINI: "
        f"{MAX_MENSAJES_HISTORIAL} mensajes"
    )


    print(
        "🧠 MAX HISTORIAL RETRIEVAL: "
        f"{MAX_MENSAJES_RETRIEVAL} mensajes"
    )


    print(
        "📦 MAX CONTEXT: "
        f"{MAX_KB_CONOCIMIENTO_CHAT} KB"
    )


    print(
        "🤖 MAX OUTPUT TOKENS OBJETIVO: "
        f"{MAX_OUTPUT_TOKENS}"
    )


    print(
        "♻️ DEDUPLICACIÓN RETRIEVAL: "
        "ACTIVADA"
    )


    print(
        "🎯 PRIORIDAD PREGUNTA ACTUAL: "
        "ACTIVADA"
    )


    print(
        "⚡ OPTIMIZACIÓN DE LATENCIA: "
        "ACTIVADA"
    )


    print(
        "🔑 GEMINI_API: "
        f"{'CONFIGURADO' if GEMINI_API_KEY else 'NO CONFIGURADO'}"
    )


    # ========================================================
    # GOOGLE DRIVE
    # ========================================================

    inicializar_google_drive()


    # ========================================================
    # ESTADO FINAL
    # ========================================================

    try:

        conocimientos = (
            cargar_conocimiento()
        )


        print(

            "📚 Conocimientos disponibles: "

            f"{len(conocimientos)}"

        )


    except Exception as error:

        print(

            "⚠️ No se pudo cargar "
            f"el conocimiento: {error}"

        )


    print(
        "✅ Penaguillo IA iniciado."
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000,

        reload=False,

    )