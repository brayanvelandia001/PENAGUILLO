# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 5.8
#
# PROVEEDOR DE IA:
# - OpenRouter
# - Modelo configurable mediante OPENROUTER_MODEL
#
# FUNCIONES:
# - Chat con Penaguillo (CON HISTORIAL)
# - Chat con búsqueda contextual
# - Enseñar texto
# - Enseñar imágenes
# - Enseñar PDF
# - PDF con texto seleccionable -> PyMuPDF
# - PDF escaneado -> OpenRouter Vision
# - Imágenes -> OpenRouter Vision
# - Persistencia local
# - Google Drive como almacenamiento permanente en Render
# - penaguillo.json como FUENTE MAESTRA de conocimiento
# - Backups opcionales después de cada cambio
# - Escritura atómica
# - Búsqueda local por relevancia
# - Deduplicación inteligente durante retrieval
# - max_tokens objetivo: 3000
# - Una sola adaptación de tokens por créditos
# - Manejo controlado de in_flight_budget_exhausted
#
# CORRECCIONES V5.8:
#
# - La pregunta actual tiene mayor peso en retrieval.
# - El historial anterior ya no domina la búsqueda.
# - Se utilizan como máximo 2 preguntas anteriores para
#   contexto de retrieval.
# - Los duplicados reales se eliminan únicamente durante
#   retrieval, incluso si tienen IDs diferentes.
# - max_tokens objetivo permanece en 3000.
# - Se permite UNA adaptación si OpenRouter informa un
#   límite menor por créditos.
# - No se encadenan adaptaciones 3000 -> 2875 -> 2352.
# - in_flight_budget_exhausted se maneja de forma controlada.
# - Google Drive permanece sin cambios.
# - Enseñar texto / imagen / PDF permanece sin cambios.
# ============================================================

import base64
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
# OPENROUTER
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemma-2-9b-it:free",
)

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


if not OPENROUTER_API_KEY:

    print(
        "⚠️ ADVERTENCIA: "
        "OPENROUTER_API_KEY no está configurada."
    )

else:

    print(
        "🔑 OPENROUTER_API_KEY: configurada"
    )


# ============================================================
# MODELOS
# ============================================================

CHAT_MODEL = OPENROUTER_MODEL
VISION_MODEL = OPENROUTER_MODEL


# ============================================================
# CONFIGURACIÓN DE TOKENS
# ============================================================

MAX_OUTPUT_TOKENS = 3000

MIN_OUTPUT_TOKENS = 256


# ============================================================
# CONFIGURACIÓN DEL RETRIEVAL LOCAL
# ============================================================

RELEVANCIA_TOP_K = 5

MAX_KB_CONOCIMIENTO_CHAT = 30

MAX_CHARS_CONOCIMIENTO_CHAT = (
    MAX_KB_CONOCIMIENTO_CHAT * 1024
)


# ============================================================
# CONFIGURACIÓN DEL HISTORIAL
# ============================================================

MAX_MENSAJES_HISTORIAL = 10


# ============================================================
# CONFIGURACIÓN DE BÚSQUEDA CONTEXTUAL
# ============================================================

# Se redujo de 6 a 2 para que las conversaciones anteriores
# no contaminen demasiado la búsqueda actual.

MAX_MENSAJES_RETRIEVAL = 2

MAX_CHARS_CONSULTA_RETRIEVAL = 2500


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS_ES = {

    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuales",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "dos",
    "el",
    "ella",
    "ellas",
    "ello",
    "ellos",
    "en",
    "entre",
    "era",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estas",
    "este",
    "esto",
    "estos",
    "fue",
    "ha",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "más",
    "me",
    "mi",
    "mis",
    "muy",
    "no",
    "nos",
    "o",
    "para",
    "pero",
    "por",
    "que",
    "qué",
    "se",
    "sea",
    "si",
    "sí",
    "sin",
    "sobre",
    "son",
    "su",
    "sus",
    "también",
    "te",
    "tener",
    "ti",
    "tu",
    "tus",
    "un",
    "una",
    "unas",
    "uno",
    "unos",
    "y",
    "ya",
    "yo",

}


# ============================================================
# ERROR CONTROLADO DE OPENROUTER
# ============================================================

class OpenRouterError(RuntimeError):

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
# OPENROUTER — EXTRAER TOKENS DISPONIBLES DEL ERROR 402
# ============================================================

def extraer_tokens_disponibles(
    mensaje_error: str,
) -> int | None:

    if not mensaje_error:

        return None


    patrones = [

        r"can only afford\s+(\d+)",

        r"only afford\s+(\d+)",

        r"puede pagar\s+(\d+)",

        r"solo puede pagar\s+(\d+)",

        r"can afford\s+(\d+)",

        r"afford\s+(\d+)\s+tokens",

    ]


    for patron in patrones:

        coincidencia = re.search(

            patron,

            mensaje_error,

            flags=re.IGNORECASE,

        )


        if coincidencia:

            try:

                tokens = int(
                    coincidencia.group(1)
                )


                if tokens >= MIN_OUTPUT_TOKENS:

                    return tokens


            except (
                TypeError,
                ValueError,
            ):

                pass


    return None


# ============================================================
# OPENROUTER — EXTRAER RETRY-AFTER
# ============================================================

def extraer_retry_after(
    respuesta: requests.Response,
    mensaje_error: str,
) -> int | None:

    # Primero intentamos leer el header real.

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


    # Algunas respuestas de OpenRouter incluyen el dato
    # dentro de metadata.headers.

    patrones = [

        r'"Retry-After"\s*:\s*"(\d+)"',

        r'"retry-after"\s*:\s*"(\d+)"',

        r"Retry-After[\"']?\s*:\s*[\"']?(\d+)",

    ]


    for patron in patrones:

        coincidencia = re.search(
            patron,
            mensaje_error,
            flags=re.IGNORECASE,
        )


        if coincidencia:

            try:

                segundos = int(
                    coincidencia.group(1)
                )

                if segundos >= 0:

                    return segundos

            except (
                TypeError,
                ValueError,
            ):

                pass


    return None


# ============================================================
# OPENROUTER — GENERAR RESPUESTA
# ============================================================

def generar_con_openrouter(
    *,
    model: str,
    messages: list,
    max_retries: int = 3,
):

    if not OPENROUTER_API_KEY:

        raise OpenRouterError(
            "OPENROUTER_API_KEY no está configurada."
        )


    headers = {

        "Authorization":
            f"Bearer {OPENROUTER_API_KEY}",

        "Content-Type":
            "application/json",

        "HTTP-Referer":
            "https://penaguillo-1.onrender.com",

        "X-Title":
            "Penaguillo IA",

    }


    # ========================================================
    # TOKENS
    # ========================================================

    max_tokens_actual = MAX_OUTPUT_TOKENS

    # IMPORTANTE:
    #
    # Solamente permitimos UNA adaptación.
    #
    # No hacemos:
    #
    # 3000 -> 2875 -> 2352 -> ...
    #
    # Esto evita consumir solicitudes adicionales y evitar
    # que una misma consulta termine empeorando el presupuesto.

    ajuste_por_creditos = False

    ultimo_error = None


    # ========================================================
    # REINTENTOS
    # ========================================================

    for intento in range(
        1,
        max_retries + 1,
    ):

        try:

            inicio = time.time()


            payload = {

                "model": model,

                "messages": messages,

                "max_tokens": max_tokens_actual,

            }


            print(
                f"🤖 OpenRouter -> "
                f"modelo={model}, "
                f"intento={intento}/{max_retries}, "
                f"max_tokens={max_tokens_actual}"
            )


            respuesta = requests.post(

                OPENROUTER_URL,

                headers=headers,

                json=payload,

                timeout=120,

            )


            duracion = (
                time.time()
                - inicio
            )


            print(
                f"🌐 OpenRouter HTTP: "
                f"{respuesta.status_code}"
            )


            print(
                f"⏱️ OpenRouter respondió "
                f"en {duracion:.2f}s"
            )


            if respuesta.status_code == 200:

                try:

                    datos = respuesta.json()

                except ValueError as error:

                    raise OpenRouterError(
                        "OpenRouter devolvió "
                        "una respuesta que no es JSON.",
                        status_code=200,
                    ) from error


                if ajuste_por_creditos:

                    print(
                        "✅ Solicitud adaptada "
                        "una sola vez a "
                        f"{max_tokens_actual} tokens."
                    )


                return datos


            mensaje_error = (
                respuesta.text[:5000]
            )


            ultimo_error = OpenRouterError(

                "OpenRouter HTTP "
                f"{respuesta.status_code}: "
                f"{mensaje_error}",

                status_code=respuesta.status_code,

            )


            print(
                "⚠️ OpenRouter falló:"
            )


            print(
                mensaje_error
            )


            # =================================================
            # 402 — CRÉDITOS
            # =================================================

            if respuesta.status_code == 402:

                texto_error_lower = (
                    mensaje_error.lower()
                )


                # ---------------------------------------------
                # IN-FLIGHT BUDGET
                # ---------------------------------------------

                if (
                    "in_flight_budget_exhausted"
                    in texto_error_lower
                    or
                    "current in-flight requests"
                    in texto_error_lower
                ):

                    retry_after = (
                        extraer_retry_after(
                            respuesta,
                            mensaje_error,
                        )
                    )


                    print(
                        "⏳ OpenRouter indica que "
                        "existen solicitudes en vuelo."
                    )


                    if retry_after is not None:

                        print(
                            "⏱️ Retry-After: "
                            f"{retry_after}s"
                        )


                    raise OpenRouterError(

                        "OpenRouter está esperando "
                        "que finalicen solicitudes anteriores. "
                        + (
                            f"Intenta nuevamente en "
                            f"{retry_after} segundos."
                            if retry_after is not None
                            else
                            "Intenta nuevamente en unos segundos."
                        ),

                        status_code=429,

                        retry_after=retry_after,

                    )


                # ---------------------------------------------
                # CRÉDITOS NORMALES
                # ---------------------------------------------

                tokens_disponibles = (
                    extraer_tokens_disponibles(
                        mensaje_error
                    )
                )


                if (
                    tokens_disponibles is not None
                    and not ajuste_por_creditos
                    and tokens_disponibles
                    < max_tokens_actual
                    and tokens_disponibles
                    >= MIN_OUTPUT_TOKENS
                ):

                    print(
                        "💰 Créditos insuficientes "
                        "para el límite objetivo."
                    )


                    print(
                        "🔄 ÚNICO ajuste automático:"
                    )


                    print(
                        f"   Objetivo: "
                        f"{max_tokens_actual}"
                    )


                    print(
                        f"   Disponible: "
                        f"{tokens_disponibles}"
                    )


                    max_tokens_actual = (
                        tokens_disponibles
                    )


                    ajuste_por_creditos = True


                    print(
                        "➡️ Se realizará "
                        "una sola nueva solicitud "
                        f"con {max_tokens_actual} tokens."
                    )


                    continue


                # ---------------------------------------------
                # SEGUNDO 402
                # ---------------------------------------------

                print(
                    "🛑 OpenRouter no permite "
                    "otra adaptación de tokens."
                )


                raise ultimo_error


            # =================================================
            # ERRORES TEMPORALES
            # =================================================

            texto_error = (
                mensaje_error.upper()
            )


            es_temporal = any(

                codigo in texto_error

                for codigo in (

                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "TIMEOUT",
                    "UNAVAILABLE",
                    "RATE LIMIT",
                    "RESOURCE_EXHAUSTED",

                )

            )


            if (
                not es_temporal
                or intento >= max_retries
            ):

                raise ultimo_error


            espera = (
                2 ** intento
            )


            print(
                f"⏳ Error temporal. "
                f"Reintentando en "
                f"{espera}s..."
            )


            time.sleep(
                espera
            )


        except requests.RequestException as error:

            ultimo_error = OpenRouterError(

                "No fue posible conectar "
                f"con OpenRouter: {error}",

            )


            print(
                "⚠️ Error de conexión "
                "con OpenRouter:"
            )


            print(
                error
            )


            if intento >= max_retries:

                raise ultimo_error


            espera = (
                2 ** intento
            )


            print(
                f"⏳ Reintentando en "
                f"{espera}s..."
            )


            time.sleep(
                espera
            )


    if ultimo_error:

        raise ultimo_error


    raise OpenRouterError(
        "OpenRouter no pudo generar una respuesta."
    )


# ============================================================
# OPENROUTER — EXTRAER CONTENIDO
# ============================================================

def extraer_contenido_openrouter(
    respuesta: dict,
) -> str:

    if not respuesta:

        raise RuntimeError(
            "OpenRouter no devolvió respuesta."
        )


    choices = respuesta.get(
        "choices",
        [],
    )


    if not choices:

        error_data = respuesta.get(
            "error"
        )


        if error_data:

            raise RuntimeError(
                "OpenRouter devolvió un error: "
                f"{error_data}"
            )


        raise RuntimeError(
            "OpenRouter no devolvió ninguna opción."
        )


    primer_choice = choices[0]


    if not isinstance(
        primer_choice,
        dict,
    ):

        raise RuntimeError(
            "Respuesta inválida de OpenRouter."
        )


    mensaje = primer_choice.get(
        "message",
        {},
    )


    if not isinstance(
        mensaje,
        dict,
    ):

        raise RuntimeError(
            "OpenRouter devolvió "
            "un mensaje inválido."
        )


    contenido = mensaje.get(
        "content"
    )


    if isinstance(
        contenido,
        str,
    ):

        if contenido.strip():

            return contenido.strip()


    if isinstance(
        contenido,
        list,
    ):

        partes = []


        for parte in contenido:

            if not isinstance(
                parte,
                dict,
            ):

                continue


            texto = parte.get(
                "text"
            )


            if texto:

                partes.append(
                    str(texto)
                )


        resultado = (
            "\n".join(
                partes
            )
            .strip()
        )


        if resultado:

            return resultado


    raise RuntimeError(
        "OpenRouter no devolvió "
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

    version="5.8.0",

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

    mensaje_actual = (
        str(mensaje)
        .strip()
    )


    if not mensaje_actual:

        return ""


    # ========================================================
    # IMPORTANTE
    #
    # La pregunta actual es la fuente principal.
    #
    # Solo agregamos las últimas 2 preguntas del usuario
    # para resolver referencias como:
    #
    # "¿y cuál?"
    # "¿y dónde?"
    # "¿y esa máquina?"
    #
    # No concatenamos todo el historial.
    # ========================================================

    historial_usuario = []


    if history:

        historial_usuario = [

            str(msg.content).strip()

            for msg in history

            if msg.role == "user"

            and str(msg.content).strip()

        ]


    anteriores = historial_usuario[
        -MAX_MENSAJES_RETRIEVAL:
    ]


    # Si la pregunta actual es suficientemente descriptiva,
    # la búsqueda utiliza principalmente esa pregunta.

    palabras_actuales = (
        extraer_palabras_importantes(
            mensaje_actual
        )
    )


    partes = []


    if palabras_actuales:

        partes.append(
            mensaje_actual
        )


        # Las preguntas anteriores solamente aportan
        # contexto adicional.

        for anterior in anteriores:

            if normalizar_texto(
                anterior
            ) == normalizar_texto(
                mensaje_actual
            ):

                continue


            partes.append(
                anterior
            )

    else:

        # Si la pregunta actual es algo como:
        # "¿y cuál?"
        #
        # necesitamos el contexto anterior.

        partes.extend(
            anteriores
        )

        partes.append(
            mensaje_actual
        )


    consulta = " ".join(
        partes
    )


    if len(consulta) > MAX_CHARS_CONSULTA_RETRIEVAL:

        # Conservamos SIEMPRE la pregunta actual.

        consulta_actual_normalizada = (
            mensaje_actual
        )


        espacio_disponible = (

            MAX_CHARS_CONSULTA_RETRIEVAL

            - len(consulta_actual_normalizada)

            - 1

        )


        if espacio_disponible > 0:

            contexto_anterior = " ".join(
                anteriores
            )


            contexto_anterior = (
                contexto_anterior[
                    -espacio_disponible:
                ]
            )


            consulta = (

                contexto_anterior

                + " "

                + consulta_actual_normalizada

            )

        else:

            consulta = (
                consulta_actual_normalizada[
                    -MAX_CHARS_CONSULTA_RETRIEVAL:
                ]
            )


    print(
        "🧠 Consulta de búsqueda contextual:"
    )


    print(
        f"   {consulta}"
    )


    print(
        "🎯 Pregunta actual priorizada:"
    )


    print(
        f"   {mensaje_actual}"
    )


    return consulta


# ============================================================
# RELEVANCIA DE UN REGISTRO
# ============================================================

def calcular_relevancia(
    pregunta: str,
    item: dict[str, Any],
    pregunta_actual: str | None = None,
) -> float:

    palabras = (
        extraer_palabras_importantes(
            pregunta
        )
    )


    palabras_actuales = (
        extraer_palabras_importantes(
            pregunta_actual
            if pregunta_actual
            else pregunta
        )
    )


    if not palabras:

        return 0.0


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


    texto_completo = " ".join(

        [

            titulo,
            contenido,
            descripcion,
            tipo,

        ]

    )


    puntuacion = 0.0


    # ========================================================
    # PALABRAS DE LA PREGUNTA ACTUAL
    #
    # Tienen más peso que el contexto histórico.
    # ========================================================

    for palabra in palabras_actuales:

        patron = (
            rf"\b{re.escape(palabra)}\b"
        )


        coincidencias_titulo = len(

            re.findall(
                patron,
                titulo,
            )

        )


        coincidencias_descripcion = len(

            re.findall(
                patron,
                descripcion,
            )

        )


        coincidencias_contenido = len(

            re.findall(
                patron,
                contenido,
            )

        )


        puntuacion += (
            coincidencias_titulo * 18
        )


        puntuacion += (
            coincidencias_descripcion * 8
        )


        puntuacion += min(

            coincidencias_contenido * 2,

            12,

        )


    # ========================================================
    # CONTEXTO DE CONVERSACIÓN
    # ========================================================

    palabras_contexto = [

        palabra

        for palabra in palabras

        if palabra not in palabras_actuales

    ]


    for palabra in palabras_contexto:

        patron = (
            rf"\b{re.escape(palabra)}\b"
        )


        if re.search(
            patron,
            titulo,
        ):

            puntuacion += 5


        if re.search(
            patron,
            descripcion,
        ):

            puntuacion += 2


        coincidencias = len(

            re.findall(
                patron,
                texto_completo,
            )

        )


        puntuacion += min(

            coincidencias * 0.75,

            4,

        )


    # ========================================================
    # BONUS DE VARIAS PALABRAS ACTUALES
    # ========================================================

    palabras_en_texto = set(
        texto_completo.split()
    )


    coincidencias_actuales = (

        set(palabras_actuales)

        & palabras_en_texto

    )


    cantidad_actuales = len(
        coincidencias_actuales
    )


    if cantidad_actuales >= 2:

        puntuacion += (
            cantidad_actuales * 5
        )


    if (

        palabras_actuales

        and

        cantidad_actuales
        == len(palabras_actuales)

    ):

        puntuacion += 20


    return puntuacion


# ============================================================
# CREAR CLAVE REAL DE DUPLICADO
# ============================================================

def clave_unica_conocimiento(
    item: dict[str, Any],
) -> str:

    # ========================================================
    # IMPORTANTE:
    #
    # NO usamos solamente el ID.
    #
    # Dos registros pueden tener:
    #
    # id=A
    # id=B
    #
    # pero contener exactamente el mismo PDF.
    #
    # Por eso generamos una huella utilizando contenido.
    # ========================================================

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

    if not conocimientos:

        return []


    resultados = []

    claves_vistas = set()


    for indice, item in enumerate(
        conocimientos
    ):

        clave = clave_unica_conocimiento(
            item
        )


        if clave in claves_vistas:

            continue


        claves_vistas.add(
            clave
        )


        puntuacion = (
            calcular_relevancia(

                pregunta,

                item,

                pregunta_actual,

            )
        )


        if puntuacion > 0:

            resultados.append(

                (

                    puntuacion,

                    indice,

                    item,

                )

            )


    resultados.sort(

        key=lambda elemento: (

            elemento[0],

            -elemento[1],

        ),

        reverse=True,

    )


    seleccionados = [

        item

        for _, _, item

        in resultados[:top_k]

    ]


    print(
        "🔎 Búsqueda de conocimiento:"
    )


    print(
        f"   Registros totales: "
        f"{len(conocimientos)}"
    )


    print(
        f"   Registros únicos evaluados: "
        f"{len(claves_vistas)}"
    )


    print(
        f"   Registros relevantes: "
        f"{len(seleccionados)}"
    )


    if len(conocimientos) != len(claves_vistas):

        print(
            "♻️ Duplicados ignorados durante retrieval: "
            f"{len(conocimientos) - len(claves_vistas)}"
        )


    if resultados:

        for puntuacion, _, item in (
            resultados[:top_k]
        ):

            print(

                "   📌 "

                f"{item.get('titulo', '')} "

                f"(score={puntuacion:.1f})"

            )

    else:

        print(
            "   ℹ️ No se encontraron "
            "coincidencias relevantes."
        )


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

            pregunta_actual=pregunta_actual,

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

            + len(bloque)

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

        "version": "5.8.0",

        "provider": "OpenRouter",

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

        "min_output_tokens": (
            MIN_OUTPUT_TOKENS
        ),

        "automatic_token_adjustment": True,

        "single_token_adjustment": True,

        "duplicate_retrieval_filter": True,

        "current_question_priority": True,

        "openrouter": (
            bool(OPENROUTER_API_KEY)
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

    mensaje = data.message.strip()


    if not mensaje:

        raise HTTPException(

            status_code=400,

            detail=(
                "El mensaje no puede estar vacío."
            ),

        )


    if not OPENROUTER_API_KEY:

        raise HTTPException(

            status_code=500,

            detail=(
                "OPENROUTER_API_KEY "
                "no está configurada."
            ),

        )


    try:

        conocimientos = (
            cargar_conocimiento()
        )


        # ====================================================
        # BUSCAR CONOCIMIENTO
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
        # SYSTEM PROMPT
        # ====================================================

        system_prompt = (

            SYSTEM_PROMPT_BASE

            + "\n\n"

            + "==============================\n"

            + "INSTRUCCIONES SOBRE LA BASE "
            "DE CONOCIMIENTO\n"

            + "==============================\n"

            + """

La información que aparece a continuación
es solamente el subconjunto de registros
que el sistema local considera relacionados
con la conversación y la pregunta del usuario.

Debes tomar tú la decisión final sobre qué
información utilizar para responder.

La pregunta actual del usuario tiene prioridad
sobre el contexto anterior utilizado para realizar
la búsqueda.

No asumas que todos los registros son relevantes.

Si la información proporcionada no permite
responder con seguridad, dilo claramente.

No inventes información.

Si la pregunta no necesita conocimiento
específico de Penaguillo, puedes responder
normalmente utilizando tus capacidades.

Utiliza también el historial de conversación
proporcionado por el sistema para comprender
referencias, pronombres y preguntas de seguimiento.

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

"""

            + "\n\n"

            + "==============================\n"

            + "CONOCIMIENTO RELEVANTE\n"

            + "==============================\n"

            + conocimiento_relevante

        )


        print(
            "📤 Enviando pregunta a OpenRouter."
        )


        print(
            "📚 Contexto seleccionado: "
            f"{len(conocimiento_relevante)} caracteres"
        )


        # ====================================================
        # CONSTRUIR MENSAJES
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
                "OpenRouter: "
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
            "a OpenRouter: "
            f"{len(mensajes_api)}"
        )


        # ====================================================
        # GENERAR
        # ====================================================

        respuesta = generar_con_openrouter(

            model=CHAT_MODEL,

            messages=mensajes_api,

        )


        contenido = (
            extraer_contenido_openrouter(
                respuesta
            )
        )


        return {

            "ok": True,

            "response": contenido or "",

        }


    except OpenRouterError as error:

        print(
            f"❌ Error OpenRouter en /chat: "
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


        if error.status_code == 402:

            raise HTTPException(

                status_code=402,

                detail=(

                    "OpenRouter no tiene "
                    "créditos suficientes para "
                    "esta solicitud. "

                    "Se mantiene el objetivo de "
                    f"{MAX_OUTPUT_TOKENS} tokens y "
                    "no se realizarán más reducciones "
                    "automáticas."

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

                    "OpenRouter alcanzó "
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
# ENSEÑAR TEXTO
# ============================================================

@app.post("/ensenar")
def ensenar(

    data: EnsenarRequest,

):

    texto = (
        data.conocimiento
        .strip()
    )


    if not texto:

        raise HTTPException(

            status_code=400,

            detail=(

                "El conocimiento "
                "no puede estar vacío."

            ),

        )


    try:

        conocimientos = (
            cargar_conocimiento()
        )


        nuevo = {

            "id": generar_id(),

            "tipo": "texto",

            "titulo": (
                "Conocimiento manual"
            ),

            "contenido": texto,

            "descripcion": "",

            "fecha": ahora_iso(),

        }


        conocimientos.append(
            nuevo
        )


        guardar_conocimiento(
            conocimientos
        )


        return {

            "ok": True,

            "mensaje": (
                "Conocimiento guardado "
                "correctamente."
            ),

            "conocimiento": nuevo,

            "total": len(
                conocimientos
            ),

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

    if not OPENROUTER_API_KEY:

        raise RuntimeError(

            "OPENROUTER_API_KEY "
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

        respuesta = generar_con_openrouter(

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
            extraer_contenido_openrouter(
                respuesta
            )
        )


    except Exception as error:

        print(
            f"❌ Error OpenRouter Vision: "
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
# PDF — VISION CON OPENROUTER
# ============================================================

def analizar_pagina_pdf_con_vision(
    pagina: fitz.Page,
    numero_pagina: int,
) -> str:

    if not OPENROUTER_API_KEY:

        raise RuntimeError(

            "OPENROUTER_API_KEY "
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

        respuesta = generar_con_openrouter(

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
            extraer_contenido_openrouter(
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
        "OpenRouter"
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
        "💬 MAX HISTORIAL OPENROUTER: "
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
        "🤖 AJUSTE AUTOMÁTICO DE TOKENS: "
        "UNA SOLA VEZ"
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
        "🔑 OPENROUTER: "
        f"{'CONFIGURADO' if OPENROUTER_API_KEY else 'NO CONFIGURADO'}"
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