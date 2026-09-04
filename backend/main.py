# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 4.0
#
# FUNCIONES:
# - Chat con Penaguillo
# - Enseñar texto
# - Enseñar imágenes
# - Enseñar PDF
# - PDF con texto seleccionable -> PyMuPDF
# - PDF escaneado -> OpenRouter Vision
# - Imágenes -> OpenRouter Vision
# - Persistencia local
# - Persistencia en Google Drive cuando corre en Render
# - Backup automático
# - Escritura atómica
# - NO usa Tesseract
#
# LOCAL:
#   conocimiento/ se mantiene local.
#
# RENDER:
#   /var/data se utiliza como caché/persistencia local
#   y Google Drive funciona como almacenamiento remoto.
#
# GOOGLE DRIVE:
#   GOOGLE_SERVICE_ACCOUNT_EMAIL
#   GOOGLE_PRIVATE_KEY
#   GOOGLE_PROJECT_ID
#   GOOGLE_DRIVE_FOLDER_ID
# ============================================================


import base64
import json
import os
import re
import shutil
import sys
import uuid

from datetime import datetime
from pathlib import Path
from typing import Any


import fitz

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from openai import OpenAI

from pydantic import BaseModel


# ============================================================
# GOOGLE DRIVE
# ============================================================

try:

    from google.oauth2 import service_account

    from googleapiclient.discovery import build

    from googleapiclient.http import MediaFileUpload

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

    CONOCIMIENTO_DIR = Path(
        "/var/data/conocimiento"
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


if not OPENROUTER_API_KEY:

    print(
        "⚠️ ADVERTENCIA: "
        "No se encontró OPENROUTER_API_KEY."
    )


client = OpenAI(

    base_url=(
        "https://openrouter.ai/api/v1"
    ),

    api_key=OPENROUTER_API_KEY,

)


CHAT_MODEL = (
    "openai/gpt-oss-20b"
)


VISION_MODEL = (
    "openrouter/free"
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
# GOOGLE DRIVE — CONFIGURACIÓN
# ============================================================

DRIVE_SCOPES = [

    "https://www.googleapis.com/auth/drive"

]


drive_service = None


DRIVE_ROOT_FOLDER = None
DRIVE_KNOWLEDGE_FOLDER = None
DRIVE_FILES_FOLDER = None
DRIVE_PDF_FOLDER = None
DRIVE_IMAGES_FOLDER = None
DRIVE_BACKUPS_FOLDER = None


# ============================================================
# INICIALIZAR GOOGLE DRIVE
# ============================================================

def inicializar_google_drive():

    global drive_service

    global DRIVE_ROOT_FOLDER
    global DRIVE_KNOWLEDGE_FOLDER
    global DRIVE_FILES_FOLDER
    global DRIVE_PDF_FOLDER
    global DRIVE_IMAGES_FOLDER
    global DRIVE_BACKUPS_FOLDER


    # --------------------------------------------------------
    # Solo necesitamos Drive en Render
    # --------------------------------------------------------

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


        DRIVE_ROOT_FOLDER = (
            GOOGLE_DRIVE_FOLDER_ID
        )


        print(
            "✅ Google Drive conectado."
        )


        # ----------------------------------------------------
        # Crear estructura
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Descargar conocimiento existente
        # ----------------------------------------------------

        sincronizar_conocimiento_desde_drive()


    except Exception as error:

        drive_service = None

        print(
            "❌ Error inicializando Google Drive:"
        )

        print(error)


# ============================================================
# GOOGLE DRIVE — BUSCAR ARCHIVO
# ============================================================

def buscar_archivo_drive(
    nombre: str,
    folder_id: str,
):

    if not drive_service:

        return None


    query = (

        f"name = '{nombre.replace(chr(39), chr(92) + chr(39))}'"

        f" and '{folder_id}' in parents"

        " and trashed = false"

    )


    resultado = (
        drive_service
        .files()
        .list(

            q=query,

            spaces="drive",

            fields=(
                "files(id,name,mimeType,size)"
            ),

            pageSize=10,

        )
        .execute()
    )


    archivos = resultado.get(
        "files",
        [],
    )


    if archivos:

        return archivos[0]


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

            fields="id",

        )
        .execute()
    )


    print(
        f"📁 Carpeta Drive creada: {nombre}"
    )


    return archivo["id"]


# ============================================================
# GOOGLE DRIVE — SUBIR / ACTUALIZAR ARCHIVO
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


    existente = buscar_archivo_drive(
        nombre_drive,
        folder_id,
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


    media = MediaFileUpload(

        str(ruta),

        mimetype=mime_type,

        resumable=True,

    )


    try:

        # ----------------------------------------------------
        # ACTUALIZAR
        # ----------------------------------------------------

        if existente:

            archivo = (
                drive_service
                .files()
                .update(

                    fileId=existente["id"],

                    media_body=media,

                    fields="id,name",

                )
                .execute()
            )


            print(
                "☁️ Archivo actualizado en Drive: "
                f"{nombre_drive}"
            )


            return archivo["id"]


        # ----------------------------------------------------
        # CREAR
        # ----------------------------------------------------

        metadata = {

            "name": nombre_drive,

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

                fields="id,name",

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
            "❌ Error subiendo archivo a Drive:"
        )

        print(error)

        return None


# ============================================================
# GOOGLE DRIVE — DESCARGAR ARCHIVO
# ============================================================

def descargar_archivo_drive(
    file_id: str,
    destino: Path,
):

    if not drive_service:

        return False


    try:

        from googleapiclient.http import (
            MediaIoBaseDownload
        )

        import io


        request = (
            drive_service
            .files()
            .get(

                fileId=file_id,

                alt="media",

            )
        )


        buffer = io.BytesIO()


        downloader = MediaIoBaseDownload(

            buffer,

            request,

        )


        terminado = False


        while not terminado:

            _, terminado = (
                downloader.next_chunk()
            )


        destino.parent.mkdir(

            parents=True,

            exist_ok=True,

        )


        with open(

            destino,

            "wb",

        ) as archivo:

            archivo.write(
                buffer.getvalue()
            )


        return True


    except Exception as error:

        print(
            "❌ Error descargando archivo "
            "desde Drive:"
        )

        print(error)

        return False


# ============================================================
# GOOGLE DRIVE — SINCRONIZAR JSON DESDE DRIVE
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

            )
        )


        if not archivo_drive:

            print(
                "ℹ️ No existe penaguillo.json "
                "en Drive todavía."
            )

            # Si existe localmente,
            # lo subimos.
            if ARCHIVO_CONOCIMIENTO.exists():

                subir_archivo_drive(

                    ARCHIVO_CONOCIMIENTO,

                    DRIVE_KNOWLEDGE_FOLDER,

                    "penaguillo.json",

                )

            return


        print(
            "☁️ Descargando "
            "penaguillo.json desde Drive..."
        )


        temporal = (
            CONOCIMIENTO_DIR
            / "penaguillo_drive.tmp"
        )


        if descargar_archivo_drive(

            archivo_drive["id"],

            temporal,

        ):

            try:

                with open(

                    temporal,

                    "r",

                    encoding="utf-8",

                ) as archivo:

                    data = json.load(
                        archivo
                    )


                if isinstance(
                    data,
                    list,
                ):

                    os.replace(

                        temporal,

                        ARCHIVO_CONOCIMIENTO,

                    )


                    print(
                        "✅ Conocimiento "
                        "sincronizado desde Drive."
                    )

                else:

                    print(
                        "⚠️ penaguillo.json "
                        "de Drive no contiene "
                        "una lista."
                    )


            except Exception as error:

                print(
                    "⚠️ No se pudo validar "
                    "penaguillo.json de Drive:"
                )

                print(error)


            finally:

                if temporal.exists():

                    temporal.unlink()


    except Exception as error:

        print(
            "❌ Error sincronizando "
            "conocimiento desde Drive:"
        )

        print(error)


# ============================================================
# GOOGLE DRIVE — SUBIR CONOCIMIENTO
# ============================================================

def sincronizar_conocimiento_a_drive():

    if not drive_service:

        return


    if not DRIVE_KNOWLEDGE_FOLDER:

        return


    subir_archivo_drive(

        ARCHIVO_CONOCIMIENTO,

        DRIVE_KNOWLEDGE_FOLDER,

        "penaguillo.json",

    )


# ============================================================
# GOOGLE DRIVE — SUBIR BACKUP
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
# GOOGLE DRIVE — SUBIR PDF
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
# GOOGLE DRIVE — SUBIR IMAGEN
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

    version="4.0.0",

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

class ChatRequest(BaseModel):

    message: str


class EnsenarRequest(BaseModel):

    conocimiento: str


class EliminarRequest(BaseModel):

    id: str


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


# ============================================================
# CARGAR CONOCIMIENTO
# ============================================================

def cargar_conocimiento() -> list[dict[str, Any]]:

    if not ARCHIVO_CONOCIMIENTO.exists():

        print(
            "ℹ️ penaguillo.json no existe."
        )


        guardar_conocimiento([])


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


    return data


# ============================================================
# BACKUP
# ============================================================

def crear_backup() -> str | None:

    if not ARCHIVO_CONOCIMIENTO.exists():

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


    try:

        shutil.copy2(

            ARCHIVO_CONOCIMIENTO,

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


    # --------------------------------------------------------
    # BACKUP ANTES DE MODIFICAR
    # --------------------------------------------------------

    crear_backup()


    # --------------------------------------------------------
    # ARCHIVO TEMPORAL
    # --------------------------------------------------------

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

            f"Total registros: "
            f"{len(conocimientos)}"

        )


        # ----------------------------------------------------
        # GOOGLE DRIVE
        # ----------------------------------------------------

        sincronizar_conocimiento_a_drive()


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
# CONSTRUIR CONOCIMIENTO
# ============================================================

def construir_conocimiento() -> str:

    conocimientos = (
        cargar_conocimiento()
    )


    if not conocimientos:

        return (

            "Actualmente no hay conocimiento "
            "adicional registrado."

        )


    bloques = []


    for indice, item in enumerate(

        conocimientos,

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


        texto = f"""

REGISTRO {indice}

ID:
{item.get("id", "")}

TIPO:
{tipo}

TÍTULO:
{titulo}

CONTENIDO:
{contenido}

DESCRIPCIÓN VISUAL:
{descripcion}

"""


        bloques.append(
            texto.strip()
        )


    return (

        "\n\n==============================\n\n"

        .join(
            bloques
        )

    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT_BASE = (
    cargar_system_prompt()
)


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

        "version": "4.0.0",

        "chat_model": CHAT_MODEL,

        "vision_model": VISION_MODEL,

        "conocimientos": len(
            conocimientos
        ),

        "google_drive": (
            drive_service is not None
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
                "El mensaje no puede "
                "estar vacío."
            ),

        )


    try:

        conocimiento = (
            construir_conocimiento()
        )


    except RuntimeError as error:

        raise HTTPException(

            status_code=500,

            detail=str(error),

        )


    system_prompt = (

        SYSTEM_PROMPT_BASE

        + "\n\n"

        + "==============================\n"

        + "BASE DE CONOCIMIENTO "
          "DE PENAGUILLO\n"

        + "==============================\n"

        + conocimiento

    )


    try:

        respuesta = (
            client
            .chat
            .completions
            .create(

                model=CHAT_MODEL,

                messages=[

                    {

                        "role": "system",

                        "content": (
                            system_prompt
                        ),

                    },

                    {

                        "role": "user",

                        "content": mensaje,

                    },

                ],

                temperature=0.2,

            )
        )


        contenido = (

            respuesta

            .choices[0]

            .message

            .content

        )


        return {

            "ok": True,

            "response": (
                contenido or ""
            ),

        }


    except Exception as error:

        print(
            f"❌ Error en /chat: {error}"
        )


        raise HTTPException(

            status_code=500,

            detail=(

                "Error consultando "
                f"Penaguillo: {error}"

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
# IMAGEN
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
# VISION
# ============================================================

def analizar_imagen_con_vision(

    ruta: Path,

    contexto: str = "",

) -> str:

    if not OPENROUTER_API_KEY:

        raise RuntimeError(

            "No existe "
            "OPENROUTER_API_KEY."

        )


    imagen_base64 = (
        imagen_a_base64(ruta)
    )


    mime = mime_imagen(
        ruta
    )


    prompt = f"""

Analiza cuidadosamente esta imagen para alimentar la base de
conocimiento de Penaguillo.

Tu respuesta debe ser útil para una empresa de maquinaria agrícola
y de procesamiento de café.

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
- Personas o contexto únicamente si aporta información útil.
- Cualquier información relevante para Penagos.

Si hay texto en la imagen, transcríbelo de forma clara.

NO inventes información que no puedas observar.

Contexto proporcionado por el usuario:
{contexto}

Devuelve una descripción estructurada y detallada.

"""


    try:

        respuesta = (

            client

            .chat

            .completions

            .create(

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

                                    "url": (

                                        f"data:{mime};"
                                        f"base64,"
                                        f"{imagen_base64}"

                                    )

                                },

                            },

                        ],

                    }

                ],

                temperature=0.1,

                max_tokens=8000,

            )

        )


        return (

            respuesta

            .choices[0]

            .message

            .content

            or ""

        )


    except Exception as error:

        print(
            f"❌ Error Vision: {error}"
        )


        raise RuntimeError(

            "No fue posible analizar "
            f"la imagen: {error}"

        )


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
                "La imagen supera "
                "15 MB."
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
            f"❌ Error /ensenar-imagen: {error}"
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
# PDF
# ============================================================

def pdf_a_imagen_base64(
    pagina: fitz.Page,
) -> str:

    matriz = fitz.Matrix(

        1.5,

        1.5,

    )


    pixmap = pagina.get_pixmap(

        matrix=matriz,

        alpha=False,

    )


    imagen_bytes = (
        pixmap.tobytes("png")
    )


    return base64.b64encode(
        imagen_bytes
    ).decode(
        "utf-8"
    )


# ============================================================
# PDF VISION
# ============================================================

def analizar_pagina_pdf_con_vision(

    pagina: fitz.Page,

    numero_pagina: int,

) -> str:

    if not OPENROUTER_API_KEY:

        raise RuntimeError(
            "No existe OPENROUTER_API_KEY."
        )


    imagen_base64 = (
        pdf_a_imagen_base64(
            pagina
        )
    )


    prompt = f"""

Analiza esta página de un PDF para alimentar la base de conocimiento
de Penaguillo.

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

Si existe una tabla, intenta conservar su estructura.

No inventes información.

Devuelve todo lo útil de esta página en texto estructurado.

"""


    try:

        respuesta = (

            client

            .chat

            .completions

            .create(

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

                                    "url": (

                                        "data:image/png;"
                                        "base64,"
                                        + imagen_base64

                                    )

                                },

                            },

                        ],

                    }

                ],

                temperature=0.1,

                max_tokens=8000,

            )

        )


        return (

            respuesta

            .choices[0]

            .message

            .content

            or ""

        )


    except Exception as error:

        raise RuntimeError(

            "Error analizando página "

            f"{numero_pagina}: {error}"

        )


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


        # ----------------------------------------------------
        # PDF NORMAL
        # ----------------------------------------------------

        if not es_escaneado:

            contenido_final = (
                texto_extraido
            )


        # ----------------------------------------------------
        # PDF ESCANEADO
        # ----------------------------------------------------

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

                        f"""

==============================
PÁGINA {indice}
==============================

{texto_pagina}

""".strip()

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

            "numero_paginas": numero_paginas,

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

            "numero_paginas": numero_paginas,

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
            f"❌ Error /ensenar-pdf: {error}"
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

            "conocimientos": conocimientos,

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

                "fecha": datetime.fromtimestamp(

                    archivo.stat().st_mtime

                ).isoformat(),

            }

            for archivo in archivos

        ],

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
        f"📂 CONOCIMIENTO_DIR: "
        f"{CONOCIMIENTO_DIR}"
    )


    print(
        f"🌐 RENDER: "
        f"{os.getenv('RENDER')}"
    )


    inicializar_google_drive()


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