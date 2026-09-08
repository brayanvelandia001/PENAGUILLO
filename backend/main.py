# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 7.0 (BÚSQUEDA SEMÁNTICA VECTORIAL / EMBEDDINGS)
#
# PROVEEDOR DE IA:
# - Google Gemini Native API (Chat + Embeddings + Vision)
#
# MODELOS:
# - CHAT/VISION: gemini-3.5-flash-lite
# - EMBEDDINGS: text-embedding-004
#
# FUNCIONES COMPLETAS:
# - Chat conversacional con historial y contexto
# - Búsqueda Semántica Vectorial por similitud de coseno
# - Enseñar texto (Generación de título dinámico + Vectorial)
# - Enseñar PDF (Texto nativo vía PyMuPDF + Escaneados vía Gemini Vision)
# - Enseñar Imágenes (Análisis vía Gemini Vision)
# - Persistencia local y Google Drive (Shared Drive compatible)
# - Backups atómicos
# - Reglas estrictas de fidelidad de datos (copia literal)
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

import pymupdf as fitz
import numpy as np
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
# GOOGLE DRIVE IMPORTS
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
# CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ============================================================

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

if os.getenv("RENDER") == "true":
    CONOCIMIENTO_DIR = BASE_DIR / "storage" / "conocimiento"
else:
    CONOCIMIENTO_DIR = BASE_DIR / "conocimiento"

ARCHIVOS_DIR = CONOCIMIENTO_DIR / "archivos"
PDF_DIR = ARCHIVOS_DIR / "pdf"
IMAGENES_DIR = ARCHIVOS_DIR / "imagenes"
BACKUP_DIR = CONOCIMIENTO_DIR / "backups"
ARCHIVO_CONOCIMIENTO = CONOCIMIENTO_DIR / "penaguillo.json"
PROMPT_FILE = BASE_DIR / "conocimiento" / "prompt.txt"

for directorio in [CONOCIMIENTO_DIR, ARCHIVOS_DIR, PDF_DIR, IMAGENES_DIR, BACKUP_DIR]:
    directorio.mkdir(parents=True, exist_ok=True)

ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


# ============================================================
# GEMINI Y CONFIGURACIÓN GENERAL
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "text-embedding-004"

CHAT_MODEL = GEMINI_MODEL
VISION_MODEL = GEMINI_MODEL

MAX_OUTPUT_TOKENS = 1200
RELEVANCIA_TOP_K = 8
MAX_KB_CONOCIMIENTO_CHAT = 32
MAX_CHARS_CONOCIMIENTO_CHAT = MAX_KB_CONOCIMIENTO_CHAT * 1024
MAX_MENSAJES_HISTORIAL = 6

MAX_PDF_SIZE = 50 * 1024 * 1024
MAX_IMAGE_SIZE = 15 * 1024 * 1024
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}
EXTENSIONES_PDF = {".pdf"}


# ============================================================
# EXCEPCIÓN Y CLIENTE GEMINI
# ============================================================

class GeminiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def generar_con_gemini(*, model: str, messages: list, max_retries: int = 2):
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY no está configurada.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    system_instruction = None
    gemini_contents = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system" and isinstance(content, str):
            system_instruction = {"parts": [{"text": content}]}
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts = []

        if isinstance(content, str) and content.strip():
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text" and item.get("text"):
                    parts.append({"text": item.get("text")})
                elif item_type == "image_url":
                    url_img = item.get("image_url", {}).get("url", "")
                    if url_img.startswith("data:"):
                        try:
                            encabezado, b64_data = url_img.split(",", 1)
                            mime_type = encabezado.split(":", 1)[1].split(";", 1)[0].strip()
                            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})
                        except Exception:
                            pass

        if parts:
            gemini_contents.append({"role": gemini_role, "parts": parts})

    payload = {
        "contents": gemini_contents,
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.3}
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    for intento in range(1, max_retries + 1):
        try:
            respuesta = requests.post(url, headers=headers, json=payload, timeout=120)
            if respuesta.status_code == 200:
                return respuesta.json()
            if intento >= max_retries:
                raise GeminiError(f"Gemini HTTP {respuesta.status_code}: {respuesta.text[:500]}")
            time.sleep(2 ** intento)
        except requests.RequestException as e:
            if intento >= max_retries:
                raise GeminiError(f"Error de conexión con Gemini: {e}")
            time.sleep(2 ** intento)

    raise GeminiError("Gemini no pudo generar una respuesta.")


def extraer_contenido_gemini(respuesta: dict) -> str:
    if not respuesta or "candidates" not in respuesta:
        raise RuntimeError("Respuesta de Gemini no válida.")
    partes = respuesta["candidates"][0].get("content", {}).get("parts", [])
    texto = "\n".join([p.get("text", "") for p in partes if "text" in p]).strip()
    if texto:
        return texto
    raise RuntimeError("Contenido devuelto por Gemini está vacío.")


# ============================================================
# MOTOR DE EMBEDDINGS (BÚSQUEDA VECTORIAL)
# ============================================================

def obtener_embedding(texto: str) -> list[float]:
    """Genera un vector matemático que representa el significado del texto."""
    if not GEMINI_API_KEY or not texto.strip():
        return []

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": texto.strip()[:2000]}]}
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("embedding", {}).get("values", [])
    except Exception as e:
        print(f"⚠️ Error generando embedding: {e}")
    return []


def sim_coseno(v1: list[float], v2: list[float]) -> float:
    """Calcula el grado de similitud conceptual entre dos vectores."""
    if not v1 or not v2:
        return 0.0
    a, b = np.array(v1), np.array(v2)
    norma_a, norma_b = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (norma_a * norma_b)) if norma_a != 0 and norma_b != 0 else 0.0


# ============================================================
# GOOGLE DRIVE Y PERSISTENCIA DE DATOS
# ============================================================

GOOGLE_SERVICE_ACCOUNT_EMAIL = os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL")
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

drive_service = None
drive_session = None
DRIVE_ROOT_FOLDER = None
DRIVE_SHARED_ID = None
DRIVE_KNOWLEDGE_FOLDER = None
DRIVE_FILES_FOLDER = None
DRIVE_PDF_FOLDER = None
DRIVE_IMAGES_FOLDER = None
DRIVE_BACKUPS_FOLDER = None


def ahora_iso() -> str:
    return datetime.now().isoformat()

def generar_id() -> str:
    return str(uuid.uuid4())

def nombre_seguro(nombre: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", Path(nombre).name) or f"archivo_{generar_id()}"


def buscar_archivo_drive(nombre: str, folder_id: str, solo_json: bool = False):
    if not drive_service:
        return None
    nombre_escapado = nombre.replace("'", "\\'")
    query = f"name = '{nombre_escapado}' and '{folder_id}' in parents and trashed = false"
    if solo_json:
        query += " and mimeType = 'application/json'"
    try:
        params = {
            "q": query, "spaces": "drive", "includeItemsFromAllDrives": True,
            "supportsAllDrives": True, "fields": "files(id, name, mimeType)", "pageSize": 100
        }
        if DRIVE_SHARED_ID:
            params["corpora"] = "drive"
            params["driveId"] = DRIVE_SHARED_ID
        res = drive_service.files().list(**params).execute().get("files", [])
        return res[0] if res else None
    except Exception:
        return None


def obtener_o_crear_carpeta(nombre: str, parent_id: str):
    ex = buscar_archivo_drive(nombre, parent_id)
    if ex and ex.get("mimeType") == "application/vnd.google-apps.folder":
        return ex["id"]
    metadata = {"name": nombre, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    return drive_service.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()["id"]


def inicializar_google_drive():
    global drive_service, drive_session, DRIVE_ROOT_FOLDER, DRIVE_SHARED_ID
    global DRIVE_KNOWLEDGE_FOLDER, DRIVE_FILES_FOLDER, DRIVE_PDF_FOLDER, DRIVE_IMAGES_FOLDER, DRIVE_BACKUPS_FOLDER

    if os.getenv("RENDER") != "true" or not GOOGLE_AVAILABLE:
        return
    if not all([GOOGLE_SERVICE_ACCOUNT_EMAIL, GOOGLE_PRIVATE_KEY, GOOGLE_PROJECT_ID, GOOGLE_DRIVE_FOLDER_ID]):
        return

    try:
        pk = GOOGLE_PRIVATE_KEY.replace("\\n", "\n")
        info = {
            "type": "service_account", "project_id": GOOGLE_PROJECT_ID,
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID", ""), "private_key": pk,
            "client_email": GOOGLE_SERVICE_ACCOUNT_EMAIL, "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL", ""),
        }
        cred = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
        drive_service = build("drive", "v3", credentials=cred, cache_discovery=False)
        drive_session = AuthorizedSession(cred)
        DRIVE_ROOT_FOLDER = GOOGLE_DRIVE_FOLDER_ID

        raiz = drive_service.files().get(fileId=DRIVE_ROOT_FOLDER, fields="id, driveId", supportsAllDrives=True).execute()
        DRIVE_SHARED_ID = raiz.get("driveId")
        DRIVE_KNOWLEDGE_FOLDER = obtener_o_crear_carpeta("conocimiento", DRIVE_ROOT_FOLDER)
        DRIVE_FILES_FOLDER = obtener_o_crear_carpeta("archivos", DRIVE_ROOT_FOLDER)
        DRIVE_PDF_FOLDER = obtener_o_crear_carpeta("pdf", DRIVE_FILES_FOLDER)
        DRIVE_IMAGES_FOLDER = obtener_o_crear_carpeta("imagenes", DRIVE_FILES_FOLDER)
        DRIVE_BACKUPS_FOLDER = obtener_o_crear_carpeta("backups", DRIVE_KNOWLEDGE_FOLDER)
        sincronizar_conocimiento_desde_drive()
    except Exception:
        drive_service = None


def subir_archivo_drive(ruta: Path, folder_id: str, nombre: str | None = None):
    if not drive_service or not ruta.exists():
        return None
    nombre_d = nombre or ruta.name
    ex = buscar_archivo_drive(nombre_d, folder_id, solo_json=(nombre_d.lower() == "penaguillo.json"))
    mime = "application/json" if ruta.suffix.lower() == ".json" else "application/octet-stream"
    try:
        media = MediaFileUpload(str(ruta), mimetype=mime, resumable=True)
        if ex:
            return drive_service.files().update(fileId=ex["id"], media_body=media, supportsAllDrives=True).execute()["id"]
        return drive_service.files().create(body={"name": nombre_d, "mimeType": mime, "parents": [folder_id]}, media_body=media, supportsAllDrives=True).execute()["id"]
    except Exception:
        return None


def descargar_archivo_drive(file_id: str, destino: Path):
    if not drive_service or not drive_session:
        return False
    try:
        resp = drive_session.get(f"https://www.googleapis.com/drive/v3/files/{file_id}", params={"alt": "media", "supportsAllDrives": "true"}, timeout=120)
        if resp.status_code == 200 and len(resp.content) > 0:
            destino.parent.mkdir(parents=True, exist_ok=True)
            tmp = destino.parent / f".tmp_{generar_id()}"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            os.replace(tmp, destino)
            return True
    except Exception:
        pass
    return False


def validar_json_conocimiento(ruta: Path) -> tuple[bool, list[dict[str, Any]]]:
    if not ruta.exists() or ruta.stat().st_size == 0:
        return False, []
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (True, data) if isinstance(data, list) and all(isinstance(i, dict) for i in data) else (False, [])
    except Exception:
        return False, []


def sincronizar_conocimiento_desde_drive():
    if not drive_service or not DRIVE_KNOWLEDGE_FOLDER:
        return
    try:
        ex = buscar_archivo_drive("penaguillo.json", DRIVE_KNOWLEDGE_FOLDER, solo_json=True)
        if ex:
            tmp = CONOCIMIENTO_DIR / "penaguillo_drive.tmp"
            if descargar_archivo_drive(ex["id"], tmp):
                valido, data = validar_json_conocimiento(tmp)
                if valido:
                    os.replace(tmp, ARCHIVO_CONOCIMIENTO)
                    print("✅ Conocimiento sincronizado desde Drive.")
                    return
            if tmp.exists():
                tmp.unlink()
    except Exception:
        pass


def sincronizar_conocimiento_a_drive():
    if drive_service and DRIVE_KNOWLEDGE_FOLDER and ARCHIVO_CONOCIMIENTO.exists():
        if validar_json_conocimiento(ARCHIVO_CONOCIMIENTO)[0]:
            subir_archivo_drive(ARCHIVO_CONOCIMIENTO, DRIVE_KNOWLEDGE_FOLDER, "penaguillo.json")


def cargar_conocimiento() -> list[dict[str, Any]]:
    if not ARCHIVO_CONOCIMIENTO.exists():
        return []
    try:
        with open(ARCHIVO_CONOCIMIENTO, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def guardar_conocimiento(conocimientos: list[dict[str, Any]]) -> None:
    tmp = CONOCIMIENTO_DIR / f"tmp_{generar_id()}.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(conocimientos, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ARCHIVO_CONOCIMIENTO)
    sincronizar_conocimiento_a_drive()


def crear_backup() -> str | None:
    if not ARCHIVO_CONOCIMIENTO.exists():
        return None
    valido, data = validar_json_conocimiento(ARCHIVO_CONOCIMIENTO)
    if not valido:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = BACKUP_DIR / f"penaguillo_{timestamp}.json"
    tmp = BACKUP_DIR / f"tmp_{generar_id()}.json"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, backup_path)
        subir_archivo_drive(backup_path, DRIVE_BACKUPS_FOLDER, backup_path.name)
        return str(backup_path)
    except Exception:
        return None


# ============================================================
# PROCESAMIENTO DE ARCHIVOS (PDF E IMÁGENES)
# ============================================================

def pdf_a_imagen_bytes(pagina: fitz.Page) -> bytes:
    return pagina.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).tobytes("png")


def analizar_pagina_pdf_con_vision(pagina: fitz.Page, numero_pagina: int) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY no configurada.")
    imagen_bytes = pdf_a_imagen_bytes(pagina)
    imagen_base64 = base64.b64encode(imagen_bytes).decode("utf-8")
    prompt = f"Analiza esta página {numero_pagina} de un PDF para la base de conocimiento de Penaguillo. Extrae todo texto, datos y tablas estructuradamente."
    
    resp = generar_con_gemini(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_base64}"}}]}]
    )
    return extraer_contenido_gemini(resp)


def analizar_imagen_con_vision(ruta: Path) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY no configurada.")
    imagen_base64 = base64.b64encode(ruta.read_bytes()).decode("utf-8")
    mime = "image/png" if ruta.suffix.lower() == ".png" else "image/jpeg"
    prompt = "Analiza detalladamente esta imagen para Penaguillo. Transcribe textos, identifica máquinas, componentes y datos relevantes."
    
    resp = generar_con_gemini(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{imagen_base64}"}}]}]
    )
    return extraer_contenido_gemini(resp)


# ============================================================
# FASTAPI CONFIGURACIÓN
# ============================================================

app = FastAPI(title="Penaguillo IA", version="7.0.0", description="Backend Inteligente Penaguillo con Búsqueda Vectorial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/archivos", StaticFiles(directory=str(ARCHIVOS_DIR)), name="archivos")


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
# BÚSQUEDA SEMÁNTICA VECTORIAL (RETRIEVAL V7)
# ============================================================

def buscar_conocimiento_vectorial(pregunta: str, conocimientos: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    if not conocimientos or not pregunta.strip():
        return []

    v_pregunta = obtener_embedding(pregunta)
    if not v_pregunta:
        return conocimientos[:top_k]

    resultados = []
    for item in conocimientos:
        v_item = item.get("embedding", [])
        
        # Genera el vector si el registro no lo tiene
        if not v_item:
            v_item = obtener_embedding(f"{item.get('titulo', '')}\n{item.get('contenido', '')}\n{item.get('descripcion', '')}")
            item["embedding"] = v_item

        score = sim_coseno(v_pregunta, v_item)
        if item.get("tipo") == "texto":
            score += 0.05  # Bono para priorizar textos directos sobre PDFs extensos

        if score > 0.35:
            resultados.append((score, item))

    resultados.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in resultados[:top_k]]


def construir_contexto_relevante(pregunta: str, conocimientos: list[dict[str, Any]]) -> str:
    relevantes = buscar_conocimiento_vectorial(pregunta, conocimientos, top_k=RELEVANCIA_TOP_K)
    if not relevantes:
        return "No se encontraron registros específicos en la base de conocimiento para esta pregunta."

    bloques = []
    for idx, item in enumerate(relevantes, 1):
        bloques.append(
            f"REGISTRO {idx}\n"
            f"TÍTULO: {item.get('titulo', '')}\n"
            f"CONTENIDO: {item.get('contenido', '')}\n"
            f"DESCRIPCIÓN: {item.get('descripcion', '')}"
        )
    return "\n\n==============================\n\n".join(bloques)


def cargar_system_prompt() -> str:
    if not PROMPT_FILE.exists():
        return "Eres Penaguillo, el asistente virtual de Penagos Hermanos."
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT_BASE = cargar_system_prompt()


# ============================================================
# ENDPOINTS API
# ============================================================

@app.get("/")
def root():
    c = cargar_conocimiento()
    return {"ok": True, "app": "Penaguillo IA v7.0", "conocimientos": len(c), "engine": "Semantic Vector Search"}


@app.post("/chat")
def chat(data: ChatRequest):
    mensaje = data.message.strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje está vacío.")

    try:
        conocimientos = cargar_conocimiento()
        contexto_vectorial = construir_contexto_relevante(mensaje, conocimientos)

        system_prompt = (
            SYSTEM_PROMPT_BASE
            + "\n\n==============================\nINSTRUCCIONES SOBRE LA BASE DE CONOCIMIENTO\n==============================\n"
            + f"DATO IMPORTANTE: Actualmente tienes exactamente {len(conocimientos)} registros en tu base de datos local.\n\n"
            + "La información a continuación es el subconjunto de registros relacionados.\n"
            + "Debes tomar la decisión final sobre qué usar. La pregunta ACTUAL tiene prioridad. No inventes información.\n\n"
            + "==============================\nCONOCIMIENTO RELEVANTE\n==============================\n"
            + contexto_vectorial
            + "\n\n==============================\nREGLAS DE COMPLETITUD Y COPIA LITERAL\n==============================\n"
            + "1. Queda ABSOLUTAMENTE PROHIBIDO alterar, acortar, modificar o inferir direcciones de correo electrónico, teléfonos o nombres propios.\n"
            + "2. COPIA Y PEGA literalmente los correos del contexto. Si el contexto dice 'modernizacion@penagos.co', DEBES escribir 'modernizacion@penagos.co'. JAMÁS recortes la extensión final (.co o .com).\n"
            + "3. Si el usuario pide un 'equipo', 'grupo' o 'área', debes listar a TODOS los miembros relevantes del contexto sin omitir a ninguno.\n"
        )

        mensajes_api = [{"role": "system", "content": system_prompt}]

        if data.history:
            for msg in data.history[-MAX_MENSAJES_HISTORIAL:]:
                if msg.content.strip() and msg.role in ("user", "assistant"):
                    mensajes_api.append({"role": msg.role, "content": msg.content.strip()})

        mensajes_api.append({"role": "user", "content": mensaje})

        resp = generar_con_gemini(model=CHAT_MODEL, messages=mensajes_api)
        contenido = extraer_contenido_gemini(resp)

        return {"ok": True, "response": contenido}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Penaguillo: {e}")


@app.post("/ensenar")
def ensenar(data: EnsenarRequest):
    texto = data.conocimiento.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    try:
        conocimientos = cargar_conocimiento()
        palabras = texto.split()
        titulo = " ".join(palabras[:8]) + ("..." if len(palabras) > 8 else "")
        vec = obtener_embedding(f"{titulo}\n{texto}")

        nuevo = {
            "id": generar_id(),
            "tipo": "texto",
            "titulo": titulo,
            "contenido": texto,
            "descripcion": f"Información clave: {titulo}",
            "embedding": vec,
            "fecha": ahora_iso(),
        }

        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)
        crear_backup()

        return {"ok": True, "mensaje": "Conocimiento aprendido con búsqueda vectorial.", "total": len(conocimientos)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ensenar-pdf")
async def ensenar_pdf(file: UploadFile = File(...)):
    nombre_original = file.filename or "documento.pdf"
    if Path(nombre_original).suffix.lower() not in EXTENSIONES_PDF:
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    contenido = await file.read()
    if len(contenido) > MAX_PDF_SIZE:
        raise HTTPException(status_code=400, detail="El PDF supera los 50 MB.")

    nombre = f"{uuid.uuid4().hex}_{nombre_seguro(nombre_original)}"
    ruta = PDF_DIR / nombre
    documento = None

    try:
        with open(ruta, "wb") as f:
            f.write(contenido)

        documento = fitz.open(str(ruta))
        numero_paginas = documento.page_count
        texto_extraido = "\n\n".join([p.get_text("text", sort=True).strip() for p in documento if p.get_text("text", sort=True).strip()])

        es_escaneado = len(re.sub(r"\s+", "", texto_extraido)) < 50
        contenido_final = texto_extraido if not es_escaneado else "\n\n".join([analizar_pagina_pdf_con_vision(p, i) for i, p in enumerate(documento, 1)])

        documento.close()
        conocimientos = cargar_conocimiento()

        vec = obtener_embedding(f"{nombre_original}\n{contenido_final[:1000]}")

        nuevo = {
            "id": generar_id(), "tipo": "pdf", "titulo": nombre_original,
            "contenido": contenido_final, "descripcion": "Documento PDF procesado por Penaguillo.",
            "archivo": f"/archivos/pdf/{nombre}", "nombre_archivo": nombre_original,
            "numero_paginas": numero_paginas, "metodo": "vision" if es_escaneado else "texto",
            "embedding": vec, "fecha": ahora_iso(),
        }

        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)
        sincronizar_pdf_a_drive(ruta)
        crear_backup()

        return {"ok": True, "mensaje": "PDF aprendido correctamente.", "conocimiento": nuevo, "total": len(conocimientos)}

    except Exception as error:
        if documento:
            documento.close()
        if ruta.exists():
            ruta.unlink()
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {error}")


@app.post("/ensenar-imagen")
async def ensenar_imagen(file: UploadFile = File(...)):
    nombre_original = file.filename or "imagen.png"
    if Path(nombre_original).suffix.lower() not in EXTENSIONES_IMAGEN:
        raise HTTPException(status_code=400, detail="Formato de imagen no permitido.")

    contenido = await file.read()
    if len(contenido) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="La imagen supera los 15 MB.")

    nombre = f"{uuid.uuid4().hex}_{nombre_seguro(nombre_original)}"
    ruta = IMAGENES_DIR / nombre

    try:
        with open(ruta, "wb") as f:
            f.write(contenido)

        descripcion = analizar_imagen_con_vision(ruta)
        conocimientos = cargar_conocimiento()
        vec = obtener_embedding(f"{nombre_original}\n{descripcion}")

        nuevo = {
            "id": generar_id(), "tipo": "imagen", "titulo": nombre_original,
            "contenido": descripcion, "descripcion": descripcion,
            "archivo": f"/archivos/imagenes/{nombre}", "nombre_archivo": nombre_original,
            "embedding": vec, "fecha": ahora_iso(),
        }

        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)
        sincronizar_imagen_a_drive(ruta)
        crear_backup()

        return {"ok": True, "mensaje": "Imagen aprendida correctamente.", "conocimiento": nuevo, "total": len(conocimientos)}

    except Exception as error:
        if ruta.exists():
            ruta.unlink()
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {error}")


@app.get("/conocimiento")
def obtener_conocimiento():
    c = cargar_conocimiento()
    return {"ok": True, "total": len(c), "conocimientos": c}


@app.delete("/conocimiento")
def eliminar_conocimiento(data: EliminarRequest):
    conocimientos = cargar_conocimiento()
    nuevos = [i for i in conocimientos if str(i.get("id")) != str(data.id)]
    guardar_conocimiento(nuevos)
    return {"ok": True, "mensaje": "Eliminado correctamente.", "total": len(nuevos)}


@app.get("/backups")
def listar_backups():
    archivos = sorted(BACKUP_DIR.glob("penaguillo_*.json"), key=lambda a: a.stat().st_mtime, reverse=True)
    return {"ok": True, "total": len(archivos), "backups": [{"nombre": a.name, "fecha": datetime.fromtimestamp(a.stat().st_mtime).isoformat()} for a in archivos]}


# ============================================================
# STARTUP Y EJECUCIÓN
# ============================================================

@app.on_event("startup")
def startup():
    print("🚀 Penaguillo IA v7.0 Iniciado.")
    inicializar_google_drive()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)