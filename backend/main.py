# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 7.0 (Búsqueda Vectorial Semántica + Cero Alucinaciones + Anti-Límites 429)
#
# PROVEEDOR DE IA:
# - Google Gemini Native API
#
# MODELOS:
# - CHAT/VISION: gemini-3.5-flash-lite
# - EMBEDDINGS: text-embedding-004
#
# FUNCIONES:
# - Búsqueda Vectorial (Embeddings) para precisión total y bajo consumo de tokens
# - Chat con historial y recuperación de contexto
# - Enseñar texto, imágenes y PDF con generación de vectores automática
# - Persistencia local y Google Drive permanente
# - Reglas estrictas de fidelidad de datos (Copia literal de correos/celulares)
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

import pymupdf as fitz  # Soluciona la advertencia de Render sobre fitz
import numpy as np      # Motor matemático para los vectores
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

CONOCIMIENTO_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVOS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
IMAGENES_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


# ============================================================
# GEMINI NATIVO
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "text-embedding-004"

CHAT_MODEL = GEMINI_MODEL
VISION_MODEL = GEMINI_MODEL

MAX_OUTPUT_TOKENS = 1200
RELEVANCIA_TOP_K = 10
MAX_MENSAJES_HISTORIAL = 6


if not GEMINI_API_KEY:
    print("⚠️ ADVERTENCIA: GEMINI_API_KEY no está configurada.")
else:
    print("🔑 GEMINI_API_KEY: configurada")


# ============================================================
# ERROR CONTROLADO DE GEMINI
# ============================================================

class GeminiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

def extraer_retry_after(respuesta: requests.Response, mensaje_error: str) -> int | None:
    valor_header = respuesta.headers.get("Retry-After")
    if valor_header:
        try:
            segundos = int(valor_header)
            if segundos >= 0:
                return segundos
        except (TypeError, ValueError):
            pass

    patrones = [r"retry in ([0-9]+(?:\.[0-9]+)?)s", r"retryDelay.*?([0-9]+)s", r"seconds.*?([0-9]+)"]
    texto = mensaje_error or ""
    for patron in patrones:
        coincidencia = re.search(patron, texto, re.IGNORECASE)
        if coincidencia:
            try:
                return max(1, int(float(coincidencia.group(1))))
            except (TypeError, ValueError):
                pass
    return None


# ============================================================
# GEMINI — GENERAR RESPUESTA
# ============================================================

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

        if role == "system":
            if isinstance(content, str):
                system_instruction = {"parts": [{"text": content}]}
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts = []

        if isinstance(content, str):
            if content.strip():
                parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict): continue
                item_type = item.get("type")
                if item_type == "text":
                    texto = item.get("text", "")
                    if texto: parts.append({"text": texto})
                elif item_type == "image_url":
                    url_img = item.get("image_url", {}).get("url", "")
                    if not url_img.startswith("data:"): continue
                    try:
                        encabezado, b64_data = url_img.split(",", 1)
                        mime_type = encabezado.split(":", 1)[1].split(";", 1)[0].strip()
                        parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})
                    except Exception:
                        pass
        if parts:
            gemini_contents.append({"role": gemini_role, "parts": parts})

    payload = {"contents": gemini_contents, "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.3}}
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    ultimo_error = None

    for intento in range(1, max_retries + 1):
        try:
            respuesta = requests.post(url, headers=headers, json=payload, timeout=120)
            if respuesta.status_code == 200:
                return respuesta.json()

            mensaje_error = respuesta.text[:5000]
            retry_after = extraer_retry_after(respuesta, mensaje_error)
            ultimo_error = GeminiError(f"Gemini HTTP {respuesta.status_code}: {mensaje_error}", status_code=respuesta.status_code, retry_after=retry_after)

            es_temporal = respuesta.status_code in (429, 500, 502, 503, 504)
            if not es_temporal or intento >= max_retries:
                raise ultimo_error

            espera = min(retry_after, 15) if retry_after is not None else 2 ** intento
            time.sleep(espera)

        except requests.RequestException as error:
            ultimo_error = GeminiError(f"No fue posible conectar con Gemini: {error}")
            if intento >= max_retries:
                raise ultimo_error
            time.sleep(2 ** intento)

    raise GeminiError("Gemini no pudo generar una respuesta.")


def extraer_contenido_gemini(respuesta: dict) -> str:
    if not respuesta: raise RuntimeError("Gemini no devolvió respuesta.")
    if "error" in respuesta: raise RuntimeError(f"Gemini devolvió un error: {respuesta['error']}")
    candidatos = respuesta.get("candidates", [])
    if not candidatos: raise RuntimeError("Gemini no devolvió ningún candidato.")
    
    partes = candidatos[0].get("content", {}).get("parts", [])
    texto_final = [p.get("text", "") for p in partes if "text" in p and p.get("text", "")]
    resultado = "\n".join(texto_final).strip()
    
    if resultado: return resultado
    raise RuntimeError("Gemini no devolvió contenido de texto.")


# ============================================================
# MOTOR DE EMBEDDINGS (BÚSQUEDA SEMÁNTICA VECTORIAL)
# ============================================================

def obtener_embedding(texto: str) -> list[float]:
    """Llama a Gemini para convertir texto en un vector matemático."""
    if not GEMINI_API_KEY or not texto.strip():
        return []
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": texto.strip()[:2000]}]}
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("embedding", {}).get("values", [])
    except Exception as e:
        print(f"⚠️ Error generando embedding: {e}")
    return []


def sim_coseno(v1: list[float], v2: list[float]) -> float:
    """Calcula la similitud de concepto entre dos vectores."""
    if not v1 or not v2:
        return 0.0
    a, b = np.array(v1), np.array(v2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def buscar_conocimiento_vectorial(pregunta: str, conocimientos: list[dict[str, Any]], top_k: int = RELEVANCIA_TOP_K) -> list[dict[str, Any]]:
    """Compara la pregunta matemáticamente contra la base de datos entera y extrae el top K."""
    if not conocimientos or not pregunta.strip():
        return []
        
    v_pregunta = obtener_embedding(pregunta)
    if not v_pregunta:
        return conocimientos[:top_k]
        
    resultados = []
    actualizados = False
    
    for item in conocimientos:
        v_item = item.get("embedding", [])
        
        # Genera el vector si la ficha vieja no lo tiene guardado
        if not v_item:
            texto_completo = f"{item.get('titulo', '')}\n{item.get('contenido', '')}\n{item.get('descripcion', '')}"
            v_item = obtener_embedding(texto_completo)
            if v_item:
                item["embedding"] = v_item
                actualizados = True
                
        score = sim_coseno(v_pregunta, v_item)
        
        # Bono de prioridad para textos enseñados a mano
        if item.get("tipo") == "texto":
            score += 0.05
            
        # Umbral: solo guarda los que de verdad tengan relación
        if score > 0.35:
            resultados.append((score, item))
            
    if actualizados:
        guardar_conocimiento(conocimientos)
        
    resultados.sort(key=lambda x: x[0], reverse=True)
    
    seleccionados = [item for _, item in resultados[:top_k]]
    
    print(f"🔎 Búsqueda Vectorial: {len(seleccionados)} resultados relevantes de {len(conocimientos)}.")
    for score, item in resultados[:top_k]:
        print(f"   📌 {item.get('titulo', '')} (score: {score:.3f})")
        
    return seleccionados


def construir_contexto_relevante(pregunta: str, conocimientos: list[dict[str, Any]]) -> str:
    relevantes = buscar_conocimiento_vectorial(pregunta, conocimientos)
    if not relevantes:
        return "No se encontraron registros en la base de datos relacionados con esta solicitud."

    bloques = []
    for idx, item in enumerate(relevantes, 1):
        bloques.append(
            f"REGISTRO {idx}\n"
            f"TIPO: {item.get('tipo', 'desconocido')}\n"
            f"TÍTULO: {item.get('titulo', '')}\n"
            f"CONTENIDO: {item.get('contenido', '')}\n"
            f"DESCRIPCIÓN: {item.get('descripcion', '')}"
        )
    return "\n\n==============================\n\n".join(bloques)


# ============================================================
# GOOGLE DRIVE & UTILIDADES
# ============================================================

GOOGLE_SERVICE_ACCOUNT_EMAIL = os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL")
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

drive_service, drive_session = None, None
DRIVE_ROOT_FOLDER, DRIVE_SHARED_ID = None, None
DRIVE_KNOWLEDGE_FOLDER, DRIVE_FILES_FOLDER, DRIVE_PDF_FOLDER, DRIVE_IMAGES_FOLDER, DRIVE_BACKUPS_FOLDER = None, None, None, None, None

def ahora_iso() -> str: return datetime.now().isoformat()
def generar_id() -> str: return str(uuid.uuid4())
def nombre_seguro(nombre: str) -> str: return re.sub(r"[^a-zA-Z0-9._-]", "_", Path(nombre).name) or f"archivo_{generar_id()}"


def buscar_archivo_drive(nombre: str, folder_id: str, solo_json: bool = False):
    if not drive_service: return None
    nombre_escapado = nombre.replace("'", "\\'")
    query = f"name = '{nombre_escapado}' and '{folder_id}' in parents and trashed = false"
    if solo_json: query += " and mimeType = 'application/json'"
    try:
        parametros = {"q": query, "spaces": "drive", "includeItemsFromAllDrives": True, "supportsAllDrives": True, "fields": "files(id, name, mimeType, size)", "pageSize": 100}
        if DRIVE_SHARED_ID:
            parametros["corpora"] = "drive"
            parametros["driveId"] = DRIVE_SHARED_ID
        archivos = drive_service.files().list(**parametros).execute().get("files", [])
        return archivos[0] if archivos else None
    except Exception:
        return None

def obtener_o_crear_carpeta(nombre: str, parent_id: str):
    ex = buscar_archivo_drive(nombre, parent_id)
    if ex and ex.get("mimeType") == "application/vnd.google-apps.folder": return ex["id"]
    return drive_service.files().create(body={"name": nombre, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}, fields="id", supportsAllDrives=True).execute()["id"]

def inicializar_google_drive():
    global drive_service, drive_session, DRIVE_ROOT_FOLDER, DRIVE_SHARED_ID
    global DRIVE_KNOWLEDGE_FOLDER, DRIVE_FILES_FOLDER, DRIVE_PDF_FOLDER, DRIVE_IMAGES_FOLDER, DRIVE_BACKUPS_FOLDER

    if os.getenv("RENDER") != "true" or not GOOGLE_AVAILABLE: return
    if not all([GOOGLE_SERVICE_ACCOUNT_EMAIL, GOOGLE_PRIVATE_KEY, GOOGLE_PROJECT_ID, GOOGLE_DRIVE_FOLDER_ID]): return

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
    except Exception as error:
        drive_service = None

def subir_archivo_drive(ruta: Path, folder_id: str, nombre: str | None = None):
    if not drive_service or not ruta.exists(): return None
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
    if not drive_service or not drive_session: return False
    try:
        resp = drive_session.get(f"https://www.googleapis.com/drive/v3/files/{file_id}", params={"alt": "media", "supportsAllDrives": "true"}, timeout=120)
        if resp.status_code == 200 and len(resp.content) > 0:
            destino.parent.mkdir(parents=True, exist_ok=True)
            tmp = destino.parent / f".tmp_{generar_id()}"
            with open(tmp, "wb") as f: f.write(resp.content)
            os.replace(tmp, destino)
            return True
    except Exception:
        pass
    return False

def validar_json_conocimiento(ruta: Path) -> tuple[bool, list[dict[str, Any]]]:
    if not ruta.exists() or ruta.stat().st_size == 0: return False, []
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            data = json.load(archivo)
        if isinstance(data, list) and all(isinstance(i, dict) for i in data): return True, data
    except Exception:
        pass
    return False, []

def sincronizar_conocimiento_desde_drive():
    if not drive_service or not DRIVE_KNOWLEDGE_FOLDER: return
    try:
        ex = buscar_archivo_drive("penaguillo.json", DRIVE_KNOWLEDGE_FOLDER, solo_json=True)
        if ex:
            tmp = CONOCIMIENTO_DIR / "penaguillo_drive.tmp"
            if descargar_archivo_drive(ex["id"], tmp):
                if validar_json_conocimiento(tmp)[0]:
                    os.replace(tmp, ARCHIVO_CONOCIMIENTO)
                    print("✅ Conocimiento sincronizado desde Google Drive.")
                    return
            if tmp.exists(): tmp.unlink()
    except Exception:
        pass

def sincronizar_conocimiento_a_drive():
    if drive_service and DRIVE_KNOWLEDGE_FOLDER and ARCHIVO_CONOCIMIENTO.exists():
        if validar_json_conocimiento(ARCHIVO_CONOCIMIENTO)[0]:
            subir_archivo_drive(ARCHIVO_CONOCIMIENTO, DRIVE_KNOWLEDGE_FOLDER, "penaguillo.json")

def cargar_conocimiento() -> list[dict[str, Any]]:
    if not ARCHIVO_CONOCIMIENTO.exists(): return []
    try:
        with open(ARCHIVO_CONOCIMIENTO, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception:
        return []

def guardar_conocimiento(conocimientos: list[dict[str, Any]]) -> None:
    tmp = CONOCIMIENTO_DIR / f"penaguillo_{generar_id()}.tmp"
    with open(tmp, "w", encoding="utf-8") as archivo:
        json.dump(conocimientos, archivo, ensure_ascii=False, indent=2)
    os.replace(tmp, ARCHIVO_CONOCIMIENTO)
    sincronizar_conocimiento_a_drive()

def crear_backup() -> str | None:
    if not ARCHIVO_CONOCIMIENTO.exists(): return None
    valido, data = validar_json_conocimiento(ARCHIVO_CONOCIMIENTO)
    if not valido: return None
    backup_path = BACKUP_DIR / f"penaguillo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    tmp = BACKUP_DIR / f"backup_{generar_id()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as archivo:
            json.dump(data, archivo, ensure_ascii=False, indent=2)
        os.replace(tmp, backup_path)
        if drive_service and DRIVE_BACKUPS_FOLDER:
            subir_archivo_drive(backup_path, DRIVE_BACKUPS_FOLDER, backup_path.name)
        return str(backup_path)
    except Exception:
        return None


def sincronizar_pdf_a_drive(ruta_pdf: Path):
    if drive_service and DRIVE_PDF_FOLDER: subir_archivo_drive(ruta_pdf, DRIVE_PDF_FOLDER, ruta_pdf.name)

def sincronizar_imagen_a_drive(ruta_imagen: Path):
    if drive_service and DRIVE_IMAGES_FOLDER: subir_archivo_drive(ruta_imagen, DRIVE_IMAGES_FOLDER, ruta_imagen.name)


# ============================================================
# SYSTEM PROMPT
# ============================================================

def cargar_system_prompt() -> str:
    if not PROMPT_FILE.exists(): return "Eres Penaguillo, el asistente virtual."
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as archivo:
            return archivo.read().strip()
    except Exception:
        return "Eres Penaguillo."

SYSTEM_PROMPT_BASE = cargar_system_prompt()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Penaguillo IA", version="7.0.0", description="Backend del asistente inteligente Penaguillo (Búsqueda Vectorial)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/archivos", StaticFiles(directory=str(ARCHIVOS_DIR)), name="archivos")

MAX_PDF_SIZE = 50 * 1024 * 1024
MAX_IMAGE_SIZE = 15 * 1024 * 1024
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}
EXTENSIONES_PDF = {".pdf"}

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
# CHAT ENDPOINT CON BÚSQUEDA VECTORIAL
# ============================================================

@app.post("/chat")
def chat(data: ChatRequest):
    mensaje = data.message.strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        tiempo_inicio_chat = time.time()
        conocimientos = cargar_conocimiento()

        # Armar un "query aumentado" si la pregunta es muy corta para captar el contexto anterior (ej. "quienes son ellos")
        query_vectorial = mensaje
        if data.history and len(mensaje.split()) <= 4:
            query_vectorial = f"{data.history[-1].content} {mensaje}"

        contexto_relevante = construir_contexto_relevante(query_vectorial, conocimientos)

        system_prompt = (
            SYSTEM_PROMPT_BASE
            + "\n\n==============================\n"
            + "BASE DE CONOCIMIENTO RELEVANTE (EXTRACCIÓN VECTORIAL)\n"
            + "==============================\n"
            + "Utiliza SOLO la información proporcionada a continuación para responder. La pregunta ACTUAL del usuario tiene prioridad.\n\n"
            + contexto_relevante
            + "\n\n==============================\n"
            + "REGLAS ESTRICTAS DE RESPUESTA Y VERACIDAD\n"
            + "==============================\n"
            + "1. Si el usuario pregunta por un 'equipo', 'grupo' o 'área' (como Modernización Tecnológica, Operaciones, etc.), DEBES nombrar a TODOS los colaboradores asignados formalmente a ese equipo en el registro. Jamás omitas a alguien.\n"
            + "2. Queda ABSOLUTAMENTE PROHIBIDO asumir, adivinar o inferir que un área o persona atiende un sistema (como ERP SAP, servidores, soporte) a menos que se declare expresamente en el contexto.\n"
            + "3. Si la relación o el responsable NO está explícitamente especificado, responde claramente: 'No tengo un responsable confirmado para esa solicitud en la información que manejo.'\n"
            + "4. Copia y pega de forma EXACTA e ÍNTEGRA los correos electrónicos y teléfonos sin alterar ni recortar dominios (.co o .com).\n"
        )

        mensajes_api = [{"role": "system", "content": system_prompt}]

        if data.history:
            for msg in data.history[-MAX_MENSAJES_HISTORIAL:]:
                if msg.content.strip() and msg.role in ("user", "assistant"):
                    mensajes_api.append({"role": msg.role, "content": msg.content.strip()})

        mensajes_api.append({"role": "user", "content": mensaje})

        respuesta = generar_con_gemini(model=CHAT_MODEL, messages=mensajes_api)
        contenido = extraer_contenido_gemini(respuesta)

        print(f"⏱️ Tiempo total /chat: {time.time() - tiempo_inicio_chat:.2f}s")
        return {"ok": True, "response": contenido or ""}

    except GeminiError as error:
        if error.status_code == 429: raise HTTPException(status_code=429, detail=str(error))
        raise HTTPException(status_code=502, detail=str(error))
    except Exception as error:
        if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error).upper():
            raise HTTPException(status_code=429, detail="Límite de API de Gemini excedido.")
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================
# ENSEÑAR TEXTO
# ============================================================

@app.post("/ensenar")
def ensenar(data: EnsenarRequest):
    texto = data.conocimiento.strip()
    if not texto: raise HTTPException(status_code=400, detail="El conocimiento no puede estar vacío.")

    try:
        conocimientos = cargar_conocimiento()
        palabras = texto.split()
        titulo_dinamico = " ".join(palabras[:8]) + ("..." if len(palabras) > 8 else "")
        
        # Generar embedding del nuevo texto
        vec = obtener_embedding(f"{titulo_dinamico}\n{texto}")

        nuevo = {
            "id": generar_id(),
            "tipo": "texto",
            "titulo": titulo_dinamico,
            "contenido": texto,
            "descripcion": f"Información importante sobre: {titulo_dinamico}",
            "embedding": vec,
            "fecha": ahora_iso(),
        }

        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)

        return {"ok": True, "mensaje": "Conocimiento guardado.", "conocimiento": nuevo, "total": len(conocimientos)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================
# ENSEÑAR IMAGEN
# ============================================================

@app.post("/ensenar-imagen")
async def ensenar_imagen(file: UploadFile = File(...)):
    nombre_original = file.filename or "imagen"
    if Path(nombre_original).suffix.lower() not in EXTENSIONES_IMAGEN:
        raise HTTPException(status_code=400, detail="Formato de imagen no permitido.")

    contenido = await file.read()
    if len(contenido) > MAX_IMAGE_SIZE: raise HTTPException(status_code=400, detail="La imagen supera 15 MB.")

    nombre = f"{uuid.uuid4().hex}_{nombre_seguro(nombre_original)}"
    ruta = IMAGENES_DIR / nombre

    try:
        with open(ruta, "wb") as archivo: archivo.write(contenido)
        
        # Visión de Gemini
        imagen_base64 = base64.b64encode(contenido).decode("utf-8")
        mime = "image/png" if ruta.suffix.lower() == ".png" else "image/jpeg"
        prompt = "Analiza cuidadosamente esta imagen para alimentar la base de conocimiento de Penaguillo. Extrae equipos, textos visibles, diagramas y procesos de forma estructurada."
        resp = generar_con_gemini(model=VISION_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{imagen_base64}"}}]}])
        descripcion = extraer_contenido_gemini(resp)
        
        # Embedding del resultado
        vec = obtener_embedding(f"{nombre_original}\n{descripcion}")
        
        conocimientos = cargar_conocimiento()
        nuevo = {
            "id": generar_id(), "tipo": "imagen", "titulo": nombre_original,
            "contenido": descripcion, "descripcion": descripcion,
            "archivo": f"/archivos/imagenes/{nombre}", "nombre_archivo": nombre_original,
            "embedding": vec, "fecha": ahora_iso(),
        }

        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)
        sincronizar_imagen_a_drive(ruta)

        return {"ok": True, "mensaje": "Imagen aprendida correctamente.", "conocimiento": nuevo, "total": len(conocimientos)}

    except Exception as error:
        if ruta.exists(): ruta.unlink()
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================
# ENSEÑAR PDF
# ============================================================

def analizar_pagina_pdf_con_vision(pagina: fitz.Page, numero_pagina: int) -> str:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY no configurada.")
    imagen_bytes = pagina.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).tobytes("png")
    imagen_base64 = base64.b64encode(imagen_bytes).decode("utf-8")
    prompt = f"Analiza esta página {numero_pagina} de un PDF para la base de conocimiento de Penaguillo. Extrae todo texto legible, tablas, procesos y datos importantes en texto estructurado."
    respuesta = generar_con_gemini(model=VISION_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_base64}"}}]}])
    return extraer_contenido_gemini(respuesta)

@app.post("/ensenar-pdf")
async def ensenar_pdf(file: UploadFile = File(...)):
    nombre_original = file.filename or "documento.pdf"
    if Path(nombre_original).suffix.lower() not in EXTENSIONES_PDF:
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    contenido = await file.read()
    if len(contenido) > MAX_PDF_SIZE: raise HTTPException(status_code=400, detail="El PDF supera los 50 MB.")

    nombre = f"{uuid.uuid4().hex}_{nombre_seguro(nombre_original)}"
    ruta = PDF_DIR / nombre
    documento = None

    try:
        with open(ruta, "wb") as archivo: archivo.write(contenido)
        documento = fitz.open(str(ruta))
        numero_paginas = documento.page_count
        
        texto_extraido = "\n\n".join([pagina.get_text("text", sort=True).strip() for pagina in documento if pagina.get_text("text", sort=True).strip()])
        es_escaneado = len(re.sub(r"\s+", "", texto_extraido)) < 50
        
        if not es_escaneado:
            contenido_final = texto_extraido
        else:
            paginas_vision = []
            for indice, pagina in enumerate(documento, start=1):
                texto_pagina = analizar_pagina_pdf_con_vision(pagina, indice)
                if texto_pagina.strip(): paginas_vision.append(f"PÁGINA {indice}\n{texto_pagina}")
            contenido_final = "\n\n".join(paginas_vision)

        documento.close()
        
        # Generar embedding del documento
        vec = obtener_embedding(f"{nombre_original}\n{contenido_final[:2000]}")
        
        conocimientos = cargar_conocimiento()
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

        return {"ok": True, "mensaje": "PDF aprendido correctamente.", "conocimiento": nuevo, "total": len(conocimientos)}

    except Exception as error:
        if documento: documento.close()
        if ruta.exists(): ruta.unlink()
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================
# RUTAS DE ADMINISTRACIÓN
# ============================================================

@app.get("/conocimiento")
def obtener_conocimiento():
    try:
        return {"ok": True, "total": len(cargar_conocimiento()), "conocimientos": cargar_conocimiento()}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.delete("/conocimiento")
def eliminar_conocimiento(data: EliminarRequest):
    try:
        conocimientos = cargar_conocimiento()
        encontrado = next((item for item in conocimientos if str(item.get("id")) == str(data.id)), None)
        if not encontrado: raise HTTPException(status_code=404, detail="No se encontró ese conocimiento.")

        nuevos_conocimientos = [item for item in conocimientos if str(item.get("id")) != str(data.id)]
        guardar_conocimiento(nuevos_conocimientos)

        archivo_relativo = encontrado.get("archivo")
        if archivo_relativo:
            ruta_archivo = ARCHIVOS_DIR / archivo_relativo.replace("/archivos/", "", 1).lstrip("/")
            if ruta_archivo.exists(): ruta_archivo.unlink()

        return {"ok": True, "mensaje": "Eliminado correctamente.", "total": len(nuevos_conocimientos)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/backups")
def listar_backups():
    archivos = sorted(BACKUP_DIR.glob("penaguillo_*.json"), key=lambda a: a.stat().st_mtime, reverse=True)
    return {"ok": True, "total": len(archivos), "backups": [{"nombre": a.name, "fecha": datetime.fromtimestamp(a.stat().st_mtime).isoformat()} for a in archivos]}

@app.get("/")
def root():
    return {
        "ok": True, "app": "Penaguillo IA", "version": "7.0.0", "engine": "Búsqueda Vectorial (Embeddings)",
        "chat_model": CHAT_MODEL, "conocimientos": len(cargar_conocimiento()), "retrieval_top_k": RELEVANCIA_TOP_K
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    print("🚀 Iniciando Penaguillo IA v7.0 (Motor Vectorial)...")
    inicializar_google_drive()
    try:
        print(f"📚 Conocimientos disponibles: {len(cargar_conocimiento())}")
    except Exception as error:
        print(f"⚠️ Error cargando JSON: {error}")
    print("✅ Penaguillo IA iniciado y listo.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)