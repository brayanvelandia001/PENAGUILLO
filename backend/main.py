# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 7.1 (Vectorial Definitiva + Control de Cuota Anti-429)
#
# PROVEEDOR DE IA: Google Gemini Native API
# MODELOS:
# - CHAT/VISION: gemini-3.5-flash-lite
# - EMBEDDINGS: text-embedding-004
#
# Cero mantenimiento de reglas de texto. Búsqueda semántica real.
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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ============================================================
# GOOGLE DRIVE & CARPETAS
# ============================================================
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import AuthorizedSession
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

CONOCIMIENTO_DIR = BASE_DIR / "storage" / "conocimiento" if os.getenv("RENDER") == "true" else BASE_DIR / "conocimiento"
ARCHIVOS_DIR = CONOCIMIENTO_DIR / "archivos"
PDF_DIR = ARCHIVOS_DIR / "pdf"
IMAGENES_DIR = ARCHIVOS_DIR / "imagenes"
BACKUP_DIR = CONOCIMIENTO_DIR / "backups"
ARCHIVO_CONOCIMIENTO = CONOCIMIENTO_DIR / "penaguillo.json"
PROMPT_FILE = BASE_DIR / "conocimiento" / "prompt.txt"

for d in [CONOCIMIENTO_DIR, ARCHIVOS_DIR, PDF_DIR, IMAGENES_DIR, BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

# ============================================================
# CONFIGURACIÓN GEMINI
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
# UTILIDADES
# ============================================================
def ahora_iso() -> str: return datetime.now().isoformat()
def generar_id() -> str: return str(uuid.uuid4())
def nombre_seguro(nombre: str) -> str: return re.sub(r"[^a-zA-Z0-9._-]", "_", Path(nombre).name) or f"archivo_{generar_id()}"

# ============================================================
# GEMINI GENERACIÓN DE CONTENIDO
# ============================================================
def extraer_retry_after(respuesta: requests.Response, mensaje_error: str) -> int | None:
    valor_header = respuesta.headers.get("Retry-After")
    if valor_header:
        try:
            if int(valor_header) >= 0: return int(valor_header)
        except: pass
    for patron in [r"retry in ([0-9]+(?:\.[0-9]+)?)s", r"retryDelay.*?([0-9]+)s", r"seconds.*?([0-9]+)"]:
        coincidencia = re.search(patron, mensaje_error or "", re.IGNORECASE)
        if coincidencia:
            try: return max(1, int(float(coincidencia.group(1))))
            except: pass
    return None

def generar_con_gemini(*, model: str, messages: list, max_retries: int = 2):
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY no está configurada.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    system_instruction, gemini_contents = None, []

    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            system_instruction = {"parts": [{"text": msg.get("content")}]}
            continue
        parts = []
        if isinstance(msg.get("content"), str) and msg.get("content").strip():
            parts.append({"text": msg.get("content")})
        if parts: gemini_contents.append({"role": "model" if msg.get("role") == "assistant" else "user", "parts": parts})

    payload = {"contents": gemini_contents, "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.3}}
    if system_instruction: payload["systemInstruction"] = system_instruction

    for intento in range(1, max_retries + 1):
        try:
            respuesta = requests.post(url, headers=headers, json=payload, timeout=120)
            if respuesta.status_code == 200: return respuesta.json()
            mensaje_error = respuesta.text[:5000]
            retry_after = extraer_retry_after(respuesta, mensaje_error)
            if respuesta.status_code in (400, 403, 404) or intento >= max_retries:
                raise RuntimeError(f"HTTP {respuesta.status_code}: {mensaje_error}")
            time.sleep(min(retry_after, 15) if retry_after else 2 ** intento)
        except requests.RequestException as error:
            if intento >= max_retries: raise RuntimeError(str(error))
            time.sleep(2 ** intento)
    raise RuntimeError("Gemini falló.")

def extraer_contenido_gemini(respuesta: dict) -> str:
    if not respuesta or "candidates" not in respuesta: raise RuntimeError("Sin respuesta.")
    partes = respuesta["candidates"][0].get("content", {}).get("parts", [])
    res = "\n".join([p.get("text", "") for p in partes if "text" in p]).strip()
    if res: return res
    raise RuntimeError("Sin texto.")

# ============================================================
# EMBEDDINGS Y MOTOR VECTORIAL ROBUSTO
# ============================================================
def obtener_embedding(texto: str, max_retries: int = 3) -> list[float]:
    """Genera el vector matemático asegurando que Google no nos bloquee (Manejo de Error 429)."""
    if not GEMINI_API_KEY or not texto.strip(): return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {"model": f"models/{EMBEDDING_MODEL}", "content": {"parts": [{"text": texto.strip()[:2000]}]}}
    
    for intento in range(max_retries):
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("embedding", {}).get("values", [])
            elif resp.status_code == 429:
                print(f"⏳ Límite de API de vectores alcanzado. Pausando 3 segundos (Intento {intento+1})...")
                time.sleep(3) # Pausa obligatoria para evitar el bloqueo total
            else:
                break
        except Exception:
            time.sleep(2)
    return []

def sim_coseno(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2: return 0.0
    a, b = np.array(v1), np.array(v2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def buscar_conocimiento_vectorial(pregunta: str, conocimientos: list, top_k: int = RELEVANCIA_TOP_K) -> list:
    if not conocimientos or not pregunta.strip(): return []
    
    v_pregunta = obtener_embedding(pregunta)
    if not v_pregunta: return conocimientos[:top_k]
    
    resultados, actualizados = [], False
    
    for item in conocimientos:
        v_item = item.get("embedding", [])
        
        # Si un registro no tiene vector (ej. Edgar), se lo creamos.
        if not v_item:
            texto_completo = f"{item.get('titulo', '')}\n{item.get('contenido', '')}\n{item.get('descripcion', '')}"
            v_item = obtener_embedding(texto_completo)
            if v_item:
                item["embedding"] = v_item
                actualizados = True
                time.sleep(1.5) # 🔥 MAGIA: Esto evita que se sature la API gratuita al iniciar.
                
        score = sim_coseno(v_pregunta, v_item)
        if score > 0.35: resultados.append((score, item))
            
    if actualizados:
        guardar_conocimiento(conocimientos) # Guardamos los nuevos vectores para no volver a calcularlos
        
    resultados.sort(key=lambda x: x[0], reverse=True)
    seleccionados = [i for _, i in resultados[:top_k]]
    
    print(f"🔎 Vectorial: {len(seleccionados)} encontrados.")
    return seleccionados

def construir_contexto_relevante(pregunta: str, conocimientos: list) -> str:
    relevantes = buscar_conocimiento_vectorial(pregunta, conocimientos)
    if not relevantes: return "No se encontró información."
    bloques = [f"REGISTRO {idx}\nTIPO: {item.get('tipo', '')}\nTÍTULO: {item.get('titulo', '')}\nCONTENIDO: {item.get('contenido', '')}" for idx, item in enumerate(relevantes, 1)]
    return "\n\n==============================\n\n".join(bloques)

# ============================================================
# PERSISTENCIA Y GOOGLE DRIVE
# ============================================================
GOOGLE_SERVICE_ACCOUNT_EMAIL = os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL")
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

drive_service, drive_session = None, None
DRIVE_ROOT_FOLDER, DRIVE_SHARED_ID = None, None
DRIVE_KNOWLEDGE_FOLDER, DRIVE_FILES_FOLDER, DRIVE_PDF_FOLDER, DRIVE_IMAGES_FOLDER, DRIVE_BACKUPS_FOLDER = None, None, None, None, None

def inicializar_google_drive():
    global drive_service, drive_session, DRIVE_ROOT_FOLDER, DRIVE_SHARED_ID, DRIVE_KNOWLEDGE_FOLDER, DRIVE_FILES_FOLDER, DRIVE_PDF_FOLDER, DRIVE_IMAGES_FOLDER, DRIVE_BACKUPS_FOLDER
    if os.getenv("RENDER") != "true" or not GOOGLE_AVAILABLE: return
    if not all([GOOGLE_SERVICE_ACCOUNT_EMAIL, GOOGLE_PRIVATE_KEY, GOOGLE_PROJECT_ID, GOOGLE_DRIVE_FOLDER_ID]): return
    try:
        pk = GOOGLE_PRIVATE_KEY.replace("\\n", "\n")
        info = {"type": "service_account", "project_id": GOOGLE_PROJECT_ID, "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID", ""), "private_key": pk, "client_email": GOOGLE_SERVICE_ACCOUNT_EMAIL, "client_id": os.getenv("GOOGLE_CLIENT_ID", ""), "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL", "")}
        cred = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
        drive_service = build("drive", "v3", credentials=cred, cache_discovery=False)
        drive_session = AuthorizedSession(cred)
        DRIVE_ROOT_FOLDER = GOOGLE_DRIVE_FOLDER_ID
        DRIVE_SHARED_ID = drive_service.files().get(fileId=DRIVE_ROOT_FOLDER, fields="id, driveId", supportsAllDrives=True).execute().get("driveId")
        DRIVE_KNOWLEDGE_FOLDER = obtener_o_crear_carpeta("conocimiento", DRIVE_ROOT_FOLDER)
        DRIVE_FILES_FOLDER = obtener_o_crear_carpeta("archivos", DRIVE_ROOT_FOLDER)
        DRIVE_PDF_FOLDER = obtener_o_crear_carpeta("pdf", DRIVE_FILES_FOLDER)
        DRIVE_IMAGES_FOLDER = obtener_o_crear_carpeta("imagenes", DRIVE_FILES_FOLDER)
        DRIVE_BACKUPS_FOLDER = obtener_o_crear_carpeta("backups", DRIVE_KNOWLEDGE_FOLDER)
        sincronizar_conocimiento_desde_drive()
    except Exception: drive_service = None

def buscar_archivo_drive(nombre: str, folder_id: str, solo_json: bool = False):
    if not drive_service: return None
    q = f"name = '{nombre.replace('\'', '\\\'')}' and '{folder_id}' in parents and trashed = false"
    if solo_json: q += " and mimeType = 'application/json'"
    try:
        p = {"q": q, "spaces": "drive", "includeItemsFromAllDrives": True, "supportsAllDrives": True, "fields": "files(id, name, mimeType)", "pageSize": 10}
        if DRIVE_SHARED_ID: p.update({"corpora": "drive", "driveId": DRIVE_SHARED_ID})
        archivos = drive_service.files().list(**p).execute().get("files", [])
        return archivos[0] if archivos else None
    except: return None

def obtener_o_crear_carpeta(nombre: str, parent_id: str):
    ex = buscar_archivo_drive(nombre, parent_id)
    if ex and ex.get("mimeType") == "application/vnd.google-apps.folder": return ex["id"]
    return drive_service.files().create(body={"name": nombre, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}, fields="id", supportsAllDrives=True).execute()["id"]

def subir_archivo_drive(ruta: Path, folder_id: str, nombre: str | None = None):
    if not drive_service or not ruta.exists(): return None
    nom = nombre or ruta.name
    ex = buscar_archivo_drive(nom, folder_id, solo_json=(nom.lower() == "penaguillo.json"))
    mime = "application/json" if ruta.suffix.lower() == ".json" else "application/octet-stream"
    try:
        media = MediaFileUpload(str(ruta), mimetype=mime, resumable=True)
        if ex: return drive_service.files().update(fileId=ex["id"], media_body=media, supportsAllDrives=True).execute()["id"]
        return drive_service.files().create(body={"name": nom, "mimeType": mime, "parents": [folder_id]}, media_body=media, supportsAllDrives=True).execute()["id"]
    except: return None

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
    except: pass
    return False

def sincronizar_conocimiento_desde_drive():
    if not drive_service or not DRIVE_KNOWLEDGE_FOLDER: return
    try:
        ex = buscar_archivo_drive("penaguillo.json", DRIVE_KNOWLEDGE_FOLDER, solo_json=True)
        if ex:
            tmp = CONOCIMIENTO_DIR / "penaguillo_drive.tmp"
            if descargar_archivo_drive(ex["id"], tmp):
                try:
                    with open(tmp, "r", encoding="utf-8") as f: json.load(f)
                    os.replace(tmp, ARCHIVO_CONOCIMIENTO)
                    print("✅ Sincronizado desde Drive.")
                except: pass
            if tmp.exists(): tmp.unlink()
    except: pass

def cargar_conocimiento() -> list:
    if not ARCHIVO_CONOCIMIENTO.exists(): return []
    try:
        with open(ARCHIVO_CONOCIMIENTO, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def guardar_conocimiento(conocimientos: list):
    tmp = CONOCIMIENTO_DIR / f"tmp_{generar_id()}.json"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(conocimientos, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ARCHIVO_CONOCIMIENTO)
    if drive_service and DRIVE_KNOWLEDGE_FOLDER: subir_archivo_drive(ARCHIVO_CONOCIMIENTO, DRIVE_KNOWLEDGE_FOLDER, "penaguillo.json")

# ============================================================
# FASTAPI Y ENDPOINTS
# ============================================================
app = FastAPI(title="Penaguillo IA", version="7.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/archivos", StaticFiles(directory=str(ARCHIVOS_DIR)), name="archivos")

class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel): message: str; history: list[ChatMessage] = []
class EnsenarRequest(BaseModel): conocimiento: str
class EliminarRequest(BaseModel): id: str

@app.get("/")
def root(): return {"ok": True, "app": "Penaguillo IA", "version": "7.1.0", "motor": "Vectorial"}

# ============================================================
# ENDPOINT CHAT
# ============================================================
@app.post("/chat")
def chat(data: ChatRequest):
    mensaje = data.message.strip()
    if not mensaje: raise HTTPException(status_code=400, detail="Vacío.")
    
    try:
        t_ini = time.time()
        conocimientos = cargar_conocimiento()
        
        # Enriquecer búsqueda si es corta y hay historial
        query_vectorial = f"{data.history[-1].content} {mensaje}" if data.history and len(mensaje.split()) <= 4 else mensaje
        contexto_relevante = construir_contexto_relevante(query_vectorial, conocimientos)

        prompt_base = "Eres Penaguillo."
        if PROMPT_FILE.exists():
            with open(PROMPT_FILE, "r", encoding="utf-8") as f: prompt_base = f.read().strip()

        sys_prompt = (
            f"{prompt_base}\n\n"
            "==============================\nBASE DE DATOS ENCONTRADA\n==============================\n"
            f"{contexto_relevante}\n\n"
            "==============================\nREGLAS OBLIGATORIAS\n==============================\n"
            "1. Si preguntan por integrantes de un área (ej. Modernización), lista a TODAS las personas que aparezcan en la base de datos para esa área.\n"
            "2. NO INVENTES responsables (ej. ERP SAP). Si el contexto no asigna a alguien expresamente, di que no tienes el dato confirmado.\n"
            "3. Manten los correos y teléfonos tal cual como están en la base de datos."
        )

        mensajes_api = [{"role": "system", "content": sys_prompt}]
        for msg in data.history[-MAX_MENSAJES_HISTORIAL:]:
            if msg.content.strip(): mensajes_api.append({"role": msg.role, "content": msg.content.strip()})
        mensajes_api.append({"role": "user", "content": mensaje})

        resp = generar_con_gemini(model=CHAT_MODEL, messages=mensajes_api)
        print(f"⏱️ /chat: {time.time() - t_ini:.2f}s")
        return {"ok": True, "response": extraer_contenido_gemini(resp)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ENDPOINTS ENSEÑAR Y ELIMINAR
# ============================================================
@app.post("/ensenar")
def ensenar(data: EnsenarRequest):
    texto = data.conocimiento.strip()
    if not texto: raise HTTPException(status_code=400)
    
    conocimientos = cargar_conocimiento()
    palabras = texto.split()
    titulo = " ".join(palabras[:8]) + ("..." if len(palabras) > 8 else "")
    
    # Crea el vector de la nueva información inmediatamente
    vec = obtener_embedding(f"{titulo}\n{texto}")
    
    nuevo = {"id": generar_id(), "tipo": "texto", "titulo": titulo, "contenido": texto, "descripcion": f"Info: {titulo}", "embedding": vec, "fecha": ahora_iso()}
    conocimientos.append(nuevo)
    guardar_conocimiento(conocimientos)
    return {"ok": True, "conocimiento": nuevo, "total": len(conocimientos)}

@app.delete("/conocimiento")
def eliminar_conocimiento(data: EliminarRequest):
    conocimientos = cargar_conocimiento()
    nuevos = [i for i in conocimientos if str(i.get("id")) != str(data.id)]
    guardar_conocimiento(nuevos)
    return {"ok": True, "total": len(nuevos)}

@app.get("/conocimiento")
def obtener_conocimiento(): return {"ok": True, "conocimientos": cargar_conocimiento()}

# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
def startup_event():
    print("🚀 Iniciando Penaguillo v7.1 Vectorial...")
    inicializar_google_drive()
    print("✅ Listo.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)