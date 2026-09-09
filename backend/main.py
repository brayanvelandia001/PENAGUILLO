# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# VERSIÓN 7.5
#
# PROVEEDOR:
# - Google Gemini Native API
#
# MODELOS:
# - CHAT/VISION: gemini-3.5-flash-lite
#
# CAMBIOS 7.3:
# - Embeddings deshabilitados temporalmente porque
#   text-embedding-004 devuelve HTTP 404.
# - Búsqueda textual rápida y priorizada.
# - Detección directa de equipos.
# - Detección directa de personas.
# - Protección de correos y teléfonos literales.
# - Mejor recuperación de equipos completos.
# - Evita saturar scores en 1.000.
# - No altera subida de imágenes.
# - No altera subida de PDF.
# - Google Drive conservado.
# ============================================================

import base64
import hashlib
import json
import os
import re
import sys
import time
import uuid
import unicodedata

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

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

# ============================================================
# EMBEDDINGS
# ============================================================
#
# IMPORTANTE:
#
# text-embedding-004 está devolviendo HTTP 404 en la API
# que está utilizando actualmente Penaguillo.
#
# Por eso se deshabilita para evitar:
# - llamadas innecesarias
# - errores 404
# - lentitud
# - regeneración de embeddings en cada búsqueda
#
# La búsqueda textual continúa funcionando.
# ============================================================

EMBEDDINGS_HABILITADOS = False

EMBEDDING_MODEL = "text-embedding-004"

CHAT_MODEL = GEMINI_MODEL
VISION_MODEL = GEMINI_MODEL

MAX_OUTPUT_TOKENS = 1200

RELEVANCIA_TOP_K = 6

MAX_REGISTROS_CONTEXTO = 6

MAX_MENSAJES_HISTORIAL = 6

# Límite del contexto local enviado a Gemini.
MAX_KB_CONOCIMIENTO_CHAT = 16
MAX_CHARS_CONOCIMIENTO_CHAT = (
    MAX_KB_CONOCIMIENTO_CHAT * 1024
)

UMBRAL_VECTOR = 0.30


# ============================================================
# PALABRAS IGNORADAS EN RETRIEVAL
# ============================================================
# Palabras que no aportan información útil para identificar
# personas, equipos, áreas o temas dentro del conocimiento.
# Se usan normalizadas porque normalizar_texto() elimina tildes.
# ============================================================

PALABRAS_IGNORADAS_RETRIEVAL = {
    "hola",
    "holi",
    "hey",
    "buenas",
    "buenos",
    "dias",
    "tardes",
    "noches",
    "gracias",
    "por",
    "favor",
    "ok",
    "okay",
    "vale",
    "si",
    "no",
    "listo",
    "perfecto",
    "jaja",
    "jajaja",
}


if not GEMINI_API_KEY:
    print(
        "⚠️ ADVERTENCIA: GEMINI_API_KEY no está configurada."
    )
else:
    print("🔑 GEMINI_API_KEY: configurada")

print(
    "🔎 Motor de búsqueda: "
    "textual prioritaria"
)

if not EMBEDDINGS_HABILITADOS:
    print(
        "ℹ️ Embeddings deshabilitados "
        "temporalmente."
    )


# ============================================================
# EQUIPOS CONOCIDOS
# ============================================================

EQUIPOS_CONOCIDOS = {
    "modernizacion": [
        "modernizacion",
        "modernizacion tecnologica",
    ],
    "operaciones": [
        "operaciones",
    ],
    "optimizacion": [
        "optimizacion",
    ],
    "comercial": [
        "comercial",
    ],
    "nomina": [
        "nomina",
    ],
    "soporte": [
        "soporte",
        "soporte tecnico",
    ],
    "sistemas": [
        "sistemas",
        "tecnologia",
    ],
}


# ============================================================
# INTENCIONES DE EQUIPO
# ============================================================

INTENCIONES_EQUIPO = [
    "quienes conforman",
    "quienes son",
    "quienes integran",
    "integrantes",
    "miembros",
    "personas que conforman",
    "personas del equipo",
    "equipo esta conformado",
    "equipo esta formado",
    "conformado por",
    "formado por",
    "cuantos son",
    "cuantas personas",
    "cuantas personas son",
    "quien forma parte",
    "quienes forman parte",
    "conoces el equipo",
    "conoce el equipo",
    "equipo de",
]


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
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


# ============================================================
# GEMINI — RETRY AFTER
# ============================================================

def extraer_retry_after(
    respuesta: requests.Response,
    mensaje_error: str,
) -> int | None:

    valor_header = respuesta.headers.get("Retry-After")

    if valor_header:
        try:
            segundos = int(valor_header)

            if segundos >= 0:
                return segundos

        except (TypeError, ValueError):
            pass

    patrones = [
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay.*?([0-9]+)s",
        r"seconds.*?([0-9]+)",
    ]

    texto = mensaje_error or ""

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
                    int(float(coincidencia.group(1))),
                )

            except (TypeError, ValueError):
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

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    headers = {
        "Content-Type": "application/json"
    }

    system_instruction = None
    gemini_contents = []

    for msg in messages:

        role = msg.get("role")
        content = msg.get("content")

        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ROLE
        # ----------------------------------------------------

        gemini_role = (
            "model"
            if role == "assistant"
            else "user"
        )

        parts = []

        # ----------------------------------------------------
        # TEXTO
        # ----------------------------------------------------

        if isinstance(content, str):

            if content.strip():

                parts.append(
                    {
                        "text": content
                    }
                )

        # ----------------------------------------------------
        # MULTIMODAL
        # ----------------------------------------------------

        elif isinstance(content, list):

            for item in content:

                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                # TEXTO
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

                # IMAGEN
                elif item_type == "image_url":

                    url_img = (
                        item
                        .get("image_url", {})
                        .get("url", "")
                    )

                    if not url_img.startswith(
                        "data:"
                    ):
                        continue

                    try:

                        encabezado, b64_data = (
                            url_img.split(",", 1)
                        )

                        mime_type = (
                            encabezado
                            .split(":", 1)[1]
                            .split(";", 1)[0]
                            .strip()
                        )

                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": b64_data,
                                }
                            }
                        )

                    except Exception:
                        pass

        if parts:

            gemini_contents.append(
                {
                    "role": gemini_role,
                    "parts": parts,
                }
            )

    payload = {
        "contents": gemini_contents,
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.3,
        },
    }

    if system_instruction:

        payload["systemInstruction"] = (
            system_instruction
        )

    ultimo_error = None

    for intento in range(
        1,
        max_retries + 1,
    ):

        try:

            respuesta = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if respuesta.status_code == 200:
                return respuesta.json()

            mensaje_error = respuesta.text[:5000]

            retry_after = extraer_retry_after(
                respuesta,
                mensaje_error,
            )

            ultimo_error = GeminiError(
                (
                    f"Gemini HTTP "
                    f"{respuesta.status_code}: "
                    f"{mensaje_error}"
                ),
                status_code=respuesta.status_code,
                retry_after=retry_after,
            )

            es_temporal = (
                respuesta.status_code
                in (429, 500, 502, 503, 504)
            )

            if (
                not es_temporal
                or intento >= max_retries
            ):
                raise ultimo_error

            espera = (
                min(retry_after, 15)
                if retry_after is not None
                else 2 ** intento
            )

            time.sleep(espera)

        except requests.RequestException as error:

            ultimo_error = GeminiError(
                "No fue posible conectar con Gemini: "
                f"{error}"
            )

            if intento >= max_retries:
                raise ultimo_error

            time.sleep(2 ** intento)

    raise GeminiError(
        "Gemini no pudo generar una respuesta."
    )


# ============================================================
# EXTRAER RESPUESTA GEMINI
# ============================================================

def extraer_contenido_gemini(
    respuesta: dict,
) -> str:

    if not respuesta:
        raise RuntimeError(
            "Gemini no devolvió respuesta."
        )

    if "error" in respuesta:

        raise RuntimeError(
            "Gemini devolvió un error: "
            f"{respuesta['error']}"
        )

    candidatos = respuesta.get(
        "candidates",
        [],
    )

    if not candidatos:

        raise RuntimeError(
            "Gemini no devolvió ningún candidato."
        )

    partes = (
        candidatos[0]
        .get("content", {})
        .get("parts", [])
    )

    texto_final = [
        p.get("text", "")
        for p in partes
        if "text" in p
        and p.get("text", "")
    ]

    resultado = "\n".join(
        texto_final
    ).strip()

    if resultado:
        return resultado

    raise RuntimeError(
        "Gemini no devolvió contenido de texto."
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def obtener_embedding(
    texto: str,
) -> list[float]:

    """
    Obtiene un embedding cuando los embeddings
    están habilitados.

    Actualmente está deshabilitado porque
    text-embedding-004 devuelve HTTP 404.
    """

    if not EMBEDDINGS_HABILITADOS:
        return []

    if (
        not GEMINI_API_KEY
        or not texto.strip()
    ):
        return []

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{EMBEDDING_MODEL}:embedContent"
        f"?key={GEMINI_API_KEY}"
    )

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": (
            f"models/{EMBEDDING_MODEL}"
        ),
        "content": {
            "parts": [
                {
                    "text": texto.strip()[:2000]
                }
            ]
        },
    }

    try:

        respuesta = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=15,
        )

        if respuesta.status_code == 200:

            return (
                respuesta.json()
                .get("embedding", {})
                .get("values", [])
            )

        print(
            "⚠️ Embedding HTTP "
            f"{respuesta.status_code}: "
            f"{respuesta.text[:500]}"
        )

    except Exception as error:

        print(
            "⚠️ Error generando embedding: "
            f"{error}"
        )

    return []


# ============================================================
# HASH DEL CONOCIMIENTO
# ============================================================

def hash_conocimiento(
    item: dict[str, Any],
) -> str:

    texto = (
        f"{item.get('titulo', '')}\n"
        f"{item.get('contenido', '')}\n"
        f"{item.get('descripcion', '')}"
    )

    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def texto_para_embedding(
    item: dict[str, Any],
) -> str:

    return (
        f"TÍTULO: {item.get('titulo', '')}\n"
        f"CONTENIDO: {item.get('contenido', '')}\n"
        f"DESCRIPCIÓN: {item.get('descripcion', '')}"
    )


# ============================================================
# NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(
    texto: str,
) -> str:

    texto = str(texto or "")

    texto = unicodedata.normalize(
        "NFD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter)
        != "Mn"
    )

    texto = texto.lower()

    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()


def tokens_texto(
    texto: str,
) -> set[str]:

    texto = normalizar_texto(
        texto
    )

    return set(
        re.findall(
            r"[a-z0-9@._+-]+",
            texto,
        )
    )


# ============================================================
# DETECTAR EQUIPO
# ============================================================

def detectar_tema_dinamico(
    pregunta: str,
    conocimientos: list[dict[str, Any]],
) -> str | None:
    """
    Intenta identificar el tema de la pregunta usando los propios
    títulos y contenidos del JSON.

    No depende de una lista fija de equipos o áreas.
    """

    pregunta_n = normalizar_texto(pregunta)
    tokens_pregunta = {
        token
        for token in tokens_texto(pregunta_n)
        if (
            len(token) >= 4
            and token not in PALABRAS_IGNORADAS_RETRIEVAL
        )
    }

    if not tokens_pregunta or not conocimientos:
        return None

    candidatos: dict[str, float] = {}

    for item in conocimientos:
        titulo = normalizar_texto(item.get("titulo", ""))
        contenido = normalizar_texto(item.get("contenido", ""))
        descripcion = normalizar_texto(item.get("descripcion", ""))

        if not titulo:
            continue

        tokens_titulo = tokens_texto(titulo)
        coincidencias_titulo = tokens_pregunta & tokens_titulo

        if not coincidencias_titulo:
            continue

        score = len(coincidencias_titulo) * 2.0

        # Los títulos tienen más valor que el contenido para detectar
        # de qué equipo/área/tema se está hablando.
        if any(
            palabra in titulo
            for palabra in (
                "equipo",
                "area",
                "grupo",
                "departamento",
            )
        ):
            score += 2.0

        tokens_contenido = tokens_texto(
            f"{contenido} {descripcion}"
        )
        score += min(
            len(tokens_pregunta & tokens_contenido) * 0.25,
            2.0,
        )

        # El título completo es una señal extremadamente fuerte.
        if titulo in pregunta_n or pregunta_n in titulo:
            score += 4.0

        candidatos[titulo] = max(
            candidatos.get(titulo, 0.0),
            score,
        )

    if not candidatos:
        return None

    mejor_titulo, mejor_score = max(
        candidatos.items(),
        key=lambda x: x[1],
    )

    # Evita declarar un tema cuando solamente existe una coincidencia
    # débil de una palabra genérica.
    if mejor_score < 2.0:
        return None

    return mejor_titulo


# ============================================================
# DETECTAR INTENCIÓN DE EQUIPO
# ============================================================

def es_intencion_equipo(
    pregunta: str,
) -> bool:

    pregunta_n = normalizar_texto(
        pregunta
    )

    return any(
        intencion in pregunta_n
        for intencion in INTENCIONES_EQUIPO
    )


# ============================================================
# DETECTAR SI REGISTRO ES DE EQUIPO
# ============================================================

def es_registro_equipo(
    item: dict[str, Any],
    tema: str | None,
) -> bool:

    if not tema:
        return False

    titulo = normalizar_texto(
        item.get("titulo", "")
    )

    contenido = normalizar_texto(
        item.get("contenido", "")
    )

    tema_n = normalizar_texto(tema)

    tokens_tema = {
        token
        for token in tokens_texto(tema_n)
        if len(token) >= 4
        and token not in PALABRAS_IGNORADAS_RETRIEVAL
    }

    tokens_titulo = tokens_texto(titulo)
    tokens_contenido = tokens_texto(contenido)

    coincidencias_titulo = (
        tokens_tema & tokens_titulo
    )
    coincidencias_contenido = (
        tokens_tema & tokens_contenido
    )

    es_equipo_titulo = any(
        palabra in titulo
        for palabra in (
            "equipo",
            "area",
            "grupo",
            "departamento",
        )
    )

    tiene_estructura_equipo = (
        "conformado" in contenido
        or "formado" in contenido
        or "integrantes" in contenido
        or "miembros" in contenido
        or "integran" in contenido
    )

    if (
        coincidencias_titulo
        and es_equipo_titulo
    ):
        return True

    if (
        coincidencias_titulo
        and tiene_estructura_equipo
    ):
        return True

    if (
        coincidencias_contenido
        and tiene_estructura_equipo
        and "equipo" in contenido
    ):
        return True

    return False


# ============================================================
# COINCIDENCIA DE PERSONA
# ============================================================

def puntuacion_persona(
    pregunta: str,
    item: dict[str, Any],
) -> float:

    pregunta_n = normalizar_texto(
        pregunta
    )

    texto_item = normalizar_texto(
        (
            f"{item.get('titulo', '')} "
            f"{item.get('contenido', '')} "
            f"{item.get('descripcion', '')}"
        )
    )

    tokens_pregunta = tokens_texto(
        pregunta_n
    )

    tokens_item = tokens_texto(
        texto_item
    )

    if not tokens_pregunta:
        return 0.0

    candidatos = {
        token
        for token in tokens_pregunta
        if (
            len(token) >= 4
            and token not in PALABRAS_IGNORADAS_RETRIEVAL
        )
    }

    if not candidatos:
        return 0.0

    coincidencias = [
        token
        for token in candidatos
        if token in tokens_item
    ]

    if not coincidencias:
        return 0.0

    score = 0.0

    if len(coincidencias) >= 2:
        score += 0.70
    elif len(coincidencias) == 1:
        score += 0.40

    # Refuerzo cuando dos tokens aparecen juntos en el texto.
    for i, token_a in enumerate(coincidencias):
        for token_b in coincidencias[i + 1:]:
            frase_a = f"{token_a} {token_b}"
            frase_b = f"{token_b} {token_a}"

            if (
                frase_a in texto_item
                or frase_b in texto_item
            ):
                score += 0.20

    return min(score, 0.95)


# ============================================================
# SIMILITUD COSENO
# ============================================================

def sim_coseno(
    v1: list[float],
    v2: list[float],
) -> float:

    if not v1 or not v2:
        return 0.0

    try:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na == 0 or nb == 0:
            return 0.0

        return float(
            np.dot(a, b) / (na * nb)
        )

    except Exception:
        return 0.0


# ============================================================
# SIMILITUD TEXTUAL
# ============================================================

def similitud_textual(
    pregunta: str,
    item: dict[str, Any],
    tema: str | None = None,
) -> float:

    pregunta_n = normalizar_texto(pregunta)

    titulo_n = normalizar_texto(
        item.get("titulo", "")
    )

    contenido_n = normalizar_texto(
        item.get("contenido", "")
    )

    descripcion_n = normalizar_texto(
        item.get("descripcion", "")
    )

    if not pregunta_n:
        return 0.0

    tokens_pregunta = tokens_texto(pregunta_n)
    if not tokens_pregunta:
        return 0.0

    tokens_titulo = tokens_texto(titulo_n)
    tokens_contenido = tokens_texto(
        f"{contenido_n} {descripcion_n}"
    )
    tokens_item = (
        tokens_titulo
        | tokens_contenido
    )

    if not tokens_item:
        return 0.0

    coincidencias = (
        tokens_pregunta & tokens_item
    )

    score = 0.0

    # --------------------------------------------------------
    # COINCIDENCIA DE TOKENS
    # --------------------------------------------------------

    if coincidencias:
        score += (
            len(coincidencias)
            / max(len(tokens_pregunta), 1)
        ) * 0.40

    # --------------------------------------------------------
    # TÍTULO — PESO MAYOR
    # --------------------------------------------------------

    coincidencias_titulo = (
        tokens_pregunta & tokens_titulo
    )

    if coincidencias_titulo:
        score += min(
            len(coincidencias_titulo) * 0.12,
            0.30,
        )

    # --------------------------------------------------------
    # FRASE EXACTA
    # --------------------------------------------------------

    if (
        len(pregunta_n) >= 6
        and pregunta_n in titulo_n
    ):
        score += 0.25

    elif (
        len(pregunta_n) >= 10
        and pregunta_n in contenido_n
    ):
        score += 0.15

    # --------------------------------------------------------
    # TEMA DINÁMICO
    # --------------------------------------------------------

    if tema and es_registro_equipo(item, tema):
        score += 0.20

    # --------------------------------------------------------
    # PERSONA
    # --------------------------------------------------------

    score += puntuacion_persona(
        pregunta,
        item,
    ) * 0.45

    return min(score, 0.99)


# ============================================================
# DETECTAR MISMO TEMA
# ============================================================

def pertenece_al_mismo_tema(
    pregunta: str,
    item: dict[str, Any],
    conocimientos: list[dict[str, Any]] | None = None,
) -> bool:

    if not conocimientos:
        return False

    tema = detectar_tema_dinamico(
        pregunta,
        conocimientos,
    )

    if not tema:
        return False

    return es_registro_equipo(
        item,
        tema,
    )


# ============================================================
# VIGENCIA Y RELACIONES DEL CONOCIMIENTO
# ============================================================

def texto_item_completo(item: dict[str, Any]) -> str:
    return normalizar_texto(
        f"{item.get('titulo', '')} "
        f"{item.get('contenido', '')} "
        f"{item.get('descripcion', '')}"
    )


def distancia_textual(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, normalizar_texto(a), normalizar_texto(b)).ratio()


def extraer_nombre_de_linea(linea: str) -> str | None:
    linea = linea.strip(" -*•\t")
    patron = re.match(
        r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+){1,5})\s+[—-]\s+",
        linea,
    )
    if not patron:
        return None
    nombre = patron.group(1).strip()
    return nombre if len(nombre.split()) >= 2 else None


def extraer_personas_conocidas(conocimientos: list[dict[str, Any]]) -> dict[str, str]:
    personas = {}
    for item in conocimientos:
        titulo = str(item.get("titulo", "")).strip()
        if titulo and "—" in titulo:
            candidato = titulo.split("—", 1)[0].strip(" -*•")
            if len(candidato.split()) >= 2:
                personas[normalizar_texto(candidato)] = candidato
        for linea in str(item.get("contenido", "")).splitlines():
            candidato = extraer_nombre_de_linea(linea)
            if candidato:
                personas[normalizar_texto(candidato)] = candidato
    return personas


def encontrar_persona_en_texto(texto: str, conocimientos: list[dict[str, Any]]) -> str | None:
    texto_n = normalizar_texto(texto)
    coincidencias = [
        original for normalizado, original in extraer_personas_conocidas(conocimientos).items()
        if normalizado and normalizado in texto_n
    ]
    return max(coincidencias, key=lambda x: len(x.split())) if coincidencias else None


def extraer_equipos_del_conocimiento(conocimientos: list[dict[str, Any]]) -> list[str]:
    equipos = []
    for item in conocimientos:
        texto = texto_item_completo(item)
        titulo = normalizar_texto(item.get("titulo", ""))
        candidatos = []
        for patron in (
            r"equipo\s+de\s+([a-z0-9áéíóúüñ ._-]+)",
            r"area\s+de\s+([a-z0-9áéíóúüñ ._-]+)",
        ):
            candidatos.extend(re.findall(patron, texto, re.IGNORECASE))
        if "equipo" in titulo:
            despues = titulo.split("equipo", 1)[-1]
            despues = re.sub(r"^(de\s+)?", "", despues).strip()
            if despues:
                candidatos.append(despues)
        for candidato in candidatos:
            candidato = re.sub(
                r"\b(esta|conformado|formado|por|penagos)\b.*$", "", normalizar_texto(candidato)
            ).strip(" .,:;-")
            candidato = re.sub(r"^(?:del|de|al|a)\s+", "", candidato).strip()
            candidato = re.sub(r"\s+(?:de|del|al)$", "", candidato).strip()
            if len(candidato) >= 3 and candidato not in equipos:
                equipos.append(candidato)
    return equipos


def resolver_nombre_equipo(texto_equipo: str, conocimientos: list[dict[str, Any]]) -> str | None:
    objetivo = normalizar_texto(texto_equipo).strip(" .,:;-")
    objetivo = re.sub(r"^(?:del|de|al|a)\s+", "", objetivo).strip()
    objetivo = re.sub(r"\s+(?:de|del|al)$", "", objetivo).strip()
    if not objetivo:
        return None
    equipos = extraer_equipos_del_conocimiento(conocimientos)
    if not equipos:
        return texto_equipo.strip()
    objetivo_tokens = set(tokens_texto(objetivo))
    mejor, mejor_score = None, 0.0
    for equipo in equipos:
        tokens = set(tokens_texto(equipo))
        inter = len(objetivo_tokens & tokens)
        union = max(len(objetivo_tokens | tokens), 1)
        score = max(inter / union, distancia_textual(objetivo, equipo))
        if score > mejor_score:
            mejor, mejor_score = equipo, score
    return mejor if mejor_score >= 0.62 else texto_equipo.strip()


def extraer_cambio_equipo(texto: str, conocimientos: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    persona = encontrar_persona_en_texto(texto, conocimientos)
    if not persona:
        return None, None, None
    texto_n = normalizar_texto(texto)
    patrones = [
        r"paso\s+de\s+(?:el\s+)?(?:equipo\s+de\s+)?(.+?)\s+(?:a|al)\s+(?:el\s+)?(?:equipo\s+de\s+)?(.+)$",
        r"(?:ya\s+no\s+pertenece|dejo\s+de\s+pertenecer).*?(?:equipo\s+de\s+)?(.+?)\s+paso\s+(?:a|al)\s+(?:el\s+)?(?:equipo\s+de\s+)?(.+)$",
    ]
    for patron in patrones:
        match = re.search(patron, texto_n)
        if match:
            anterior = resolver_nombre_equipo(match.group(1), conocimientos)
            actual = resolver_nombre_equipo(match.group(2), conocimientos)
            return persona, anterior, actual
    match = re.search(r"paso\s+a\s+(?:el\s+)?equipo\s+de\s+(.+)$", texto_n)
    if match:
        return persona, None, resolver_nombre_equipo(match.group(1), conocimientos)
    return persona, None, None


def linea_persona_en_equipo(contenido: str, nombre: str) -> str | None:
    nombre_n = normalizar_texto(nombre)
    for linea in contenido.splitlines():
        if nombre_n in normalizar_texto(linea):
            return linea.strip()
    return None


def es_registro_de_equipo_generico(item: dict[str, Any]) -> bool:
    texto = texto_item_completo(item)
    titulo = normalizar_texto(item.get("titulo", ""))
    return "equipo" in titulo or (
        "equipo" in texto and any(x in texto for x in ("conformado", "integrantes", "miembros", "formado por"))
    )


def equipo_corresponde_a_registro(item: dict[str, Any], equipo: str) -> bool:
    objetivo = set(tokens_texto(equipo))
    if not objetivo or not es_registro_de_equipo_generico(item):
        return False
    tokens = set(tokens_texto(texto_item_completo(item)))
    return len(objetivo & tokens) >= 1


def quitar_persona_de_registro_equipo(item: dict[str, Any], nombre: str) -> bool:
    contenido = str(item.get("contenido", ""))
    nombre_n = normalizar_texto(nombre)
    lineas = contenido.splitlines()
    nuevas = [linea for linea in lineas if nombre_n not in normalizar_texto(linea)]
    if len(nuevas) == len(lineas):
        return False
    item["contenido"] = "\n".join(nuevas).strip()
    item["descripcion"] = f"Información actualizada sobre: {item.get('titulo', '')}"
    item["estado"] = "vigente"
    item["fecha_actualizacion"] = ahora_iso()
    item["embedding"] = []
    item["embedding_hash"] = hash_conocimiento(item)
    return True


def agregar_persona_a_registro_equipo(item: dict[str, Any], nombre: str, linea_persona: str) -> bool:
    contenido = str(item.get("contenido", "")).strip()
    if normalizar_texto(nombre) in normalizar_texto(contenido):
        return False
    item["contenido"] = (contenido + ("\n\n" if contenido else "") + linea_persona.strip()).strip()
    item["descripcion"] = f"Información actualizada sobre: {item.get('titulo', '')}"
    item["estado"] = "vigente"
    item["fecha_actualizacion"] = ahora_iso()
    item["embedding"] = []
    item["embedding_hash"] = hash_conocimiento(item)
    return True


def actualizar_ficha_persona(conocimientos: list[dict[str, Any]], nombre: str, equipo_anterior: str | None, equipo_actual: str | None) -> dict[str, Any]:
    nombre_n = normalizar_texto(nombre)
    ficha = None
    for item in conocimientos:
        titulo_n = normalizar_texto(item.get("titulo", ""))
        if titulo_n.startswith(nombre_n) and "equipo" not in titulo_n:
            ficha = item
            break
    if ficha is None:
        ficha = {
            "id": generar_id(),
            "tipo": "persona",
            "titulo": f"{nombre} — Información actual",
            "contenido": f"{nombre} actualmente pertenece al equipo de {equipo_actual or 'no confirmado'}.",
            "descripcion": f"Información actual de {nombre}.",
            "embedding": [],
            "embedding_hash": "",
            "fecha": ahora_iso(),
        }
        conocimientos.append(ficha)
    else:
        contenido = str(ficha.get("contenido", ""))
        contenido = re.sub(r"(?:actualmente|ahora)\s+[^.\n]*?pertenece\s+(?:al\s+)?equipo\s+de\s+[^.\n]*[.]?", "", contenido, flags=re.IGNORECASE).strip()
        if equipo_actual:
            contenido = (contenido + " " + f"{nombre} actualmente pertenece al equipo de {equipo_actual}.").strip()
        ficha["contenido"] = contenido
    ficha["equipo_actual"] = equipo_actual
    ficha["equipo_anterior"] = equipo_anterior
    ficha["estado"] = "vigente"
    ficha["fecha_actualizacion"] = ahora_iso()
    ficha["embedding"] = []
    ficha["embedding_hash"] = hash_conocimiento(ficha)
    return ficha


def reconciliar_relaciones_vigentes(conocimientos: list[dict[str, Any]]) -> bool:
    """Repara automáticamente contradicciones de pertenencia ya existentes."""
    import copy

    original = copy.deepcopy(conocimientos)

    for item in list(conocimientos):
        texto = str(item.get("contenido", ""))
        texto_n = normalizar_texto(texto)
        if "paso" not in texto_n or "equipo" not in texto_n:
            continue
        aplicar_cambio_de_equipo(texto, conocimientos)

    return original != conocimientos


def aplicar_cambio_de_equipo(texto: str, conocimientos: list[dict[str, Any]]) -> dict[str, Any] | None:
    persona, equipo_anterior, equipo_actual = extraer_cambio_equipo(texto, conocimientos)
    if not persona or not equipo_actual:
        return None

    linea_persona = None
    for item in conocimientos:
        linea = linea_persona_en_equipo(str(item.get("contenido", "")), persona)
        if linea:
            linea_persona = linea
            break
    if not linea_persona:
        linea_persona = f"{persona} — Información registrada."

    salidas, entradas = [], []
    for item in conocimientos:
        if not es_registro_de_equipo_generico(item):
            continue
        if equipo_anterior and equipo_corresponde_a_registro(item, equipo_anterior):
            if quitar_persona_de_registro_equipo(item, persona):
                salidas.append(item.get("titulo", ""))
        if equipo_corresponde_a_registro(item, equipo_actual):
            if agregar_persona_a_registro_equipo(item, persona, linea_persona):
                entradas.append(item.get("titulo", ""))

    ficha = actualizar_ficha_persona(conocimientos, persona, equipo_anterior, equipo_actual)
    return {
        "persona": persona,
        "equipo_anterior": equipo_anterior,
        "equipo_actual": equipo_actual,
        "registros_actualizados": len(salidas) + len(entradas) + 1,
        "equipos_actualizados": {"salida": salidas, "entrada": entradas},
        "ficha": ficha,
    }


# ============================================================
# BÚSQUEDA TEXTUAL INTELIGENTE
# ============================================================

def buscar_conocimiento_vectorial(
    pregunta: str,
    conocimientos: list[dict[str, Any]],
    top_k: int = RELEVANCIA_TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieval local optimizado.

    Actualmente funciona sin embeddings para evitar la llamada rota
    a text-embedding-004.

    Características:
    - evalúa todos los registros del JSON;
    - identifica el tema dinámicamente desde los propios registros;
    - prioriza títulos y coincidencias fuertes;
    - reconoce personas por coincidencia textual;
    - para preguntas de equipo conserva el registro completo del equipo;
    - elimina duplicados;
    - aplica un umbral mínimo de relevancia;
    - nunca rellena artificialmente hasta top_k;
    - devuelve como máximo top_k resultados.
    """

    if not conocimientos or not pregunta.strip():
        return []

    top_k = max(1, min(int(top_k), 8))

    print("\n🔎 ==================================")
    print(f"🔎 BUSCANDO: {pregunta}")
    print(
        "🔎 Registros disponibles: "
        f"{len(conocimientos)}"
    )

    pregunta_n = normalizar_texto(pregunta)

    tema = detectar_tema_dinamico(
        pregunta,
        conocimientos,
    )

    intencion_equipo = es_intencion_equipo(
        pregunta
    )

    print(
        "🎯 Tema dinámico detectado: "
        f"{tema or 'ninguno'}"
    )

    v_pregunta = obtener_embedding(pregunta)
    resultados = []

    for indice, item in enumerate(conocimientos):
        if not isinstance(item, dict):
            continue

        score_vectorial = 0.0
        embedding_guardado = item.get("embedding", [])

        if (
            EMBEDDINGS_HABILITADOS
            and v_pregunta
            and embedding_guardado
        ):
            score_vectorial = sim_coseno(
                v_pregunta,
                embedding_guardado,
            )

        score_textual = similitud_textual(
            pregunta,
            item,
            tema,
        )

        if (
            EMBEDDINGS_HABILITADOS
            and v_pregunta
        ):
            score = (
                score_vectorial * 0.65
                + score_textual * 0.35
            )
        else:
            score = score_textual

        score_persona = puntuacion_persona(
            pregunta,
            item,
        )

        if normalizar_texto(item.get("estado", "")) == "historico":
            score *= 0.35

        equipo_actual_item = normalizar_texto(item.get("equipo_actual", ""))
        if equipo_actual_item and equipo_actual_item in pregunta_n:
            score = max(score, 0.90)

        if score_persona:
            score = max(
                score,
                score_persona,
            )

        es_equipo = (
            bool(tema)
            and es_registro_equipo(item, tema)
        )

        # Para una pregunta explícita sobre integrantes, el registro
        # de equipo correspondiente recibe una prioridad fuerte, pero
        # solamente si realmente pertenece al tema detectado.
        if tema and es_equipo:
            if intencion_equipo:
                score = max(score, 0.92)
            else:
                score = max(score, 0.72)

        # Los registros de texto suelen ser más fáciles de consultar,
        # pero el bono es pequeño para no aplastar otras coincidencias.
        if item.get("tipo") == "texto":
            score = min(score + 0.02, 0.99)

        resultados.append({
            "score": min(score, 0.99),
            "score_vectorial": score_vectorial,
            "score_textual": score_textual,
            "score_persona": score_persona,
            "es_equipo": es_equipo,
            "indice": indice,
            "item": item,
        })

    # --------------------------------------------------------
    # ORDENAR
    # --------------------------------------------------------

    resultados.sort(
        key=lambda resultado: (
            resultado["score"],
            resultado["score_textual"],
            resultado["score_persona"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------

    print("🔎 TOP CANDIDATOS:")

    for resultado in resultados[:top_k]:
        item = resultado["item"]
        print(
            "   📌 "
            f"{item.get('titulo', '')} "
            f"| final={resultado['score']:.3f} "
            f"| texto={resultado['score_textual']:.3f} "
            f"| persona={resultado['score_persona']:.3f}"
        )

    # --------------------------------------------------------
    # SELECCIÓN FINAL
    # --------------------------------------------------------

    seleccionados = []
    ids_vistos = set()

    def agregar(item: dict[str, Any]) -> None:
        if len(seleccionados) >= top_k:
            return

        identificador = str(
            item.get("id")
            or id(item)
        )

        if identificador in ids_vistos:
            return

        ids_vistos.add(identificador)
        seleccionados.append(item)

    # 1. Para preguntas de equipo, primero el registro que describe
    #    explícitamente ese equipo.
    if tema and intencion_equipo:
        for resultado in resultados:
            if resultado["es_equipo"]:
                agregar(resultado["item"])

    # 2. Para preguntas de persona, priorizar coincidencias fuertes.
    for resultado in resultados:
        if resultado["score_persona"] >= 0.40:
            agregar(resultado["item"])

    # 3. Agregar solamente resultados con relevancia suficiente.
    #    No se completa la lista con resultados irrelevantes.
    for resultado in resultados:
        score = resultado["score"]

        if score >= 0.28:
            agregar(resultado["item"])

        if len(seleccionados) >= top_k:
            break

    # 4. Si hubo una coincidencia muy fuerte pero quedó por debajo
    #    del umbral debido a la distribución del texto, conservarla.
    if not seleccionados and resultados:
        mejor = resultados[0]
        if mejor["score"] >= 0.20:
            agregar(mejor["item"])

    print(
        "🔎 RESULTADOS FINALES: "
        f"{len(seleccionados)}"
    )

    for item in seleccionados:
        print(
            "   ✅ "
            f"{item.get('titulo', '')}"
        )

    if intencion_equipo:
        print("👥 Intención de equipo detectada.")

    print("🔎 ==================================\n")

    return seleccionados


# ============================================================
# PROTEGER DATOS LITERALES
# ============================================================

def proteger_datos_literales(
    respuesta: str,
    contexto: str,
) -> str:

    """
    Protege datos exactos recuperados del conocimiento.

    Especialmente útil para evitar que Gemini transforme:

    modernizacion@penagos.co
    en
    modernizacion@penagos

    o:

    administrador@penagos.co
    en
    administrador@penagos.c
    """

    if not respuesta or not contexto:
        return respuesta

    # --------------------------------------------------------
    # EMAILS
    # --------------------------------------------------------

    emails = sorted(
        set(
            re.findall(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                contexto,
            )
        ),
        key=len,
        reverse=True,
    )

    for email in emails:

        local = email.split(
            "@",
            1,
        )[0]

        if not local:
            continue

        patron = re.compile(
            rf"\b{re.escape(local)}@[\w.-]+",
            re.IGNORECASE,
        )

        respuesta = patron.sub(
            email,
            respuesta,
        )

    # --------------------------------------------------------
    # TELÉFONOS
    # --------------------------------------------------------

    telefonos = sorted(
        set(
            re.findall(
                r"(?<!\d)\d{7,15}(?!\d)",
                contexto,
            )
        ),
        key=len,
        reverse=True,
    )

    for telefono in telefonos:

        if len(telefono) < 7:
            continue

        # Solo reemplaza si aparece exactamente
        # el mismo número o con separadores.
        patron = re.compile(
            rf"(?<!\d)"
            rf"{re.escape(telefono)}"
            rf"(?!\d)"
        )

        respuesta = patron.sub(
            telefono,
            respuesta,
        )

    return respuesta


# ============================================================
# CONSTRUIR CONTEXTO
# ============================================================

def construir_contexto_relevante(
    pregunta: str,
    conocimientos: list[dict[str, Any]],
) -> str:
    """Construye el contexto local que será enviado a Gemini."""

    pregunta_n = normalizar_texto(pregunta)

    saludos = {
        "hola",
        "holi",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "hey",
        "que tal",
        "como estas",
    }

    if pregunta_n in saludos:
        return (
            "No se requiere consultar la base de conocimiento "
            "para este saludo."
        )

    relevantes = buscar_conocimiento_vectorial(
        pregunta,
        conocimientos,
    )

    if not relevantes:
        return (
            "No se encontraron registros en la "
            "base de datos relacionados con "
            "esta solicitud."
        )

    bloques = []
    caracteres_usados = 0

    for idx, item in enumerate(relevantes, 1):
        bloque = (
            f"REGISTRO {idx}\n"
            f"ID: {item.get('id', '')}\n"
            f"TIPO: {item.get('tipo', 'desconocido')}\n"
            f"TÍTULO: {item.get('titulo', '')}\n"
            f"CONTENIDO: {item.get('contenido', '')}\n"
            f"DESCRIPCIÓN: {item.get('descripcion', '')}\n"
            f"ESTADO: {item.get('estado', 'vigente')}\n"
            f"EQUIPO ACTUAL: {item.get('equipo_actual', '')}\n"
            f"EQUIPO ANTERIOR: {item.get('equipo_anterior', '')}"
        )

        separador = (
            "\n\n"
            "==============================\n\n"
        )

        costo = len(bloque)
        costo_separador = (
            len(separador) if bloques else 0
        )

        if (
            bloques
            and caracteres_usados
            + costo_separador
            + costo
            > MAX_CHARS_CONOCIMIENTO_CHAT
        ):
            break

        if (
            not bloques
            and costo > MAX_CHARS_CONOCIMIENTO_CHAT
        ):
            bloque = bloque[:MAX_CHARS_CONOCIMIENTO_CHAT]
            costo = len(bloque)

        bloques.append(bloque)
        caracteres_usados += costo + costo_separador

    if not bloques:
        return (
            "No se encontraron registros en la "
            "base de datos relacionados con "
            "esta solicitud."
        )

    return (
        "\n\n"
        "==============================\n\n"
    ).join(bloques)


# ============================================================
# GOOGLE DRIVE
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
    return str(uuid.uuid4())


def nombre_seguro(
    nombre: str,
) -> str:

    return (
        re.sub(
            r"[^a-zA-Z0-9.\_-]",
            "_",
            Path(nombre).name,
        )
        or f"archivo_{generar_id()}"
    )


# ============================================================
# DRIVE — BUSCAR ARCHIVO
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
        f"name = '{nombre_escapado}' "
        f"and '{folder_id}' in parents "
        f"and trashed = false"
    )

    if solo_json:

        query += (
            " and mimeType = "
            "'application/json'"
        )

    try:

        parametros = {
            "q": query,
            "spaces": "drive",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "fields": (
                "files(id, name, mimeType, size)"
            ),
            "pageSize": 100,
        }

        if DRIVE_SHARED_ID:

            parametros["corpora"] = "drive"
            parametros["driveId"] = (
                DRIVE_SHARED_ID
            )

        archivos = (
            drive_service
            .files()
            .list(**parametros)
            .execute()
            .get("files", [])
        )

        return (
            archivos[0]
            if archivos
            else None
        )

    except Exception:

        return None


# ============================================================
# DRIVE — CREAR CARPETA
# ============================================================

def obtener_o_crear_carpeta(
    nombre: str,
    parent_id: str,
):

    existente = buscar_archivo_drive(
        nombre,
        parent_id,
    )

    if (
        existente
        and existente.get("mimeType")
        == "application/vnd.google-apps.folder"
    ):

        return existente["id"]

    return (
        drive_service
        .files()
        .create(
            body={
                "name": nombre,
                "mimeType":
                    "application/vnd.google-apps.folder",
                "parents": [parent_id],
            },
            fields="id",
            supportsAllDrives=True,
        )
        .execute()["id"]
    )


# ============================================================
# DRIVE — INICIALIZAR
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

    if (
        os.getenv("RENDER")
        != "true"
    ):
        return

    if not GOOGLE_AVAILABLE:
        return

    if not all(
        [
            GOOGLE_SERVICE_ACCOUNT_EMAIL,
            GOOGLE_PRIVATE_KEY,
            GOOGLE_PROJECT_ID,
            GOOGLE_DRIVE_FOLDER_ID,
        ]
    ):
        return

    try:

        pk = (
            GOOGLE_PRIVATE_KEY
            .replace("\\n", "\n")
        )

        info = {
            "type": "service_account",
            "project_id":
                GOOGLE_PROJECT_ID,
            "private_key_id":
                os.getenv(
                    "GOOGLE_PRIVATE_KEY_ID",
                    "",
                ),
            "private_key": pk,
            "client_email":
                GOOGLE_SERVICE_ACCOUNT_EMAIL,
            "client_id":
                os.getenv(
                    "GOOGLE_CLIENT_ID",
                    "",
                ),
            "auth_uri":
                "https://accounts.google.com/o/oauth2/auth",
            "token_uri":
                "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url":
                "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url":
                os.getenv(
                    "GOOGLE_CLIENT_X509_CERT_URL",
                    "",
                ),
        }

        cred = (
            service_account
            .Credentials
            .from_service_account_info(
                info,
                scopes=DRIVE_SCOPES,
            )
        )

        drive_service = build(
            "drive",
            "v3",
            credentials=cred,
            cache_discovery=False,
        )

        drive_session = AuthorizedSession(
            cred
        )

        DRIVE_ROOT_FOLDER = (
            GOOGLE_DRIVE_FOLDER_ID
        )

        raiz = (
            drive_service
            .files()
            .get(
                fileId=DRIVE_ROOT_FOLDER,
                fields="id, driveId",
                supportsAllDrives=True,
            )
            .execute()
        )

        DRIVE_SHARED_ID = raiz.get(
            "driveId"
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

        sincronizar_conocimiento_desde_drive()

    except Exception as error:

        print(
            "⚠️ Error inicializando Google Drive: "
            f"{error}"
        )

        drive_service = None


# ============================================================
# DRIVE — SUBIR
# ============================================================

def subir_archivo_drive(
    ruta: Path,
    folder_id: str,
    nombre: str | None = None,
):

    if (
        not drive_service
        or not ruta.exists()
    ):
        return None

    nombre_drive = (
        nombre
        or ruta.name
    )

    existente = buscar_archivo_drive(
        nombre_drive,
        folder_id,
        solo_json=(
            nombre_drive.lower()
            == "penaguillo.json"
        ),
    )

    mime = (
        "application/json"
        if ruta.suffix.lower() == ".json"
        else "application/octet-stream"
    )

    try:

        media = MediaFileUpload(
            str(ruta),
            mimetype=mime,
            resumable=True,
        )

        if existente:

            return (
                drive_service
                .files()
                .update(
                    fileId=existente["id"],
                    media_body=media,
                    supportsAllDrives=True,
                )
                .execute()["id"]
            )

        return (
            drive_service
            .files()
            .create(
                body={
                    "name": nombre_drive,
                    "mimeType": mime,
                    "parents": [folder_id],
                },
                media_body=media,
                supportsAllDrives=True,
            )
            .execute()["id"]
        )

    except Exception as error:

        print(
            "⚠️ Error subiendo a Drive: "
            f"{error}"
        )

        return None


# ============================================================
# DRIVE — DESCARGAR
# ============================================================

def descargar_archivo_drive(
    file_id: str,
    destino: Path,
):

    if (
        not drive_service
        or not drive_session
    ):
        return False

    try:

        respuesta = drive_session.get(
            (
                "https://www.googleapis.com/"
                f"drive/v3/files/{file_id}"
            ),
            params={
                "alt": "media",
                "supportsAllDrives": "true",
            },
            timeout=120,
        )

        if (
            respuesta.status_code == 200
            and len(respuesta.content) > 0
        ):

            destino.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            tmp = (
                destino.parent
                / f".tmp_{generar_id()}"
            )

            with open(
                tmp,
                "wb",
            ) as archivo:

                archivo.write(
                    respuesta.content
                )

            os.replace(
                tmp,
                destino,
            )

            return True

    except Exception:
        pass

    return False


# ============================================================
# VALIDAR JSON
# ============================================================

def validar_json_conocimiento(
    ruta: Path,
) -> tuple[
    bool,
    list[dict[str, Any]],
]:

    if (
        not ruta.exists()
        or ruta.stat().st_size == 0
    ):
        return False, []

    try:

        with open(
            ruta,
            "r",
            encoding="utf-8",
        ) as archivo:

            data = json.load(
                archivo
            )

        if (
            isinstance(data, list)
            and all(
                isinstance(i, dict)
                for i in data
            )
        ):

            return True, data

    except Exception:
        pass

    return False, []


# ============================================================
# DRIVE — SINCRONIZAR DESDE DRIVE
# ============================================================

def sincronizar_conocimiento_desde_drive():

    if (
        not drive_service
        or not DRIVE_KNOWLEDGE_FOLDER
    ):
        return

    try:

        existente = buscar_archivo_drive(
            "penaguillo.json",
            DRIVE_KNOWLEDGE_FOLDER,
            solo_json=True,
        )

        if existente:

            tmp = (
                CONOCIMIENTO_DIR
                / "penaguillo_drive.tmp"
            )

            if descargar_archivo_drive(
                existente["id"],
                tmp,
            ):

                if validar_json_conocimiento(
                    tmp
                )[0]:

                    os.replace(
                        tmp,
                        ARCHIVO_CONOCIMIENTO,
                    )

                    print(
                        "✅ Conocimiento "
                        "sincronizado desde "
                        "Google Drive."
                    )

                    return

            if tmp.exists():
                tmp.unlink()

    except Exception:
        pass


# ============================================================
# DRIVE — SINCRONIZAR A DRIVE
# ============================================================

def sincronizar_conocimiento_a_drive():

    if (
        drive_service
        and DRIVE_KNOWLEDGE_FOLDER
        and ARCHIVO_CONOCIMIENTO.exists()
    ):

        if validar_json_conocimiento(
            ARCHIVO_CONOCIMIENTO
        )[0]:

            subir_archivo_drive(
                ARCHIVO_CONOCIMIENTO,
                DRIVE_KNOWLEDGE_FOLDER,
                "penaguillo.json",
            )


# ============================================================
# CARGAR CONOCIMIENTO
# ============================================================

def cargar_conocimiento() -> list[
    dict[str, Any]
]:

    if not ARCHIVO_CONOCIMIENTO.exists():
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

        if isinstance(data, list):
            return data

        return []

    except Exception as error:

        print(
            "⚠️ Error cargando conocimiento: "
            f"{error}"
        )

        return []


# ============================================================
# GUARDAR CONOCIMIENTO
# ============================================================

def guardar_conocimiento(
    conocimientos: list[
        dict[str, Any]
    ],
) -> None:

    tmp = (
        CONOCIMIENTO_DIR
        / f"penaguillo_{generar_id()}.tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            conocimientos,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        tmp,
        ARCHIVO_CONOCIMIENTO,
    )

    sincronizar_conocimiento_a_drive()


# ============================================================
# BACKUP
# ============================================================

def crear_backup() -> str | None:

    if not ARCHIVO_CONOCIMIENTO.exists():
        return None

    valido, data = (
        validar_json_conocimiento(
            ARCHIVO_CONOCIMIENTO
        )
    )

    if not valido:
        return None

    backup_path = (
        BACKUP_DIR
        / (
            "penaguillo_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            ".json"
        )
    )

    tmp = (
        BACKUP_DIR
        / f"backup_{generar_id()}.tmp"
    )

    try:

        with open(
            tmp,
            "w",
            encoding="utf-8",
        ) as archivo:

            json.dump(
                data,
                archivo,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            tmp,
            backup_path,
        )

        if (
            drive_service
            and DRIVE_BACKUPS_FOLDER
        ):

            subir_archivo_drive(
                backup_path,
                DRIVE_BACKUPS_FOLDER,
                backup_path.name,
            )

        return str(
            backup_path
        )

    except Exception:

        return None


# ============================================================
# DRIVE — PDF
# ============================================================

def sincronizar_pdf_a_drive(
    ruta_pdf: Path,
):

    if (
        drive_service
        and DRIVE_PDF_FOLDER
    ):

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

    if (
        drive_service
        and DRIVE_IMAGES_FOLDER
    ):

        subir_archivo_drive(
            ruta_imagen,
            DRIVE_IMAGES_FOLDER,
            ruta_imagen.name,
        )


# ============================================================
# SYSTEM PROMPT
# ============================================================

def cargar_system_prompt() -> str:

    if not PROMPT_FILE.exists():

        return (
            "Eres Penaguillo, "
            "el asistente virtual."
        )

    try:

        with open(
            PROMPT_FILE,
            "r",
            encoding="utf-8",
        ) as archivo:

            return archivo.read().strip()

    except Exception:

        return "Eres Penaguillo."


SYSTEM_PROMPT_BASE = (
    cargar_system_prompt()
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Penaguillo IA",
    version="7.5.0",
    description=(
        "Backend del asistente inteligente "
        "Penaguillo"
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    ".pdf"
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
# CONSTRUIR QUERY CON HISTORIAL
# ============================================================

def construir_query_conversacional(
    mensaje: str,
    history: list[ChatMessage],
) -> str:
    """
    Construye una consulta de retrieval conservando contexto cuando
    la pregunta actual es claramente un seguimiento.

    Importante:
    - la pregunta actual siempre queda al final;
    - se priorizan mensajes del usuario, no respuestas largas de Gemini;
    - no se arrastra todo el historial al buscador;
    - saludos y mensajes autónomos cortos no heredan contexto anterior.
    """

    mensaje_limpio = mensaje.strip()

    if not history:
        return mensaje_limpio

    mensaje_n = normalizar_texto(mensaje_limpio)

    saludos = {
        "hola",
        "holi",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "hey",
        "que tal",
        "como estas",
    }

    if mensaje_n in saludos:
        return mensaje_limpio

    palabras = mensaje_n.split()

    palabras_seguimiento = {
        "ellos",
        "ellas",
        "el",
        "ella",
        "ese",
        "esa",
        "esos",
        "esas",
        "sus",
        "su",
        "tambien",
        "y",
        "quienes",
        "cual",
        "cuales",
        "correo",
        "correos",
        "telefono",
        "celular",
        "celulares",
        "proyecto",
        "persona",
        "equipo",
    }

    es_corta = len(palabras) <= 8
    tiene_referencia = any(
        palabra in palabras_seguimiento
        for palabra in palabras
    )

    if not (es_corta or tiene_referencia):
        return mensaje_limpio

    # Solamente mensajes anteriores del usuario. Las respuestas de
    # Gemini pueden ser largas y contaminar mucho el retrieval.
    mensajes_usuario = []

    for msg in reversed(history):
        if (
            msg.role == "user"
            and msg.content
            and msg.content.strip()
        ):
            mensajes_usuario.append(
                msg.content.strip()
            )

        if len(mensajes_usuario) >= 2:
            break

    if not mensajes_usuario:
        return mensaje_limpio

    mensajes_usuario.reverse()

    return (
        " ".join(mensajes_usuario)
        + " "
        + mensaje_limpio
    ).strip()


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")

def chat(
    data: ChatRequest,
):

    mensaje = (
        data.message.strip()
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

        tiempo_inicio_chat = (
            time.time()
        )

        conocimientos = (
            cargar_conocimiento()
        )

        # ----------------------------------------------------
        # QUERY PARA RAG
        # ----------------------------------------------------

        query_vectorial = (
            construir_query_conversacional(
                mensaje,
                data.history,
            )
        )

        print(
            "\n🧠 PREGUNTA ACTUAL:"
            f" {mensaje}"
        )

        print(
            "🧠 QUERY RAG:"
            f" {query_vectorial}"
        )

        # ----------------------------------------------------
        # RECUPERAR CONTEXTO
        # ----------------------------------------------------

        contexto_relevante = (
            construir_contexto_relevante(
                query_vectorial,
                conocimientos,
            )
        )

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        system_prompt = (
            SYSTEM_PROMPT_BASE

            + "\n\n"
            + "==============================\n"
            + "BASE DE CONOCIMIENTO RELEVANTE\n"
            + "==============================\n"

            + "\n"
            + "La información que aparece "
            + "a continuación proviene "
            + "de la base de conocimiento "
            + "de Penaguillo.\n"

            + "Utiliza esta información "
            + "como fuente principal y "
            + "única para los datos "
            + "específicos de Penagos.\n\n"

            + contexto_relevante

            + "\n\n"
            + "==============================\n"
            + "REGLAS ESTRICTAS\n"
            + "==============================\n"

            + "\n"
            + "1. Si preguntan quiénes "
            + "conforman un equipo, grupo "
            + "o área, debes mencionar "
            + "TODAS las personas que "
            + "aparezcan en el registro "
            + "correspondiente.\n"

            + "\n"
            + "2. NO inventes personas, "
            + "cargos, departamentos, "
            + "responsables ni funciones.\n"

            + "\n"
            + "3. Pertenecer a un equipo "
            + "NO significa automáticamente "
            + "ser responsable de un sistema "
            + "o proceso.\n"

            + "\n"
            + "4. Si preguntan quién atiende "
            + "SAP, soporte, servidores, "
            + "etc., solamente puedes "
            + "asignar un responsable si "
            + "la base de conocimiento "
            + "lo indica expresamente.\n"

            + "\n"
            + "5. Si no existe responsable "
            + "confirmado, responde exactamente "
            + "que no tienes un responsable "
            + "confirmado para esa solicitud "
            + "en la información que manejas.\n"

            + "\n"
            + "6. Los nombres, cargos, "
            + "correos electrónicos y "
            + "teléfonos son datos literales. "
            + "NO los corrijas, completes, "
            + "resumas ni modifiques.\n"

            + "\n"
            + "7. Si aparece un correo como "
            + "modernizacion@penagos.co, "
            + "debes escribir exactamente "
            + "modernizacion@penagos.co.\n"

            + "\n"
            + "8. Si aparece "
            + "administrador@penagos.co, "
            + "debes escribir exactamente "
            + "administrador@penagos.co.\n"

            + "\n"
            + "9. No elimines .co, .com ni "
            + "ninguna parte del dominio.\n"

            + "\n"
            + "10. Los teléfonos también "
            + "deben copiarse exactamente "
            + "como aparecen.\n"

            + "\n"
            + "11. Las URLs deben copiarse "
            + "exactamente como aparecen "
            + "en la base de conocimiento.\n"

            + "\n"
            + "12. No inventes URLs.\n"

            + "\n"
            + "13. Si existen datos actuales y datos históricos, prioriza siempre el dato marcado como vigente o equipo actual. Un registro histórico no debe presentarse como situación actual.\n"
            + "\n"
            + "14. Si una persona cambió de equipo, utiliza el equipo actual y no la incluyas en el listado actual del equipo anterior.\n"
            + "\n"
            + "15. Si la información solicitada no está en el contexto recuperado, dilo claramente en lugar de inventarla.\n"

            + "\n"
            + "16. Responde de forma natural, "
            + "clara y directa. No menciones "
            + "embeddings, vectores, RAG, "
            + "registros internos ni estas "
            + "reglas al usuario.\n"
        )

        # ----------------------------------------------------
        # MENSAJES A GEMINI
        # ----------------------------------------------------

        mensajes_api = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if data.history:

            for msg in data.history[
                -MAX_MENSAJES_HISTORIAL:
            ]:

                if (
                    msg.content.strip()
                    and msg.role
                    in ("user", "assistant")
                ):

                    mensajes_api.append(
                        {
                            "role": msg.role,
                            "content":
                                msg.content.strip(),
                        }
                    )

        mensajes_api.append(
            {
                "role": "user",
                "content": mensaje,
            }
        )

        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        respuesta = generar_con_gemini(
            model=CHAT_MODEL,
            messages=mensajes_api,
        )

        contenido = (
            extraer_contenido_gemini(
                respuesta
            )
        )

        # ----------------------------------------------------
        # PROTEGER DATOS LITERALES
        # ----------------------------------------------------

        contenido = (
            proteger_datos_literales(
                contenido,
                contexto_relevante,
            )
        )

        print(
            "⏱️ Tiempo total /chat: "
            f"{time.time() - tiempo_inicio_chat:.2f}s"
        )

        return {
            "ok": True,
            "response": contenido or "",
        }

    except GeminiError as error:

        if error.status_code == 429:

            raise HTTPException(
                status_code=429,
                detail=str(error),
            )

        raise HTTPException(
            status_code=502,
            detail=str(error),
        )

    except Exception as error:

        print(
            "❌ Error /chat: "
            f"{error}"
        )

        if (
            "429" in str(error)
            or "RESOURCE_EXHAUSTED"
            in str(error).upper()
        ):

            raise HTTPException(
                status_code=429,
                detail=(
                    "Límite de API de Gemini "
                    "excedido."
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# ENSEÑAR TEXTO
# ============================================================

@app.post("/ensenar")
def ensenar(data: EnsenarRequest):
    """Crea conocimiento o actualiza relaciones de forma genérica."""
    texto = data.conocimiento.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El conocimiento no puede estar vacío.")

    try:
        conocimientos = cargar_conocimiento()

        cambio = aplicar_cambio_de_equipo(texto, conocimientos)
        if cambio:
            guardar_conocimiento(conocimientos)
            return {
                "ok": True,
                "mensaje": "Cambio de conocimiento aplicado correctamente.",
                "modo": "actualizacion_relacional",
                "cambio": cambio,
                "total": len(conocimientos),
            }

        palabras = texto.split()
        titulo_dinamico = " ".join(palabras[:8]) + ("..." if len(palabras) > 8 else "")
        nuevo = {
            "id": generar_id(),
            "tipo": "texto",
            "titulo": titulo_dinamico,
            "contenido": texto,
            "descripcion": f"Información importante sobre: {titulo_dinamico}",
            "estado": "vigente",
            "embedding": [],
            "embedding_hash": "",
            "fecha": ahora_iso(),
        }
        nuevo["embedding"] = obtener_embedding(texto_para_embedding(nuevo))
        nuevo["embedding_hash"] = hash_conocimiento(nuevo)
        conocimientos.append(nuevo)
        guardar_conocimiento(conocimientos)

        return {
            "ok": True,
            "mensaje": "Conocimiento guardado.",
            "modo": "nuevo_conocimiento",
            "conocimiento": nuevo,
            "total": len(conocimientos),
        }
    except Exception as error:
        print(f"❌ Error /ensenar: {error}")
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================
# ENSEÑAR IMAGEN
# ============================================================
# ESTA PARTE SE MANTIENE FUNCIONAL
# ============================================================

@app.post("/ensenar-imagen")
async def ensenar_imagen(
    file: UploadFile = File(...),
):

    nombre_original = (
        file.filename
        or "imagen"
    )

    if (
        Path(
            nombre_original
        ).suffix.lower()
        not in EXTENSIONES_IMAGEN
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Formato de imagen "
                "no permitido."
            ),
        )

    contenido = await file.read()

    if (
        len(contenido)
        > MAX_IMAGE_SIZE
    ):

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

        # ----------------------------------------------------
        # VISION GEMINI
        # ----------------------------------------------------

        imagen_base64 = (
            base64.b64encode(
                contenido
            ).decode("utf-8")
        )

        extension = (
            ruta.suffix.lower()
        )

        if extension == ".png":

            mime = "image/png"

        elif extension == ".webp":

            mime = "image/webp"

        else:

            mime = "image/jpeg"

        prompt = (
            "Analiza cuidadosamente "
            "esta imagen para alimentar "
            "la base de conocimiento "
            "de Penaguillo. "
            "Extrae equipos, textos "
            "visibles, diagramas y "
            "procesos de forma "
            "estructurada. "
            "No inventes información."
        )

        resp = generar_con_gemini(
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
                                    (
                                        f"data:{mime};"
                                        f"base64,"
                                        f"{imagen_base64}"
                                    )
                            },
                        },

                    ],
                }
            ],
        )

        descripcion = (
            extraer_contenido_gemini(
                resp
            )
        )

        # ----------------------------------------------------
        # CREAR REGISTRO
        # ----------------------------------------------------

        nuevo = {

            "id": generar_id(),

            "tipo": "imagen",

            "titulo":
                nombre_original,

            "contenido":
                descripcion,

            "descripcion":
                descripcion,

            "archivo":
                f"/archivos/imagenes/{nombre}",

            "nombre_archivo":
                nombre_original,

            "embedding": [],

            "embedding_hash": "",

            "fecha":
                ahora_iso(),
        }

        nuevo["embedding"] = (
            obtener_embedding(
                texto_para_embedding(
                    nuevo
                )
            )
        )

        nuevo["embedding_hash"] = (
            hash_conocimiento(
                nuevo
            )
        )

        conocimientos = (
            cargar_conocimiento()
        )

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

            "mensaje":
                "Imagen aprendida "
                "correctamente.",

            "conocimiento":
                nuevo,

            "total":
                len(conocimientos),
        }

    except Exception as error:

        if ruta.exists():

            ruta.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# ENSEÑAR PDF
# ============================================================

def analizar_pagina_pdf_con_vision(
    pagina: fitz.Page,
    numero_pagina: int,
) -> str:

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY no configurada."
        )

    imagen_bytes = (
        pagina
        .get_pixmap(
            matrix=fitz.Matrix(
                1.5,
                1.5,
            ),
            alpha=False,
        )
        .tobytes("png")
    )

    imagen_base64 = (
        base64.b64encode(
            imagen_bytes
        ).decode("utf-8")
    )

    prompt = (
        f"Analiza esta página "
        f"{numero_pagina} de un PDF "
        "para la base de conocimiento "
        "de Penaguillo. "
        "Extrae todo texto legible, "
        "tablas, procesos y datos "
        "importantes en texto "
        "estructurado. "
        "No inventes información."
    )

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
                                (
                                    "data:image/png;"
                                    "base64,"
                                    f"{imagen_base64}"
                                )
                        },
                    },

                ],
            }
        ],
    )

    return extraer_contenido_gemini(
        respuesta
    )


@app.post("/ensenar-pdf")
async def ensenar_pdf(
    file: UploadFile = File(...),
):

    nombre_original = (
        file.filename
        or "documento.pdf"
    )

    if (
        Path(
            nombre_original
        ).suffix.lower()
        not in EXTENSIONES_PDF
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo debe ser "
                "un PDF."
            ),
        )

    contenido = await file.read()

    if (
        len(contenido)
        > MAX_PDF_SIZE
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "El PDF supera "
                "los 50 MB."
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

        textos = []

        for pagina in documento:

            texto_pagina = (
                pagina
                .get_text(
                    "text",
                    sort=True,
                )
                .strip()
            )

            if texto_pagina:

                textos.append(
                    texto_pagina
                )

        texto_extraido = (
            "\n\n".join(textos)
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

        if not es_escaneado:

            contenido_final = (
                texto_extraido
            )

        else:

            paginas_vision = []

            for (
                indice,
                pagina,
            ) in enumerate(
                documento,
                start=1,
            ):

                texto_pagina = (
                    analizar_pagina_pdf_con_vision(
                        pagina,
                        indice,
                    )
                )

                if texto_pagina.strip():

                    paginas_vision.append(
                        (
                            f"PÁGINA {indice}\n"
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

        # ----------------------------------------------------
        # CREAR REGISTRO
        # ----------------------------------------------------

        nuevo = {

            "id": generar_id(),

            "tipo": "pdf",

            "titulo":
                nombre_original,

            "contenido":
                contenido_final,

            "descripcion":
                (
                    "Documento PDF "
                    "procesado por "
                    "Penaguillo."
                ),

            "archivo":
                f"/archivos/pdf/{nombre}",

            "nombre_archivo":
                nombre_original,

            "numero_paginas":
                numero_paginas,

            "metodo":
                (
                    "vision"
                    if es_escaneado
                    else "texto"
                ),

            "embedding": [],

            "embedding_hash": "",

            "fecha":
                ahora_iso(),
        }

        nuevo["embedding"] = (
            obtener_embedding(
                texto_para_embedding(
                    nuevo
                )
            )
        )

        nuevo["embedding_hash"] = (
            hash_conocimiento(
                nuevo
            )
        )

        conocimientos = (
            cargar_conocimiento()
        )

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

            "mensaje":
                "PDF aprendido "
                "correctamente.",

            "conocimiento":
                nuevo,

            "total":
                len(conocimientos),
        }

    except Exception as error:

        if documento:

            documento.close()

        if ruta.exists():

            ruta.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(error),
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

            "total":
                len(conocimientos),

            "conocimientos":
                conocimientos,
        }

    except Exception as error:

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

        encontrado = next(
            (
                item
                for item in conocimientos
                if str(
                    item.get("id")
                )
                == str(data.id)
            ),
            None,
        )

        if not encontrado:

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
                item.get("id")
            )
            != str(data.id)
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

            ruta_archivo = (
                ARCHIVOS_DIR
                / archivo_relativo
                .replace(
                    "/archivos/",
                    "",
                    1,
                )
                .lstrip("/")
            )

            if ruta_archivo.exists():

                ruta_archivo.unlink()

        return {

            "ok": True,

            "mensaje":
                "Eliminado correctamente.",

            "total":
                len(
                    nuevos_conocimientos
                ),
        }

    except HTTPException:

        raise

    except Exception as error:

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

        "total":
            len(archivos),

        "backups": [

            {
                "nombre":
                    archivo.name,

                "fecha":
                    datetime.fromtimestamp(
                        archivo.stat().st_mtime
                    ).isoformat(),
            }

            for archivo in archivos
        ],
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "ok": True,

        "app":
            "Penaguillo IA",

        "version":
            "7.5.0",

        "engine":
            "Búsqueda textual + relaciones vigentes",

        "embeddings":
            EMBEDDINGS_HABILITADOS,

        "chat_model":
            CHAT_MODEL,

        "conocimientos":
            len(
                cargar_conocimiento()
            ),

        "retrieval_top_k":
            RELEVANCIA_TOP_K,
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        "🚀 Iniciando "
        "Penaguillo IA v7.3..."
    )

    inicializar_google_drive()

    try:

        conocimientos = (
            cargar_conocimiento()
        )

        if reconciliar_relaciones_vigentes(conocimientos):
            guardar_conocimiento(conocimientos)
            print("🔄 Relaciones vigentes reconciliadas automáticamente.")

        print(
            "📚 Conocimientos disponibles: "
            f"{len(conocimientos)}"
        )

    except Exception as error:

        print(
            "⚠️ Error cargando JSON: "
            f"{error}"
        )

    print(
        "✅ Penaguillo IA iniciado "
        "y listo."
    )


# ============================================================
# EJECUCIÓN LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )