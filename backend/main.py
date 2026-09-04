# ============================================================
# PENAGUILLO IA — BACKEND FASTAPI
# ============================================================
# Funciones:
# - Chat con Penaguillo
# - Enseñar texto
# - Enseñar imágenes
# - Enseñar PDF
# - PDF con texto seleccionable -> PyMuPDF
# - PDF escaneado -> OpenRouter Vision
# - Imágenes -> OpenRouter Vision
# - Persistencia segura en penaguillo.json
# - Backup automático antes de modificar conocimiento
# - Escritura atómica para evitar corrupción del JSON
# - NO usa Tesseract
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

import fitz  # PyMuPDF

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

# ------------------------------------------------------------
# IMPORTANTE PARA PYINSTALLER
#
# En desarrollo:
#   BASE_DIR = carpeta donde está main.py
#
# En el .exe:
#   BASE_DIR = carpeta donde está penaguillo-backend.exe
#
# Esto permite que el ejecutable encuentre:
#
#   .env
#   conocimiento/
#   archivos/
#
# junto al ejecutable.
# ------------------------------------------------------------

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# DIRECTORIOS
# ============================================================

CONOCIMIENTO_DIR = BASE_DIR / "conocimiento"

ARCHIVOS_DIR = CONOCIMIENTO_DIR / "archivos"

PDF_DIR = ARCHIVOS_DIR / "pdf"

IMAGENES_DIR = ARCHIVOS_DIR / "imagenes"

BACKUP_DIR = CONOCIMIENTO_DIR / "backups"

ARCHIVO_CONOCIMIENTO = CONOCIMIENTO_DIR / "penaguillo.json"


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

ENV_FILE = BASE_DIR / ".env"

load_dotenv(
    ENV_FILE
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

if not OPENROUTER_API_KEY:

    print(
        "⚠️ ADVERTENCIA: "
        f"No se encontró OPENROUTER_API_KEY en: {ENV_FILE}"
    )


# ============================================================
# OPENROUTER
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CHAT_MODEL = "openai/gpt-oss-20b"

# Router gratuito de OpenRouter.
# Puede seleccionar modelos gratuitos compatibles con visión.
VISION_MODEL = "openrouter/free"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Penaguillo IA",
    version="3.0.0",
    description="Backend del asistente inteligente Penaguillo",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
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

MAX_PDF_SIZE = 50 * 1024 * 1024

MAX_IMAGE_SIZE = 15 * 1024 * 1024


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

    """
    Evita caracteres peligrosos
    en nombres de archivos.
    """

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
        or f"archivo_{generar_id()}"
    )


# ============================================================
# CARGAR CONOCIMIENTO
# ============================================================

def cargar_conocimiento() -> list[dict[str, Any]]:

    """
    Carga penaguillo.json SIN BORRAR NADA.

    IMPORTANTE:

    - Si el archivo no existe -> crea uno nuevo.
    - Si está corrupto -> lanza error.
    - NUNCA devuelve [] silenciosamente
      ante un JSON corrupto.
    """

    if not ARCHIVO_CONOCIMIENTO.exists():

        print(
            "ℹ️ penaguillo.json no existe. "
            "Se creará uno nuevo."
        )

        guardar_conocimiento(
            []
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
            "o tiene JSON inválido. "
            "No se modificará para evitar "
            "perder información."
        ) from error


    except OSError as error:

        raise RuntimeError(
            f"No fue posible leer "
            f"penaguillo.json: {error}"
        ) from error


    if not isinstance(
        data,
        list,
    ):

        raise RuntimeError(
            "penaguillo.json no tiene "
            "el formato esperado. "
            "Debe contener una lista "
            "de conocimientos. "
            "No se modificará para "
            "evitar pérdida de información."
        )


    return data


# ============================================================
# BACKUP
# ============================================================

def crear_backup() -> str | None:

    """
    Hace una copia del JSON existente
    ANTES de modificarlo.
    """

    if not ARCHIVO_CONOCIMIENTO.exists():

        return None


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
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
            f"🛡️ Backup creado: "
            f"{backup_path.name}"
        )


        return str(
            backup_path
        )


    except OSError as error:

        raise RuntimeError(
            "No se pudo crear backup "
            f"de penaguillo.json: {error}"
        ) from error


# ============================================================
# GUARDAR CONOCIMIENTO
# ============================================================

def guardar_conocimiento(
    conocimientos: list[dict[str, Any]],
) -> None:

    """
    Guarda conocimiento de manera segura.

    PASOS:

    1. Valida que sea lista.
    2. Hace backup.
    3. Escribe un archivo temporal.
    4. Fuerza escritura en disco.
    5. Reemplaza el JSON original.

    Esto evita que una interrupción
    deje el JSON a medias.
    """

    if not isinstance(
        conocimientos,
        list,
    ):

        raise RuntimeError(
            "No se puede guardar conocimiento "
            "porque la estructura no es una lista."
        )


    # --------------------------------------------------------
    # BACKUP
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


        # ----------------------------------------------------
        # REEMPLAZO ATÓMICO
        # ----------------------------------------------------

        os.replace(
            archivo_temporal,
            ARCHIVO_CONOCIMIENTO,
        )


        print(
            "💾 Conocimiento guardado. "
            f"Total registros: "
            f"{len(conocimientos)}"
        )


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
# CONSTRUIR CONOCIMIENTO PARA EL CHAT
# ============================================================

def construir_conocimiento() -> str:

    conocimientos = cargar_conocimiento()


    if not conocimientos:

        return (
            "Actualmente no hay conocimiento adicional "
            "registrado en la base de conocimientos."
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
ID: {item.get("id", "")}
TIPO: {tipo}
TÍTULO: {titulo}

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

SYSTEM_PROMPT_BASE = """
Eres Penaguillo, el asistente inteligente de Penagos Hermanos.

Tu función es ayudar a usuarios y colaboradores con información
sobre Penagos, sus productos, maquinaria, procesos de café,
agricultura, servicio técnico, postventa, proyectos y conocimiento
interno que haya sido proporcionado al sistema.

REGLAS IMPORTANTES:

1. No inventes información.

2. Si no encuentras la respuesta en el conocimiento disponible,
   dilo claramente.

3. Diferencia entre información confirmada y suposiciones.

4. No inventes precios, referencias, capacidades, especificaciones
   técnicas, contactos ni disponibilidad.

5. Si una pregunta requiere información que no existe en tu
   conocimiento, solicita los datos necesarios.

6. Responde de forma profesional pero natural.

7. Puedes explicar temas técnicos de manera sencilla.

8. Cuando corresponda, recomienda contactar al área comercial,
   técnica o de servicio.

9. Utiliza TODO el conocimiento proporcionado.

10. El conocimiento interno tiene prioridad sobre respuestas
    genéricas cuando responde directamente a la pregunta.

11. Nunca digas que sabes algo si no aparece en la información
    disponible o no puedes confirmarlo.

Eres un asistente de Penagos Hermanos, no un asistente genérico.
"""


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    conocimientos = cargar_conocimiento()


    return {

        "ok": True,

        "app": "Penaguillo IA",

        "version": "3.0.0",

        "chat_model": CHAT_MODEL,

        "vision_model": VISION_MODEL,

        "conocimientos": len(
            conocimientos
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
            detail="El mensaje no puede estar vacío.",
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
        + "BASE DE CONOCIMIENTO DE PENAGUILLO\n"
        + "==============================\n"
        + conocimiento
    )


    try:

        respuesta = client.chat.completions.create(

            model=CHAT_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt,
                },

                {
                    "role": "user",
                    "content": mensaje,
                },

            ],

            temperature=0.2,

        )


        contenido = (
            respuesta
            .choices[0]
            .message
            .content
        )


        return {

            "ok": True,

            "response": contenido or "",

        }


    except Exception as error:

        print(
            f"❌ Error en /chat: {error}"
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

    texto = data.conocimiento.strip()


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

            "titulo": "Conocimiento manual",

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
# IMAGEN -> BASE64
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

    extension = ruta.suffix.lower()


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
# VISIÓN OPENROUTER
# ============================================================

def analizar_imagen_con_vision(
    ruta: Path,
    contexto: str = "",
) -> str:

    if not OPENROUTER_API_KEY:

        raise RuntimeError(
            "No existe OPENROUTER_API_KEY "
            "en el archivo .env."
        )


    imagen_base64 = (
        imagen_a_base64(
            ruta
        )
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

        respuesta = client.chat.completions.create(

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
                                    f"data:{mime};base64,"
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


        contenido = (
            respuesta
            .choices[0]
            .message
            .content
        )


        return contenido or ""


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


    extension = Path(
        nombre_original
    ).suffix.lower()


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
                "el límite de 15 MB."
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


    imagen_bytes = pixmap.tobytes(
        "png"
    )


    return base64.b64encode(
        imagen_bytes
    ).decode(
        "utf-8"
    )


# ============================================================
# ANALIZAR PÁGINA PDF CON VISIÓN
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

        respuesta = client.chat.completions.create(

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
                                    "data:image/png;base64,"
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


    extension = Path(
        nombre_original
    ).suffix.lower()


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
                    "🔎 Analizando PDF — página "
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
# OBTENER TODO EL CONOCIMIENTO
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
# ELIMINAR UN CONOCIMIENTO
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
# BACKUPS DISPONIBLES
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
