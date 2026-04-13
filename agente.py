"""
Digitaliza Bot Base
Recepcionista virtual para WhatsApp vía YCloud.
Cerebro: Google Gemini 2.0 Flash.
Transcripción de audio: Groq Whisper large-v3.
"""

import os
import io
import json
import time
import traceback
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

import requests
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from groq import Groq

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODELO = os.environ.get("GROQ_MODELO", "whisper-large-v3")
YCLOUD_API_KEY = os.environ.get("YCLOUD_API_KEY", "")
YCLOUD_VERIFY_TOKEN = os.environ.get("YCLOUD_WEBHOOK_VERIFY_TOKEN", "digitaliza2026")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PORT = int(os.environ.get("PORT", "5000"))

CONVERSACIONES_DIR = DATA_DIR / "conversaciones"
CITAS_DIR = DATA_DIR / "citas"
MEDIA_DIR = DATA_DIR / "media"
for d in (CONVERSACIONES_DIR, CITAS_DIR, MEDIA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_HISTORIAL = 50
CONTEXTO_DEFAULT = 5
CONTEXTO_EXTENDIDO = 20
MAX_CHARS_MENSAJE = 1500
SENAL_MAS_CONTEXTO = "[NECESITO_MAS_CONTEXTO]"

YCLOUD_SEND_URL = "https://api.ycloud.com/v2/whatsapp/messages"
YCLOUD_MEDIA_URL = "https://api.ycloud.com/v2/whatsapp/media"

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("digitaliza")

# ─────────────────────────────────────────────────────────────
# Clientes de IA
# ─────────────────────────────────────────────────────────────

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """Eres el asistente virtual de {nombre_negocio}, {tipo_negocio} en {direccion}.

IDENTIDAD:
- No tienes nombre propio: eres "el asistente de Digitaliza".
- Si te preguntan si eres humano, aclara con naturalidad que eres el asistente virtual
  de Digitaliza, pero que un asesor humano puede continuar la conversación cuando lo
  necesite.

PÚBLICO:
- Estás hablando con DUEÑOS DE NEGOCIOS LOCALES (salones, barberías, consultorios
  médicos y dentales, veterinarias, restaurantes, spas, etc.) que están evaluando
  automatizar su atención al cliente con IA.

ROL: VENDEDOR CONSULTIVO, NO FOLLETO
- Primero ENTIENDE. Haz 1 o 2 preguntas cortas antes de recomendar:
  "¿Qué tipo de negocio tienes?", "¿Cuál es tu dolor más grande hoy con los clientes?",
  "¿Cuántos mensajes de WhatsApp atiendes al día aproximado?".
- Luego RECOMIENDA el servicio del catálogo que mejor resuelva eso.
- Habla del VALOR antes que del precio: cuántas horas libera, cuántos leads no se
  pierden, cuánto vale no tener que contestar a las 10 de la noche.
- Si te preguntan precios directo, dalos SIN RODEOS pero explica brevemente el valor.

TONO:
- Profesional pero cercano. TUTEA al prospecto (tú, te, contigo). NO usar "usted".
- Español mexicano natural. Yucateco de trato si fluye, sin forzar.
- Mensajes cortos, 1-3 líneas por mensaje de WhatsApp. Evita párrafos largos.
- Máximo 1 emoji por mensaje, solo si suma. Nada de spam de emojis.

INFORMACIÓN DE DIGITALIZA:
- Nombre: {nombre_negocio}
- Ubicación: {direccion}
- Contacto: {telefono}
- Horario asesores humanos: {horario}
- Web: digitaliza.mx

CATÁLOGO OFICIAL (única fuente de verdad para servicios y precios):
{servicios}

REGLAS ESTRICTAS:
1. NUNCA inventes servicios, features o precios que no estén en el catálogo.
2. Si no sabes algo técnico o específico, di: "Déjame consultarlo con el equipo y te
   respondo pronto." NO adivines.
3. Si el prospecto quiere contratar, saber más, o agendar una llamada, pide estos
   datos y GUÁRDALOS en tu respuesta para que quede registro:
     a) Nombre del prospecto
     b) Nombre de su negocio
     c) Tipo de negocio (salón, consultorio, etc.)
     d) Ciudad (para saber si es Mérida)
   Después dile: "Perfecto, un asesor te contacta pronto en horario laboral."
4. Si te mandan una FOTO (menú, local, tarjeta, pantalla actual, etc.) analízala y
   sugiere concretamente cómo Digitaliza la digitalizaría o mejoraría. Sé específico.
5. Si te preguntan "¿ya trabajan con [mi competencia]?" o cosas parecidas, no
   confirmes ni niegues clientes específicos; di que por confidencialidad no
   compartes nombres pero que trabajan con varios negocios del giro.
6. Rangos de precio: si dudan por precio, pregunta el tamaño del negocio para
   ubicar en qué rango cae, en vez de dar el precio más alto.
7. Horario: el BOT responde 24/7, pero los asesores humanos solo en horario laboral.
   Si algo requiere humano fuera de ese horario, agenda el contacto.

MEMORIA Y CONTEXTO:
- Recibes los últimos mensajes. Si el prospecto hace referencia a algo anterior que
  NO ves en el historial (ej. "como te dije ayer", "el precio que me pasaste"),
  responde EXACTAMENTE con la señal interna: {senal}
- Esa señal NO se muestra al cliente, es solo interna. No agregues nada más cuando
  la uses.

Tu objetivo final: calificar al prospecto, generar confianza y conseguir que acepte
una llamada con un asesor humano de Digitaliza."""

# ─────────────────────────────────────────────────────────────
# Persistencia de conversaciones
# ─────────────────────────────────────────────────────────────

_file_locks: dict[str, Lock] = {}
_global_lock = Lock()


def _lock_for(phone: str) -> Lock:
    with _global_lock:
        if phone not in _file_locks:
            _file_locks[phone] = Lock()
        return _file_locks[phone]


def _conv_path(phone: str) -> Path:
    safe = "".join(c for c in phone if c.isalnum() or c in "+-_")
    return CONVERSACIONES_DIR / f"{safe}.json"


def cargar_historial(phone: str) -> list[dict]:
    path = _conv_path(phone)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Error leyendo historial %s", phone)
        return []


def guardar_mensaje(phone: str, role: str, content: str) -> None:
    with _lock_for(phone):
        historial = cargar_historial(phone)
        historial.append({
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat() + "Z",
        })
        historial = historial[-MAX_HISTORIAL:]
        _conv_path(phone).write_text(
            json.dumps(historial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ─────────────────────────────────────────────────────────────
# Carga de datos del negocio
# ─────────────────────────────────────────────────────────────

def _leer_archivo(nombre: str) -> str:
    p = Path(__file__).parent / nombre
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


def _parse_negocio(texto: str) -> dict[str, str]:
    campos = {"nombre": "", "tipo": "", "direccion": "", "telefono": "", "horario": ""}
    mapeo = {
        "NOMBRE": "nombre",
        "TIPO": "tipo",
        "DIRECCIÓN": "direccion",
        "DIRECCION": "direccion",
        "TELÉFONO": "telefono",
        "TELEFONO": "telefono",
        "HORARIO": "horario",
    }
    for linea in texto.splitlines():
        if ":" not in linea:
            continue
        clave, valor = linea.split(":", 1)
        k = clave.strip().upper()
        if k in mapeo:
            campos[mapeo[k]] = valor.strip()
    return campos


def build_system_prompt() -> str:
    negocio = _parse_negocio(_leer_archivo("negocio.txt"))
    servicios = _leer_archivo("catalogo.txt") or "(Catálogo vacío)"
    return SYSTEM_PROMPT_TEMPLATE.format(
        nombre_negocio=negocio.get("nombre") or "el negocio",
        tipo_negocio=negocio.get("tipo") or "negocio",
        direccion=negocio.get("direccion") or "(no especificada)",
        telefono=negocio.get("telefono") or "(no especificado)",
        horario=negocio.get("horario") or "(no especificado)",
        servicios=servicios,
        senal=SENAL_MAS_CONTEXTO,
    )


# ─────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────

def _build_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=build_system_prompt(),
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG,
    )


def _historial_a_gemini(historial: list[dict]) -> list[dict]:
    salida = []
    for m in historial:
        rol = "user" if m["role"] == "user" else "model"
        salida.append({"role": rol, "parts": [m["content"]]})
    return salida


def preguntar_gemini(phone: str, entrada_usuario, n_contexto: int = CONTEXTO_DEFAULT) -> str:
    """entrada_usuario puede ser str o lista [texto, PIL.Image]."""
    modelo = _build_model()
    historial = cargar_historial(phone)[-n_contexto:]
    chat = modelo.start_chat(history=_historial_a_gemini(historial))
    resp = chat.send_message(entrada_usuario)
    texto = (resp.text or "").strip()

    if SENAL_MAS_CONTEXTO in texto and n_contexto < CONTEXTO_EXTENDIDO:
        log.info("[%s] Gemini pidió más contexto. Reintentando con %d mensajes.",
                 phone, CONTEXTO_EXTENDIDO)
        modelo = _build_model()
        historial_ext = cargar_historial(phone)[-CONTEXTO_EXTENDIDO:]
        chat = modelo.start_chat(history=_historial_a_gemini(historial_ext))
        aviso = entrada_usuario
        if isinstance(entrada_usuario, list):
            aviso = ["[SISTEMA: aquí tienes más contexto, responde al usuario]"] + entrada_usuario
        else:
            aviso = f"[SISTEMA: aquí tienes más contexto, responde al usuario]\n\n{entrada_usuario}"
        resp = chat.send_message(aviso)
        texto = (resp.text or "").strip()

    return texto or "Disculpe, tuve un problema para responder. ¿Podría repetir su mensaje?"


# ─────────────────────────────────────────────────────────────
# YCloud: descargar media y enviar mensajes
# ─────────────────────────────────────────────────────────────

def ycloud_descargar_media(media_id: str) -> bytes | None:
    """YCloud expone el binario en /v2/whatsapp/media/{id}."""
    if not media_id:
        return None
    url = f"{YCLOUD_MEDIA_URL}/{media_id}"
    try:
        r = requests.get(url, headers={"X-API-Key": YCLOUD_API_KEY}, timeout=30)
        if r.status_code == 200 and r.content:
            return r.content
        # Fallback: algunas cuentas devuelven JSON con una URL firmada
        try:
            data = r.json()
            download_url = data.get("url") or data.get("downloadUrl")
            if download_url:
                r2 = requests.get(download_url, timeout=30)
                if r2.status_code == 200:
                    return r2.content
        except Exception:
            pass
        log.error("Fallo descargando media %s: %s %s", media_id, r.status_code, r.text[:200])
        return None
    except Exception:
        log.exception("Excepción descargando media %s", media_id)
        return None


def ycloud_enviar_texto(from_number: str, to_number: str, texto: str) -> None:
    partes = _trocear(texto, MAX_CHARS_MENSAJE)
    for i, parte in enumerate(partes):
        payload = {
            "from": from_number,
            "to": to_number,
            "type": "text",
            "text": {"body": parte},
        }
        try:
            r = requests.post(
                YCLOUD_SEND_URL,
                headers={
                    "X-API-Key": YCLOUD_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=20,
            )
            log.info("[OUT %d/%d -> %s] %s | %s",
                     i + 1, len(partes), to_number, r.status_code, parte[:80])
            if r.status_code >= 400:
                log.error("YCloud error: %s", r.text[:500])
        except Exception:
            log.exception("Error enviando mensaje a %s", to_number)
        time.sleep(0.4)  # pequeño respiro entre partes


def _trocear(texto: str, limite: int) -> list[str]:
    if len(texto) <= limite:
        return [texto]
    partes, actual = [], ""
    for parrafo in texto.split("\n"):
        if len(actual) + len(parrafo) + 1 > limite:
            if actual:
                partes.append(actual.strip())
            if len(parrafo) > limite:
                for i in range(0, len(parrafo), limite):
                    partes.append(parrafo[i:i + limite])
                actual = ""
            else:
                actual = parrafo + "\n"
        else:
            actual += parrafo + "\n"
    if actual.strip():
        partes.append(actual.strip())
    return partes


# ─────────────────────────────────────────────────────────────
# Transcripción de audio (Groq Whisper)
# ─────────────────────────────────────────────────────────────

def transcribir_audio(audio_bytes: bytes) -> str:
    if not groq_client:
        log.error("GROQ_API_KEY no configurada")
        return ""
    try:
        # Groq acepta tupla (nombre, bytes) en el parámetro file
        resp = groq_client.audio.transcriptions.create(
            file=("audio.ogg", audio_bytes),
            model=GROQ_MODELO,
            language="es",
            response_format="text",
        )
        return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()
    except Exception:
        log.exception("Error transcribiendo audio")
        return ""


# ─────────────────────────────────────────────────────────────
# Procesamiento de un mensaje entrante YCloud
# ─────────────────────────────────────────────────────────────

def procesar_mensaje_ycloud(msg: dict) -> None:
    """
    Estructura de YCloud (evento whatsapp.inbound_message.received):
    {
      "id": "...",
      "wabaId": "...",
      "from": "521999...",   # cliente
      "to":   "521999...",   # número del negocio
      "type": "text" | "audio" | "image" | "video" | ...,
      "text":  {"body": "hola"},
      "audio": {"id": "...", "mimeType": "audio/ogg"},
      "image": {"id": "...", "mimeType": "image/jpeg", "caption": "..."}
    }
    """
    try:
        from_number = msg.get("from", "")
        to_number = msg.get("to", "")
        tipo = msg.get("type", "")
        if not from_number or not to_number:
            log.warning("Mensaje sin from/to: %s", msg)
            return

        log.info("[IN  %s -> %s] type=%s", from_number, to_number, tipo)

        entrada_usuario = None
        texto_guardar = ""

        if tipo == "text":
            cuerpo = (msg.get("text") or {}).get("body", "").strip()
            if not cuerpo:
                return
            entrada_usuario = cuerpo
            texto_guardar = cuerpo

        elif tipo == "audio":
            media_id = (msg.get("audio") or {}).get("id", "")
            audio_bytes = ycloud_descargar_media(media_id)
            if not audio_bytes:
                ycloud_enviar_texto(to_number, from_number,
                                    "No pude escuchar el audio, ¿podría reenviarlo o escribirlo?")
                return
            transcripcion = transcribir_audio(audio_bytes)
            if not transcripcion:
                ycloud_enviar_texto(to_number, from_number,
                                    "No logré entender el audio. ¿Me lo podría escribir?")
                return
            log.info("[%s] Transcripción: %s", from_number, transcripcion[:120])
            entrada_usuario = f"[Audio transcrito]: {transcripcion}"
            texto_guardar = entrada_usuario

        elif tipo == "image":
            img_obj = msg.get("image") or {}
            media_id = img_obj.get("id", "")
            caption = (img_obj.get("caption") or "").strip()
            img_bytes = ycloud_descargar_media(media_id)
            if not img_bytes:
                ycloud_enviar_texto(to_number, from_number,
                                    "No pude abrir la imagen, ¿podría reenviarla?")
                return
            try:
                pil = Image.open(io.BytesIO(img_bytes))
            except Exception:
                log.exception("No se pudo abrir imagen")
                ycloud_enviar_texto(to_number, from_number,
                                    "La imagen parece dañada, ¿podría reenviarla?")
                return
            texto_acompanante = caption or "El cliente envió una imagen. Describe lo relevante y responde."
            entrada_usuario = [texto_acompanante, pil]
            texto_guardar = f"[Imagen] {caption}".strip()

        else:
            ycloud_enviar_texto(to_number, from_number,
                                "Por ahora solo puedo procesar texto, audio e imágenes. ¿Me lo puede escribir?")
            return

        guardar_mensaje(from_number, "user", texto_guardar)
        respuesta = preguntar_gemini(from_number, entrada_usuario)
        guardar_mensaje(from_number, "assistant", respuesta)
        ycloud_enviar_texto(to_number, from_number, respuesta)

    except Exception:
        log.error("Error procesando mensaje:\n%s", traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({"service": "digitaliza-bot-base", "status": "ok"})


@app.get("/healthz")
def health():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat() + "Z"})


@app.get("/webhook")
def webhook_verify():
    # YCloud permite configurar verify token en su panel; responde lo que envíen en ?challenge=
    token = request.args.get("verify_token") or request.args.get("hub.verify_token")
    challenge = request.args.get("challenge") or request.args.get("hub.challenge", "")
    if token and token != YCLOUD_VERIFY_TOKEN:
        log.warning("Verify token inválido: %s", token)
        return "forbidden", 403
    return challenge or "ok", 200


@app.post("/webhook")
def webhook_receive():
    try:
        data = request.get_json(silent=True) or {}
        log.info("[WEBHOOK] %s", json.dumps(data, ensure_ascii=False)[:500])

        # YCloud manda un array o un objeto. Normalizamos.
        eventos = data if isinstance(data, list) else [data]

        for ev in eventos:
            tipo_ev = ev.get("type", "")
            # Inbound message de WhatsApp
            if tipo_ev in ("whatsapp.inbound_message.received",
                           "whatsapp.message.received",
                           "whatsapp:inbound_message.received"):
                msg = ev.get("whatsappInboundMessage") or ev.get("message") or ev.get("data") or {}
                if msg:
                    procesar_mensaje_ycloud(msg)
            elif "from" in ev and "type" in ev:
                # por si YCloud manda el mensaje sin envoltorio
                procesar_mensaje_ycloud(ev)
            else:
                log.info("[WEBHOOK] Evento ignorado: %s", tipo_ev)

        return jsonify({"received": True}), 200
    except Exception:
        log.error("Webhook error:\n%s", traceback.format_exc())
        return jsonify({"received": True}), 200  # siempre 200 para que YCloud no reintente infinito


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
