"""
Detección y manejo de leads provenientes de email frío.

Cuando un prospecto contacta el bot por primera vez con un texto que
indica que vino del email (cold outreach de Eduardo), lo marcamos
con un flag persistente. Eso permite:

- Saludo personalizado (vía contexto inyectado al prompt de Gemini).
- Notificación inmediata a Eduardo (vía OWNER_PHONE).
- Tono y reglas especiales para este tipo de lead.

El flag vive en SEGUIMIENTO_DIR/{phone}_email_lead.flag — mismo
patrón que el resto del sistema de seguimiento.

Aislado del resto del bot para que sea fácil de mantener / quitar
sin tocar la lógica core de agente.py.
"""
import json
import re
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Detección
# ─────────────────────────────────────────────────────────────

# Patrones tolerantes a variantes comunes (correo/email/mail, acentos
# opcionales, género). Solo matchea cuando hay señal clara de email
# frío — NO matchea si el prospecto solo dice "vi tu página web".
EMAIL_LEAD_RE = re.compile(
    r"("
    # 1. "vengo del correo / email / mail"
    r"vengo\s+(del|de\s+(su|tu)\s+|desde\s+(el\s+)?)?"
    r"(correo|emails?|e-?mails?|mails?)"
    r"|"
    # 2. "vi su / tu / el correo / email / mail"
    # 3. "leí / lei su / tu / el correo / email / mail"
    r"(vi|le[íi]|recib[íi])\s+(su|tu|el|un)\s+"
    r"(correo|emails?|e-?mails?|mails?)"
    r"|"
    # 4. "interesad[oa] en sitios? web | páginas? web"
    r"interesad[oa]\s+en\s+(un\s+|una\s+)?"
    r"(sitios?\s+web|p[aá]ginas?\s+web)"
    r"|"
    # 5. (correo/email/mail) ... (sitio web / página web) — dentro de 60 chars
    r"(correo|email|e-?mail|mail).{0,60}"
    r"(sitios?\s+web|p[aá]ginas?\s+web)"
    r"|"
    # 6. Inverso: (sitio web / página web) ... (correo/email/mail)
    r"(sitios?\s+web|p[aá]ginas?\s+web).{0,60}"
    r"(correo|email|e-?mail|mail)"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def detectar_email_lead(texto: str) -> bool:
    """True si el texto matchea algún patrón de lead de email frío.

    Tolerante a:
    - correo / email / e-mail / mail (singular y plural)
    - acentos opcionales (lei/leí, pagina/página)
    - género (interesado / interesada)

    NO matchea solo "página web" o "sitio web" sin contexto de email.
    """
    if not texto:
        return False
    return bool(EMAIL_LEAD_RE.search(texto))


# ─────────────────────────────────────────────────────────────
# Estado persistente (flags)
# ─────────────────────────────────────────────────────────────

def _flag_path(seguimiento_dir: Path, phone_norm: str) -> Path:
    return seguimiento_dir / f"{phone_norm}_email_lead.flag"


def marcar_email_lead(
    seguimiento_dir: Path,
    phone_norm: str,
    primer_msg: str = "",
) -> bool:
    """Marca persistente al teléfono como email_lead.

    Devuelve True si lo marcó (primera vez), False si ya estaba marcado.
    El archivo guarda timestamp + el primer mensaje (truncado), útil
    para notify y para reconstruir contexto en sesiones futuras.
    """
    flag = _flag_path(seguimiento_dir, phone_norm)
    if flag.exists():
        return False
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "primer_mensaje": (primer_msg or "")[:500],
    }
    flag.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def es_email_lead(seguimiento_dir: Path, phone_norm: str) -> bool:
    """True si el lead fue marcado previamente como email_lead."""
    return _flag_path(seguimiento_dir, phone_norm).exists()


def datos_email_lead(
    seguimiento_dir: Path,
    phone_norm: str,
) -> dict | None:
    """Devuelve los datos guardados al marcar (ts + primer_mensaje),
    o None si nunca fue marcado o el archivo está corrupto."""
    flag = _flag_path(seguimiento_dir, phone_norm)
    if not flag.exists():
        return None
    try:
        return json.loads(flag.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Notificación a Eduardo
# ─────────────────────────────────────────────────────────────

def formatear_notificacion(
    phone_norm: str,
    primer_msg: str,
    nombre: str = "",
    timezone_name: str = "America/Merida",
) -> str:
    """Formatea el WhatsApp que se manda a OWNER_PHONE cuando llega
    un EMAIL_LEAD nuevo. Hora en zona Mérida (CDT/CST)."""
    try:
        from zoneinfo import ZoneInfo
        ahora = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        ahora = datetime.now()
    hora_str = ahora.strftime("%H:%M")
    fecha_str = ahora.strftime("%d-%m-%Y")
    nombre_display = nombre.strip() if nombre and nombre.strip() else "no especificado"
    msg_display = (primer_msg or "").strip()[:300]

    return (
        f"📧 Nuevo lead de EMAIL FRÍO\n"
        f"Número: {phone_norm}\n"
        f"Nombre: {nombre_display}\n"
        f"Mensaje: '{msg_display}'\n"
        f"Hora: {fecha_str} {hora_str}\n\n"
        f"El bot ya está atendiendo. Toma el chat cuando quieras."
    )


# ─────────────────────────────────────────────────────────────
# Contexto inyectado al system_instruction de Gemini
# ─────────────────────────────────────────────────────────────

# Bloque que se concatena al system prompt cuando el lead actual
# está marcado como email_lead. Le dice a Gemini cómo saludar y
# qué tono usar — sin reemplazar el resto del prompt.
_CONTEXTO_PROMPT = (
    "\n\n═══════════════════════════════════════════════\n"
    "FUENTE DEL LEAD: EMAIL FRÍO (cold outreach de Eduardo)\n"
    "═══════════════════════════════════════════════\n"
    "Este prospecto contactó porque Eduardo le mandó un email frío "
    "sobre PÁGINAS WEB. Es un lead escaso y valioso — trátalo con "
    "atención prioritaria.\n\n"
    "REGLAS ESPECIALES PARA ESTE LEAD:\n\n"
    "1) SALUDO INICIAL (tu primera respuesta a este lead, NO uses "
    "saludo genérico):\n"
    "   Reconoce que viste el correo y que su interés es páginas web. "
    "Algo así (adapta al tono natural, no copies literal):\n"
    '   "¡Hola! Qué gusto que escribieras. Soy el asistente de Eduardo '
    "en Digitaliza. Veo que viste el correo y te interesa platicar de "
    "sitios web — perfecto, justo para eso estoy. ¿Qué te gustaría "
    "saber primero? Tenemos dos paquetes principales (Nivel 1 a $3,500 "
    "y Nivel 2 a $6,500) y puedo contarte de cualquiera, o platicar "
    'sobre el proceso y tiempos."\n\n'
    "2) ENFOQUE: el interés CONFIRMADO es PÁGINAS WEB. No empieces "
    "ofreciendo chatbots a menos que el prospecto pregunte por ellos.\n\n"
    "3) PRECIOS: usa SOLO los del catálogo oficial — Nivel 1 ($3,500 "
    "lanzamiento / $4,200 normal), Nivel 2 ($6,500 lanzamiento / "
    "$8,200 normal), Renovación ($1,500/año), proyectos custom se "
    "escalan a Eduardo con [EVENTO:QUIERE_WEB]. NUNCA inventes precios "
    "ni rondas de revisión.\n\n"
    "4) TONO: cálido, profesional, directo. El prospecto ya mostró "
    "interés concreto — no necesitas convencerlo de la idea, ayúdalo "
    "a elegir el paquete y avanzar al siguiente paso (cita o cierre).\n"
)


def contexto_para_prompt(seguimiento_dir: Path, phone_norm: str) -> str:
    """Bloque para concatenar al system_instruction.
    Devuelve string vacío si el lead no es email_lead."""
    if not es_email_lead(seguimiento_dir, phone_norm):
        return ""
    return _CONTEXTO_PROMPT
