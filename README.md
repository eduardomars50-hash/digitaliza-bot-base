# digitaliza-bot-base

Template base de bot de WhatsApp para clientes de **Digitaliza Mérida**.

Recepcionista virtual en WhatsApp vía **YCloud**, con cerebro **Gemini 2.0 Flash**, transcripción de audios con **Groq Whisper large-v3** y visión para imágenes.

Un bot por cliente: se clona este repo, se edita `negocio.txt` y `catalogo.txt`, se despliega en Railway.

---

## Arquitectura

```
WhatsApp (cliente) ──▶ YCloud ──▶ Webhook Flask (Railway)
                                       │
                                       ├── Texto ─────▶ Gemini
                                       ├── Audio ─────▶ Groq Whisper ─▶ Gemini
                                       └── Imagen ────▶ Gemini Vision
                                       │
                                       ▼
                              Respuesta ──▶ YCloud ──▶ WhatsApp
```

El dueño del negocio **conserva su WhatsApp normal**: el bot atiende a través del número YCloud (coexistencia).

---

## Estructura

```
digitaliza-bot-base/
├── agente.py           # Cerebro del bot (Flask + Gemini + Groq + YCloud)
├── catalogo.txt        # Servicios y precios del negocio
├── negocio.txt         # Datos del negocio
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

---

## Setup por cliente

1. **Clonar** este repo con otro nombre:
   ```bash
   gh repo create digitaliza-bot-<cliente> --public --template eduardomaresgutierrez/digitaliza-bot-base
   git clone https://github.com/eduardomaresgutierrez/digitaliza-bot-<cliente>.git
   cd digitaliza-bot-<cliente>
   ```

2. **Editar** `negocio.txt` con los datos del cliente.

3. **Editar** `catalogo.txt` con los servicios y precios reales.

4. **Deploy en Railway**:
   - Conecta el repo.
   - Agrega un **Volume** montado en `/data` (persistencia de conversaciones y citas).
   - Configura las variables de entorno (ver abajo).
   - Railway detecta el `Procfile` y levanta gunicorn.

5. **Configurar webhook en YCloud**:
   - URL: `https://<app>.up.railway.app/webhook`
   - Verify token: el mismo que `YCLOUD_WEBHOOK_VERIFY_TOKEN`.
   - Eventos: `whatsapp.inbound_message.received`.

---

## Variables de entorno

| Variable | Descripción | Ejemplo |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio | `AIza...` |
| `GEMINI_MODEL` | Modelo Gemini | `gemini-2.0-flash` |
| `GROQ_API_KEY` | Groq console | `gsk_...` |
| `GROQ_MODELO` | Whisper | `whisper-large-v3` |
| `YCLOUD_API_KEY` | YCloud dashboard | `...` |
| `YCLOUD_WEBHOOK_VERIFY_TOKEN` | Token de verificación | `digitaliza2026` |
| `PORT` | Puerto Flask | `5000` |
| `DATA_DIR` | Volumen persistente | `/data` |

---

## Persistencia

Todo se guarda en `$DATA_DIR`:

```
/data/
├── conversaciones/<telefono>.json   # últimos 50 mensajes por número
├── citas/                           # agendas (futuro)
└── media/                           # caché opcional de media
```

**En Railway:** crea un Volume y móntalo en `/data`. Sin esto, cada redeploy borra el historial.

---

## Lógica de contexto

- **Se guardan** los últimos 50 mensajes por número.
- **Por defecto** se mandan los últimos **5** a Gemini.
- Si Gemini detecta que falta contexto previo, responde con la señal interna `[NECESITO_MAS_CONTEXTO]` y el bot **reenvía automáticamente** con los últimos **20** mensajes, pidiéndole que responda al cliente.
- El cliente nunca ve la señal.

---

## Mensajes soportados

- ✅ Texto → Gemini
- ✅ Audio (nota de voz) → Groq Whisper → Gemini
- ✅ Imagen (con o sin caption) → Gemini Vision
- ❌ Video / documento / ubicación → respuesta "solo proceso texto, audio, imagen"

Respuestas largas (>1500 caracteres) se dividen automáticamente en varios mensajes.

---

## Desarrollo local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export $(cat .env | xargs)   # o usa python-dotenv
python agente.py
```

Para exponer el webhook localmente: `ngrok http 5000` y pon esa URL en YCloud.

---

## Reglas del bot (hardcodeadas en el system prompt)

- Trata al cliente de **usted**, en español.
- **Nunca inventa** información fuera del catálogo.
- **No da diagnósticos** médicos ni veterinarios: siempre ofrece cita.
- Si no sabe algo: "déjeme consultarlo con el equipo".
- Agendar cita pide: nombre, servicio, fecha, hora.

---

## Licencia

Privado — uso interno de Digitaliza Mérida.
