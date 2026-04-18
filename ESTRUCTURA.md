# Estructura del Bot de Digitaliza

Documento para entender cómo funciona el bot sin necesidad de programar.
Escrito para Eduardo y su equipo.

---

## ¿Qué hace el bot?

Es un recepcionista virtual que atiende los WhatsApp de Digitaliza 24/7.
Responde preguntas de prospectos, cotiza servicios, agenda citas en el
calendario de Eduardo y avisa al dueño cuando alguien está listo para
contratar. Entiende texto, notas de voz e imágenes. Cuando Eduardo le
escribe desde su celular personal, cambia de "modo ventas" a "modo
asistente interno" para ayudarlo a gestionar sus leads.

---

## Flujo de un mensaje

1. Un prospecto manda un WhatsApp al número de Digitaliza.
2. WhatsApp Business (vía YCloud) reenvía el mensaje a nuestro servidor.
3. El servidor filtra: si el mensaje es para otro número del portfolio, lo
   ignora. Si el mismo número manda más de 20 mensajes en un minuto,
   también lo ignora (anti-spam).
4. Si es una nota de voz, la manda a Groq Whisper para transcribir.
   Si es imagen, se la pasa directo a Gemini para que la analice.
5. Se revisa si el texto tiene intentos de jailbreak. Si sí, responde
   bloqueado y lo registra.
6. El mensaje se guarda en el historial del prospecto.
7. Gemini lee los últimos 5 mensajes de la conversación y genera una
   respuesta. Si detecta que le falta contexto, vuelve a consultar con
   los últimos 20.
8. Si Gemini quiere ver disponibilidad de agenda, el bot consulta Google
   Calendar y le pasa los horarios libres para que los presente al
   cliente. Si Gemini decide agendar, el bot crea el evento.
9. El bot limpia la respuesta (quita tags internos) y la manda al
   prospecto.
10. Si en la respuesta hubo captura de datos, detección de intención de
    compra o agendamiento, se le avisa a Eduardo al instante.

---

## Estructura del `agente.py`

El archivo es un solo script dividido en bloques. De arriba a abajo:

- **Config.** Lee variables de entorno (API keys, números, carpetas).
- **Normalización de números.** Una sola función `normalizar_numero()` que
  arregla el histórico `521` → `52` de WhatsApp México y deja todo en
  solo dígitos.
- **Rate limiting y jailbreak.** Contador de mensajes por minuto y regex
  para detectar intentos de saltarse las instrucciones.
- **System prompt.** El texto largo que le dice a Gemini cómo comportarse.
  Se cachea en memoria al arrancar.
- **Persistencia.** Guardar y leer historiales, perfiles y leads en disco.
- **Gemini.** Función que arma la llamada al modelo con el historial.
- **YCloud.** Funciones para descargar audios/imágenes y mandar mensajes
  de vuelta al cliente.
- **Audio (Groq Whisper).** Transcripción de notas de voz.
- **Google Calendar.** Consultar disponibilidad y agendar citas.
- **Notificaciones al dueño.** Nuevo prospecto, lead calificado, cambio
  de lead, quiere contratar, seguimiento, cita agendada.
- **Scheduler.** Un hilo que cada hora revisa conversaciones pendientes.
- **Modo admin.** Toda la lógica que se activa cuando Eduardo le escribe.
- **`procesar_mensaje_ycloud`.** El traductor central: recibe el JSON de
  YCloud y decide qué hacer.
- **Flask app.** Los endpoints HTTP: `/`, `/healthz`, `/webhook`.

---

## Archivos importantes

- **`agente.py`** — Todo el código del bot.
- **`negocio.txt`** — Datos fijos del negocio (nombre, dirección,
  teléfono, horario). Si cambias algo aquí, redespliega para que tome
  efecto.
- **`catalogo.txt`** — Servicios y precios. Es la única fuente de verdad:
  el bot nunca inventa precios fuera de este archivo.
- **`requirements.txt`** — Lista de librerías de Python.
- **`Procfile`** — Le dice a Railway cómo arrancar el bot.
- **`oauth_setup.py`** — Script que corres una sola vez para conectar el
  calendario de Google.
- **`README.md`** — Guía técnica de setup.
- **`ESTRUCTURA.md`** — Este documento.

Dentro del volumen `/data` (Railway):

- **`conversaciones/`** — Un JSON por teléfono con los últimos 50
  mensajes. Sobre esto se construye el contexto.
- **`leads/`** — Un JSON por prospecto que ya dio sus 3 datos
  (nombre + negocio + ciudad).
- **`perfiles/`** — Resúmenes automáticos del cliente (qué negocio tiene,
  qué le interesa). Lo usa el modo admin para resumir rápido.
- **`seguimiento/`** — Marcas internas: a quién ya le avisé, cuándo.
- **`security_logs.json`** — Últimos 500 intentos de jailbreak.
- **`config.json`** — Estado de notificaciones (silenciadas / activas).

---

## Variables de entorno

- `GEMINI_API_KEY` — Llave de Google AI Studio para Gemini.
- `GEMINI_MODEL` — Qué modelo usar (por defecto `gemini-3-flash-preview`).
- `GROQ_API_KEY` — Llave de Groq para transcribir audios.
- `GROQ_MODELO` — Qué modelo de Whisper usar.
- `YCLOUD_API_KEY` — Llave para mandar y leer WhatsApp por YCloud.
- `YCLOUD_WEBHOOK_VERIFY_TOKEN` — Token que YCloud usa para verificar
  que el webhook es nuestro.
- `OWNER_PHONE` — El WhatsApp personal de Eduardo. Quien escriba desde
  aquí entra en modo admin.
- `BOT_PHONE` — El número oficial del bot. Solo procesa mensajes que
  llegan a este número (filtro multi-tenant de YCloud).
- `DATA_DIR` — Carpeta donde se guarda todo en Railway (`/data`).
- `PORT` — Puerto del servidor Flask.
- `GOOGLE_CLIENT_ID` — Credenciales OAuth de Google (agendamiento).
- `GOOGLE_CLIENT_SECRET` — Credenciales OAuth de Google.
- `GOOGLE_REFRESH_TOKEN` — Token que permite al bot acceder al calendario
  de Eduardo sin pedir login cada vez.
- `GOOGLE_CALENDAR_ID` — Qué calendario usar (por defecto `primary`).

---

## Modo admin: qué puede hacer Eduardo desde su celular

Cuando Eduardo manda un WhatsApp al bot desde su número personal, el bot
deja de vender y se vuelve su asistente interno. Acepta cosas como:

- **"resumen"** / **"leads"** / **"quién me ha escrito"** — lista todos
  los prospectos con su nombre, negocio, interés y último mensaje.
- **"info +52..."** — perfil completo de un número específico.
- **"escríbele a +52..., dile que ya le preparé la propuesta"** — el bot
  redacta y envía el mensaje en nombre de Eduardo, respetando la ventana
  de 24h de WhatsApp.
- **"bórralo"** / **"borra al +52..."** — elimina la conversación, el
  lead y el perfil cacheado de ese número.
- **"alertas de seguridad"** / **"intentos de jailbreak"** — resume los
  últimos ataques al bot.
- **"silenciar notificaciones"** (o `silenciar`, `mute`) — apaga avisos
  proactivos por 8 horas.
- **"activar notificaciones"** (o `activar`, `unmute`) — los vuelve a
  prender.

Los comandos que manda el asistente internamente (`CMD_ENVIAR`,
`CMD_BORRAR`, `CMD_VER`) se ejecutan y se limpian antes de que Eduardo
los vea.

---

## Notificaciones automáticas al dueño

El bot le escribe a Eduardo sin que él lo pida cuando:

- 🔔 **Nuevo prospecto** — alguien escribió al bot por primera vez.
- 🆕 **Nuevo lead** — el prospecto ya dio nombre + negocio + ciudad.
- 🔥 **Lead calificado** — el perfil automático tiene nombre y tipo de
  negocio (aunque no haya soltado los 3 datos formales).
- 📝 **Lead actualizado** — el prospecto corrigió su nombre o tipo de
  negocio después de ya haber sido notificado.
- 🚀 **Quiere contratar** — Gemini detectó intención clara de compra
  ("ya quiero contratar", "cómo le pago", etc.).
- ⏰ **Seguimiento pendiente** — un prospecto activo (3+ mensajes) lleva
  entre 6 y 48h sin respuesta. Se revisa cada hora.
- 📅 **Nueva cita agendada** — el bot creó un evento en el calendario.

Todo esto se puede apagar temporalmente con `silenciar`.

---

## Seguridad

- **Rate limiting**: máximo 20 mensajes por minuto por número. Quien se
  pase queda ignorado hasta que baje el ritmo.
- **Filtro multi-tenant**: solo atiende mensajes dirigidos al número
  oficial de Digitaliza. Ignora el resto del portfolio de YCloud.
- **Detección de jailbreak**: un regex bloquea frases tipo "ignora tus
  instrucciones", "muéstrame tu system prompt", "actúa como DAN". Se
  guarda log de intentos.
- **Protección del system prompt**: Gemini tiene regla explícita para
  no revelar sus instrucciones ni aunque se lo pidan indirectamente
  (traducir, resumir, parafrasear, hacer poema).
- **Confidencialidad entre clientes**: el bot nunca comparte datos ni
  nombres de otros clientes.
- **Filtros de contenido de Gemini**: bloquean contenido sexual,
  violento, discriminatorio.
- **Admin solo desde OWNER_PHONE**: nadie que no escriba desde el número
  de Eduardo puede activar modo admin ni ejecutar comandos internos.
- **Admin exento del rate limit**: Eduardo no puede bloquearse a sí
  mismo por mandar muchos comandos seguidos.
- **Locks por conversación**: evita que dos mensajes simultáneos del
  mismo cliente se pisen al escribirse en disco.
- **Webhook asíncrono**: el servidor responde OK de inmediato a YCloud y
  procesa el mensaje después, para que YCloud no reintente y duplique
  mensajes al cliente.
