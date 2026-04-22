# Bot Demo — Salón Luna

Bot demostrativo usado en la prospección de Digitaliza. Los prospectos lo prueban vía `wa.me/<numero>` antes de agendar llamada con Eduardo.

## Cómo se monta

1. Clonar el repo principal de `digitaliza-bot-base` a un nuevo directorio (ej. `salon-luna-bot`).
2. Sobrescribir los archivos `negocio.txt` y `catalogo.txt` de la raíz con los de este directorio.
3. Conseguir un **SIM nueva** para este bot (~$150 MXN, cualquier OXXO).
4. Alta en Meta Business Suite con el número nuevo + WABA propio (o sub-account de tu portfolio).
5. Conectar YCloud con el número y apuntar webhook a la nueva URL de Railway del bot Luna.
6. Deploy a Railway nuevo servicio:
   - Repo: `salon-luna-bot` (clone de `digitaliza-bot-base`)
   - Variables de entorno (mismas que el principal pero con keys separadas si quieres aislar):
     - `GEMINI_API_KEY`
     - `YCLOUD_API_KEY`
     - `YCLOUD_WEBHOOK_VERIFY_TOKEN` (distinto del principal)
     - `BOT_PHONE=<numero-sim-nueva-sin-+>`
     - `OWNER_PHONE=525635849043` (tu personal — quieres recibir leads curiosos)
     - `GOOGLE_*` (calendario opcional — Salón Luna podría usar un calendario demo también, o desactivado)
     - `BACKUP_ADMIN_TOKEN` (recomendado, distinto del principal)
7. Cuando Railway dé el URL público, ponerlo en el webhook de YCloud.
8. Probar: mándale WhatsApp desde otro número y ve que responda como Salón Luna.

## Cómo el prospecto llega al bot

En el prompt del bot PRINCIPAL de Digitaliza, cuando un prospecto muestre interés claro ("¿cómo funciona?", "sí me interesa"), el bot comparte:

- Link del video Loom (ver `../GUION-VIDEO-DEMO.md`)
- Link directo `wa.me/<numero-salon-luna>` para que platique con Salón Luna

## Notas

- Calendario de Salón Luna: si quieres que las "citas" que se agenden no contaminen tu calendario personal, crea un Google Calendar dedicado "Demo — Salón Luna" y úsalo como `GOOGLE_CALENDAR_ID` del bot Luna. Si te da flojera, desactiva agenda quitando las variables de Google — el bot responde horarios hipotéticos.
- Este bot NO debe aparecer públicamente como bot de Digitaliza; el prospecto debe creer que es un salón real. Si el prospecto pregunta directamente si es demo o si quiere contratar el sistema, el `catalogo.txt` ya trae la regla para redirigirlo a ti.
- Considera loggear aparte las conversaciones del bot Luna, no se mezclen con clientes reales.
