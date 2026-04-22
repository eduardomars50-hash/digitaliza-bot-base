# Seguridad Operativa — Digitaliza Bot

Guía corta de mantenimiento de credenciales y respuesta a incidentes.

---

## Rotación periódica de credenciales

**Calendario sugerido:** trimestral (cada 3 meses) para keys activas, inmediato si se sospecha filtración.

### Inventario de credenciales

| Credencial | Dónde vive | Rotación | Acción rotativa |
|---|---|---|---|
| `GEMINI_API_KEY` | Railway env + Google AI Studio | Trimestral | Crear nueva en [aistudio.google.com/apikey](https://aistudio.google.com/apikey), actualizar Railway, revocar vieja |
| `GROQ_API_KEY` | Railway env + Groq Cloud | Trimestral | [console.groq.com/keys](https://console.groq.com/keys) → crear nueva, actualizar, revocar vieja |
| `YCLOUD_API_KEY` | Railway env + panel YCloud | Trimestral | Panel YCloud → API Keys → regenerar, actualizar Railway |
| `YCLOUD_WEBHOOK_VERIFY_TOKEN` | Railway env + panel YCloud webhook | Semestral | Cambiar en Railway, luego en YCloud webhook config |
| `GOOGLE_REFRESH_TOKEN` | Railway env | Anual (a menos que Google caduque) | Re-autorizar con `python oauth_setup.py`, pegar nuevo |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Railway env + Google Cloud Console | Solo si se sospecha filtración | [console.cloud.google.com](https://console.cloud.google.com) → OAuth consent → credenciales |
| `BACKUP_ADMIN_TOKEN` | Railway env | Semestral o ante filtración | `openssl rand -hex 32` → actualizar Railway → actualizar `reference_digitaliza_tokens.md` en memoria PAI |
| Cuenta de Stripe | Stripe dashboard | Semestral password + 2FA siempre activo | Cambiar password + refrescar API keys de Stripe |
| GitHub token (si existe backup repo privado) | Railway env | Trimestral | [github.com/settings/tokens](https://github.com/settings/tokens) |

### Procedimiento estándar de rotación

Para cualquier key de las de arriba:

1. **Generar la nueva** en el panel correspondiente. No revocar la vieja todavía.
2. **Actualizar en Railway** la variable de entorno:
   ```
   railway variables --set "CLAVE=nuevo_valor"
   ```
   Railway redeploya automáticamente.
3. **Verificar que el bot siga funcionando** (mándale un mensaje de prueba).
4. **Revocar la vieja key** en el panel origen (esperar 5 min después del redeploy, por si hay races).
5. Registrar la rotación en `security_logs.json` manualmente o vía log:
   ```
   "tipo": "rotacion_key", "mensaje": "GEMINI_API_KEY rotada 2026-04-21"
   ```

---

## Qué hacer si crees que una key se filtró

**Filtración = cualquiera de estos escenarios:**
- Commit accidental de la key en git (incluso si se revierte, git conserva historial)
- Compartir por error la key en chat/correo
- Laptop comprometida o clonada sin autorización
- Aparece la key en un servicio público (GitHub, pastebin, etc.)

**Acción inmediata (máx 10 min):**

1. **Revocar la vieja key AHORA** en el panel de origen. No esperes a rotar con calma.
2. Generar y setear nueva en Railway (procedimiento estándar, pasos 1-2 de arriba).
3. Buscar en los logs de Railway y en `security_logs.json` actividad sospechosa desde el momento de la filtración (uso inusual de API, IPs desconocidas, volúmenes de mensajes no coherentes).
4. Si hay evidencia de uso no autorizado: **notificar al INAI** si hubo acceso a datos personales de clientes finales (obligación LFPDPPP). Contactar a un abogado.
5. Post-mortem: documentar causa raíz (¿cómo se filtró?) y ajustar proceso para que no vuelva a pasar.

---

## Backup: buenas prácticas

- **Descargar snapshot off-site al menos 1 vez por semana.** Automatízalo con un script local:
  ```bash
  curl "https://tu-bot.up.railway.app/admin/backup-latest?token=$BACKUP_TOKEN" \
    -o ~/digitaliza-backups/backup-$(date +%Y%m%d).tar.gz
  ```
  Agenda en cron local:
  ```
  0 9 * * 1 /path/al/script/backup.sh  # lunes 9am
  ```
- **Verifica periódicamente** que los backups se puedan descomprimir y no estén corruptos:
  ```bash
  tar -tzf ultimo-backup.tar.gz | head
  ```
- **No compartas** los backups. Contienen historial completo de conversaciones con clientes finales, datos personales, etc.

---

## Mínimo indispensable de monitoreo

### Alertas a revisar semanalmente

1. `/admin/metrics` — ¿subió el tráfico? ¿bajó? ¿hay tasas de conversión anómalas?
2. `security_logs.json` (lo puedes ver descargando el backup y abriendo el archivo) — ¿cuántos intentos de jailbreak? ¿todos del mismo número? Si sí, bloquearlo manualmente.
3. Logs de Railway — buscar `[GEMINI] Falló` o `[BACKUP] No se pudo`. Cada ocurrencia merece revisión.

### Alertas proactivas al OWNER (ya configuradas)

El bot ya notifica al `OWNER_PHONE` automáticamente cuando:

- Falla Gemini 3 veces seguidas (`🚨 Gemini falló 3 veces...`)
- Se detecta takeover manual desde app nativa (`🟡 Detecté que le escribiste a...`)
- Hay nuevo prospecto entrando
- Lead capturado + llamada agendada
- Prospecto quiere contratar

Si no recibes estas notificaciones por varios días y sabes que hubo tráfico, algo se rompió. Revisa logs de Railway y `/admin/metrics`.

---

## Checklist trimestral (imprimible)

```
FECHA: ______________

[ ] Rotar GEMINI_API_KEY
[ ] Rotar GROQ_API_KEY
[ ] Rotar YCLOUD_API_KEY
[ ] Rotar BACKUP_ADMIN_TOKEN (si han pasado 6+ meses)
[ ] Verificar /admin/metrics — tasas de conversión
[ ] Descargar y probar descompresión del último backup
[ ] Revisar security_logs.json — ¿hay patrones de ataque?
[ ] Verificar que las notificaciones al OWNER siguen llegando
[ ] Actualizar AVISO-PRIVACIDAD.md si hay cambios relevantes
[ ] Revisar proveedores BSP/IA: ¿cambios en políticas o precios?
```
