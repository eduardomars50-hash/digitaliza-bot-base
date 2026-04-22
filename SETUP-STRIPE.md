# Setup Stripe — Links de pago para Digitaliza

No puedo crear los links de Stripe por ti (necesita tu login), pero aquí está el paso a paso exacto para dejar listos los 3 links que vas a reusar con todos los clientes.

**Tiempo estimado:** 15 minutos (si ya tienes cuenta Stripe México verificada).

---

## Paso 0 — Si no tienes cuenta Stripe México

1. Entra a `dashboard.stripe.com/register` y crea cuenta con tu correo `eduardomars50@gmail.com`.
2. Completa verificación de negocio:
   - RFC de Eduardo Mares Gutiérrez (persona física con actividad empresarial)
   - Domicilio fiscal
   - INE escaneada
   - Cuenta bancaria mexicana (CLABE) para depósitos
3. Activa la cuenta. Stripe tarda 1-2 días hábiles en verificar.

Mientras, pasa al Paso 1 con la cuenta en modo **Test** para preparar los links.

---

## Paso 1 — Crear los 3 Payment Links base

En Stripe, los "Payment Links" son URLs que el cliente abre y paga con tarjeta sin que tú hagas nada. Se configuran una vez y los reusa con todos los clientes.

### Link 1 — Anticipo Tier Arranque (50%)

1. Dashboard Stripe → **Payment Links** → botón **"New"** (arriba derecha).
2. **Product**: click "Add a product" → "Create a new product".
3. Llena:
   - **Name:** `Digitaliza — Anticipo Tier Arranque`
   - **Description:** `50% de anticipo para implementación de chatbot IA en WhatsApp, Tier Arranque.`
   - **Amount:** `$750 MXN`
   - **Currency:** MXN
   - **Pricing:** one-time (no recurrente).
4. Click **Save product**.
5. En la sección "Collect tax and billing address" → activa "Collect customer's address" (para tu factura de IVA si lo emites).
6. Click **Create link**. Copia el URL que te da (tipo `https://buy.stripe.com/...`).
7. Guárdalo como `ANTICIPO_ARRANQUE`.

### Link 2 — Anticipo Tier Estándar (50%)

Mismo proceso pero con:
- Name: `Digitaliza — Anticipo Tier Estándar`
- Amount: `$1,250 MXN`
- Descripción equivalente.

Guárdalo como `ANTICIPO_ESTANDAR`.

### Link 3 — Anticipo Tier Avanzado (si aplica)

Mismo proceso con el monto de tu Tier Avanzado (50% del setup correspondiente).

---

## Paso 2 — Links para liquidación (50% final)

Repite el Paso 1 pero con los montos de liquidación:

- `Digitaliza — Liquidación Tier Arranque` → `$750 MXN`
- `Digitaliza — Liquidación Tier Estándar` → `$1,250 MXN`
- `Digitaliza — Liquidación Tier Avanzado` → (según tier)

---

## Paso 3 — Suscripciones mensuales

Para las mensualidades, Stripe recomienda usar **Subscriptions** (cobro recurrente automático).

1. Dashboard → **Products** → **Add product**.
2. Name: `Digitaliza — Mensualidad Tier Arranque`
3. **Pricing model**: "Recurring"
4. Amount: `$1,500 MXN`
5. Billing period: **Monthly**
6. Save → en el producto creado, click **Create payment link**.
7. Copia el URL como `MENSUALIDAD_ARRANQUE`.

Repite para Tier Estándar ($2,500/mes) y Avanzado.

---

## Paso 4 — Tabla final de links

Guarda estos links en tu notas o en Obsidian:

```
ANTICIPO_ARRANQUE:       https://buy.stripe.com/...
LIQUIDACIÓN_ARRANQUE:    https://buy.stripe.com/...
MENSUALIDAD_ARRANQUE:    https://buy.stripe.com/...

ANTICIPO_ESTANDAR:       https://buy.stripe.com/...
LIQUIDACIÓN_ESTANDAR:    https://buy.stripe.com/...
MENSUALIDAD_ESTANDAR:    https://buy.stripe.com/...

ANTICIPO_AVANZADO:       https://buy.stripe.com/...
LIQUIDACIÓN_AVANZADO:    https://buy.stripe.com/...
MENSUALIDAD_AVANZADO:    https://buy.stripe.com/...
```

---

## Paso 5 — Configurar webhook de Stripe al bot (opcional)

Si quieres que el bot se entere automáticamente cuando un cliente paga (sin que tú revises Stripe), puedes configurar un webhook que apunte a tu bot.

- **Endpoint Stripe** → tu bot en Railway, ej. `https://digitaliza-bot.up.railway.app/webhook/stripe`.
- **Eventos a escuchar:** `checkout.session.completed`, `invoice.paid`, `customer.subscription.deleted`.
- En el bot habría que agregar el handler Flask (yo lo hago cuando me avises que los links ya están vivos).

Esto es opcional; sin webhook, tú checas Stripe manualmente cuando un cliente diga "ya pagué".

---

## Paso 6 — SPEI como alternativa

Stripe no procesa SPEI nativo en México. Para clientes que prefieren SPEI:

- Dale los datos de tu cuenta bancaria directo (CLABE + banco + titular "Eduardo Mares Gutiérrez").
- Pídele comprobante de la transferencia.
- Registra el pago manual en tu Stripe para llevar contabilidad centralizada (Dashboard → Payments → Add manual payment).

---

## Cuando ya tengas los links

Avísame. Haré 2 cosas en el bot:

1. Agregar al prompt los links correspondientes para que, cuando un cliente cierre y pida cómo pagar, el bot le comparta directo el link del tier que acordaron.
2. Si configuraste webhook, implemento el handler `/webhook/stripe` para que el bot detecte pagos y actualice el estado del cliente automáticamente.
