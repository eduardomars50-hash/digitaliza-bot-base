# Contrato de Prestación de Servicios — Digitaliza

**Entre:**

- **El Prestador:** Eduardo Mares Gutiérrez, con domicilio en Mérida, Yucatán, México, en lo sucesivo referido como **"Digitaliza"**.
- **El Cliente:** `{{NOMBRE_CLIENTE_O_RAZÓN_SOCIAL}}`, con domicilio en `{{DOMICILIO_CLIENTE}}`, RFC `{{RFC_CLIENTE}}`, representado por `{{NOMBRE_REPRESENTANTE}}` en su carácter de `{{CARGO}}`, en lo sucesivo referido como **"El Cliente"**.

**Fecha de firma:** `{{FECHA}}`
**Vigencia inicial:** `{{MESES_VIGENCIA}}` meses desde la fecha de firma.

---

## 1. Objeto del contrato

Digitaliza se obliga a desarrollar, implementar y mantener en operación un asistente virtual ("el Bot") para WhatsApp Business, basado en inteligencia artificial (Gemini 2.0 Flash de Google o el modelo equivalente vigente al momento de la operación), que responderá en nombre del Cliente a mensajes entrantes en el número de WhatsApp dedicado al negocio del Cliente.

## 2. Alcance del servicio

El servicio incluye:

- Configuración del Bot con el catálogo de servicios, precios, horarios y políticas proporcionados por el Cliente.
- Conexión a WhatsApp Business Platform vía el proveedor de servicios (BSP) YCloud.
- Capacidad de respuesta 24/7 a mensajes de texto, notas de voz, imágenes, videos y documentos PDF.
- Agenda de citas conectada a Google Calendar del Cliente (si aplica al tier contratado).
- Notificaciones al WhatsApp personal del Cliente cuando un prospecto califique como lead o manifieste intención de compra.
- Modo de coexistencia con WhatsApp Business nativo del Cliente (pausa automática cuando el Cliente toma el control manual de una conversación).
- Soporte técnico durante los primeros 30 días posteriores a la puesta en producción ("Go Live").

El servicio NO incluye:

- Compra del número telefónico SIM dedicado para el Bot (lo adquiere el Cliente).
- Costos de mensajes facturados por Meta/WhatsApp Platform fuera del plan gratuito (1,000 conversaciones mensuales del tipo "Utility" y "Service" al momento de la firma).
- Desarrollo de funcionalidades no incluidas en el Tier contratado.
- Consultoría de estrategia comercial, marketing o publicidad.

## 3. Plan contratado

- **Tier:** `{{TIER}}` (Arranque | Estándar | Avanzado)
- **Inversión inicial (setup, pago único):** `${{SETUP_MXN}}` MXN más IVA.
- **Mensualidad:** `${{MENSUALIDAD_MXN}}` MXN más IVA, a partir del segundo mes de operación.
- El primer mes de operación está incluido en la inversión inicial.

## 4. Forma y condiciones de pago

- **Anticipo:** 50% de la inversión inicial (`${{ANTICIPO_MXN}}` MXN más IVA) al momento de la firma de este contrato.
- **Liquidación:** 50% restante al término de la implementación y puesta en producción del Bot.
- **Mensualidad:** domiciliada los días 1 de cada mes a partir del segundo mes, vía Stripe, transferencia SPEI o el medio que ambas partes acuerden por escrito.

Formas de pago aceptadas: transferencia electrónica SPEI, tarjeta de crédito/débito vía Stripe, o efectivo previa coordinación.

## 5. Plazo de implementación

Digitaliza se compromete a poner el Bot en producción dentro de los **5 días hábiles posteriores** a haber recibido por parte del Cliente:

1. Comprobante del pago del anticipo.
2. Catálogo completo de servicios y precios del Cliente.
3. Información del negocio (horarios, políticas, tono de comunicación, temas prohibidos).
4. Acceso al número SIM dedicado conectado a WhatsApp Business Platform.
5. Credenciales de Google Calendar (si aplica al tier).

Si el Cliente demora en proporcionar alguno de estos insumos, el plazo se extiende proporcionalmente.

## 6. Obligaciones del Cliente

- Proporcionar información veraz y actualizada del negocio al momento del onboarding.
- Adquirir y mantener activa una SIM dedicada para el Bot.
- Dar acceso a Digitaliza al panel de Meta Business Suite del Cliente cuando se requiera administrar la cuenta WhatsApp Business del Cliente.
- Notificar por escrito (WhatsApp o correo) cualquier cambio de precios, servicios, horarios o políticas del negocio que deban reflejarse en el Bot.
- Pagar puntualmente las mensualidades del servicio.

## 7. Obligaciones de Digitaliza

- Entregar el Bot funcional en los tiempos pactados en la cláusula 5.
- Brindar soporte técnico de primer nivel durante los primeros 30 días.
- Mantener respaldos automáticos (snapshots cada 6 horas) de las conversaciones, leads y configuración del Bot.
- Notificar al Cliente ante cualquier incidencia que afecte la operación del Bot por más de 2 horas continuas.
- Cumplir con la Ley Federal de Protección de Datos Personales en Posesión de los Particulares (ver Aviso de Privacidad adjunto).

## 8. Propiedad intelectual

- El **código del Bot** es propiedad intelectual de Digitaliza. El Cliente recibe una licencia de uso no exclusiva y no transferible durante la vigencia del contrato.
- El **catálogo, historial de conversaciones y base de leads** es propiedad del Cliente. A la terminación del contrato, Digitaliza se compromete a entregar al Cliente un respaldo completo de estos datos en formato descargable.
- El Cliente autoriza a Digitaliza a usar el nombre y logo del negocio como referencia comercial (casos de éxito) a menos que lo exprese en contrario por escrito.

## 9. Confidencialidad

Ambas partes se obligan a mantener confidencialidad sobre toda información no pública que reciban con motivo del presente contrato, incluyendo pero no limitado a: catálogos, precios, clientes finales, estrategias comerciales y credenciales técnicas. Esta obligación subsiste por 2 años posteriores a la terminación del contrato.

## 10. Protección de datos personales

Digitaliza actúa como **Encargado de Datos Personales** bajo la LFPDPPP. El tratamiento se limita a los fines operativos del Bot. Digitaliza no cede ni comercializa los datos de los clientes finales del Cliente. El Aviso de Privacidad completo se anexa al presente contrato.

## 11. Terminación

Cualquiera de las partes podrá dar por terminado este contrato dando aviso por escrito con **30 días naturales de anticipación**. En caso de terminación:

- El Cliente pagará las mensualidades devengadas hasta la fecha de terminación.
- Digitaliza entregará al Cliente el respaldo completo de datos en 5 días hábiles.
- Digitaliza desactivará el Bot en la fecha efectiva de terminación.

## 12. Rescisión por incumplimiento

Podrá rescindirse el contrato sin responsabilidad para la parte cumplida si:

- El Cliente no paga dos mensualidades consecutivas.
- Cualquiera de las partes incumple obligaciones esenciales y no subsana en 10 días hábiles después del requerimiento por escrito.
- El Cliente usa el Bot para fines ilegales, engañosos, o contrarios a los Términos de Servicio de WhatsApp/Meta.

## 13. Jurisdicción y ley aplicable

Las partes se someten expresamente a la jurisdicción de los tribunales competentes de la Ciudad de Mérida, Yucatán, México, y a la legislación mexicana vigente.

---

## Firmas

**Por Digitaliza:**

`______________________________________`
Eduardo Mares Gutiérrez
eduardomars50@gmail.com
+52 56 3183 2858

**Por El Cliente:**

`______________________________________`
`{{NOMBRE_REPRESENTANTE}}`
`{{CORREO_CLIENTE}}`
`{{TELEFONO_CLIENTE}}`

---

## Anexos

- **Anexo A:** Aviso de Privacidad (ver `AVISO-PRIVACIDAD.md`)
- **Anexo B:** Descripción técnica del servicio (ver `MANUAL-CLIENTE.md`)
- **Anexo C:** Catálogo de servicios del Cliente (adjuntar por el Cliente)
