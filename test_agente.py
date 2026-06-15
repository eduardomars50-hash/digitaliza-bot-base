"""Tests mínimos de regresión para funciones puras del bot.

Objetivo: blindar que rename/refactor no rompa helpers críticos.
No arranca Flask, no llama a Gemini/YCloud/Groq, no toca /data real.

Ejecutar (sin dependencias externas, solo stdlib):
    cd ~/digitaliza-bot-base
    python3 test_agente.py

Salida esperada: todos los tests en verde + "OK".
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Apuntar DATA_DIR a un tmp antes de importar agente.py
_TMP_DATA = tempfile.mkdtemp(prefix="digitaliza_test_")
os.environ["DATA_DIR"] = _TMP_DATA
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("YCLOUD_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")

sys.path.insert(0, str(Path(__file__).parent))

# Stubs para módulos pesados (no queremos instalarlos para correr tests)
import types  # noqa: E402

for name in ["flask", "google", "google.generativeai",
             "google.generativeai.types", "groq", "PIL", "PIL.Image",
             "requests"]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["flask"].Flask = lambda *a, **k: types.SimpleNamespace(
    route=lambda *a, **k: (lambda f: f),
    get=lambda *a, **k: (lambda f: f),
    post=lambda *a, **k: (lambda f: f),
)
sys.modules["flask"].request = types.SimpleNamespace(
    args={}, get_json=lambda silent=True: {})
sys.modules["flask"].jsonify = lambda *a, **k: (a, k)
sys.modules["flask"].send_file = lambda *a, **k: None
sys.modules["PIL"].Image = types.SimpleNamespace(open=lambda *a, **k: None)


class _FakeEnum:
    def __getattr__(self, name):
        return name


sys.modules["google.generativeai.types"].HarmCategory = _FakeEnum()
sys.modules["google.generativeai.types"].HarmBlockThreshold = _FakeEnum()
sys.modules["google.generativeai"].configure = lambda *a, **k: None
sys.modules["google.generativeai"].GenerativeModel = lambda *a, **k: None
sys.modules["google.generativeai"].types = sys.modules["google.generativeai.types"]
sys.modules["groq"].Groq = lambda *a, **k: None
sys.modules["requests"].post = lambda *a, **k: None
sys.modules["requests"].get = lambda *a, **k: None

import agente  # noqa: E402


class TestNormalizarNumero(unittest.TestCase):
    def test_quita_plus(self):
        self.assertEqual(agente.normalizar_numero("+525512345678"), "525512345678")

    def test_quita_espacios_y_guiones(self):
        self.assertEqual(
            agente.normalizar_numero("+52 55 1234-5678"), "525512345678"
        )

    def test_solo_digitos(self):
        self.assertEqual(agente.normalizar_numero("525512345678"), "525512345678")

    def test_vacio_o_invalido(self):
        self.assertEqual(agente.normalizar_numero(""), "")
        self.assertEqual(agente.normalizar_numero("abc"), "")


class TestParseNegocio(unittest.TestCase):
    def test_campos_basicos(self):
        texto = "NOMBRE: Salón Demo\nTIPO: Salón\nHORARIO: L-V 10-18\n"
        r = agente._parse_negocio(texto)
        self.assertEqual(r["nombre"], "Salón Demo")
        self.assertEqual(r["tipo"], "Salón")
        self.assertEqual(r["horario"], "L-V 10-18")

    def test_nuevos_campos_agenda(self):
        texto = ("NOMBRE: Test\nAGENDA_DIAS: M,X,J,V,S\n"
                 "AGENDA_HORA_INICIO: 10\nAGENDA_HORA_FIN: 19\n")
        r = agente._parse_negocio(texto)
        self.assertEqual(r["agenda_dias"], "M,X,J,V,S")
        self.assertEqual(r["agenda_hora_inicio"], "10")
        self.assertEqual(r["agenda_hora_fin"], "19")

    def test_defaults_vacios(self):
        r = agente._parse_negocio("")
        self.assertEqual(r["nombre"], "")
        self.assertEqual(r["agenda_dias"], "")


class TestParseDias(unittest.TestCase):
    def test_lista_abreviada(self):
        self.assertEqual(agente._parse_dias("L,M,X,J,V"), {0, 1, 2, 3, 4})

    def test_rango_L_V(self):
        self.assertEqual(agente._parse_dias("L-V"), {0, 1, 2, 3, 4})

    def test_rango_L_S(self):
        self.assertEqual(agente._parse_dias("L-S"), {0, 1, 2, 3, 4, 5})

    def test_nombres_completos(self):
        self.assertEqual(
            agente._parse_dias("Lunes,Martes,Miércoles"), {0, 1, 2}
        )

    def test_default_L_V(self):
        self.assertEqual(agente._parse_dias(""), {0, 1, 2, 3, 4})

    def test_solo_domingo(self):
        self.assertEqual(agente._parse_dias("D"), {6})


class TestParseHora(unittest.TestCase):
    def test_formatos_validos(self):
        self.assertEqual(agente._parse_hora("15", 0), 15)
        self.assertEqual(agente._parse_hora("15:00", 0), 15)
        self.assertEqual(agente._parse_hora("9h", 0), 9)

    def test_default_cuando_invalido(self):
        self.assertEqual(agente._parse_hora("", 15), 15)
        self.assertEqual(agente._parse_hora("abc", 20), 20)
        self.assertEqual(agente._parse_hora("25", 15), 15)
        self.assertEqual(agente._parse_hora("-1", 15), 15)


class TestTroceoMensajes(unittest.TestCase):
    def test_partido_por_frases(self):
        texto = (
            "Hola, gracias por escribirnos. "
            "Claro que sí, te explico con calma. "
            "Primero revisamos lo que necesitas. "
            "Después te paso la mejor opción. "
        ) * 8
        partes = agente._trocear(texto, 200)
        self.assertGreater(len(partes), 1)
        self.assertTrue(all(len(p) <= 200 for p in partes))
        self.assertTrue(all(p.strip() for p in partes))


class TestPdfSoporte(unittest.TestCase):
    def test_document_pdf_pasa_por_meta(self):
        meta_calls = []
        old_meta = agente.meta_descargar_media
        old_ycloud = agente.ycloud_descargar_media
        agente.meta_descargar_media = lambda media_id, media_obj=None: (
            meta_calls.append((media_id, media_obj)) or b"%PDF-1.4 fake"
        )
        agente.ycloud_descargar_media = lambda media_id, media_obj=None: b"ycloud"
        try:
            blob, obj = agente.descargar_media_mensaje(
                {"document": {"id": "pdf123", "mimeType": "application/pdf"}},
                "document",
                "meta",
            )
        finally:
            agente.meta_descargar_media = old_meta
            agente.ycloud_descargar_media = old_ycloud

        self.assertEqual(blob, b"%PDF-1.4 fake")
        self.assertEqual(obj["id"], "pdf123")
        self.assertEqual(len(meta_calls), 1)

    def test_tier_detectado_se_elimina(self):
        t = agente._sanitizar_salida(
            "[TIER_DETECTADO: tier=2-alto]\nPerfecto, te ayudo con eso."
        )
        self.assertNotIn("TIER_DETECTADO", t.upper())
        self.assertIn("Perfecto", t)


class TestPausas(unittest.TestCase):
    def setUp(self):
        # Reset config path to clean slate per test
        self.phone = "525599887766"
        try:
            agente._despausar_chat(self.phone)
        except Exception:
            pass

    def test_ciclo_completo(self):
        self.assertFalse(agente._esta_pausado(self.phone))
        exp = agente._pausar_chat(self.phone, minutos=5, source="test")
        self.assertIsInstance(exp, datetime)
        self.assertTrue(agente._esta_pausado(self.phone))
        self.assertTrue(agente._esta_pausado("+" + self.phone))  # normaliza
        vivos = agente._listar_pausados()
        self.assertTrue(any(v["phone"] == self.phone for v in vivos))
        self.assertTrue(agente._despausar_chat(self.phone))
        self.assertFalse(agente._esta_pausado(self.phone))
        self.assertFalse(agente._despausar_chat(self.phone))

    def test_cap_maximo(self):
        phone = "525511112222"
        exp = agente._pausar_chat(phone, minutos=99999, source="test")
        diff_min = (exp - datetime.utcnow()).total_seconds() / 60
        self.assertLess(abs(diff_min - agente.PAUSA_MAX_MIN), 2)
        agente._despausar_chat(phone)


class TestBotSentTracking(unittest.TestCase):
    def test_por_prefijo(self):
        ext_id = f"{agente._BOT_SENT_PREFIX}abc123"
        self.assertTrue(agente._es_id_de_bot(ext_id))

    def test_por_registry(self):
        wamid = "wamid.HBgNNTE1NTk4ODc3NjYVAgARGBI_custom"
        agente._marcar_id_de_bot(wamid)
        self.assertTrue(agente._es_id_de_bot(wamid))

    def test_desconocido_no_es_bot(self):
        self.assertFalse(agente._es_id_de_bot("wamid.externo_xyz"))
        self.assertFalse(agente._es_id_de_bot(""))


class TestMetaCloudAPI(unittest.TestCase):
    def setUp(self):
        self._old_phone_id = agente.WHATSAPP_PHONE_NUMBER_ID
        self._old_token = agente.WHATSAPP_ACCESS_TOKEN
        self._old_url = agente.META_MESSAGES_URL
        self._old_post = agente.requests.post

    def tearDown(self):
        agente.WHATSAPP_PHONE_NUMBER_ID = self._old_phone_id
        agente.WHATSAPP_ACCESS_TOKEN = self._old_token
        agente.META_MESSAGES_URL = self._old_url
        agente.requests.post = self._old_post

    def test_meta_message_to_internal_text(self):
        agente.WHATSAPP_PHONE_NUMBER_ID = "1108506122354011"
        value = {
            "metadata": {
                "display_phone_number": "525638678463",
                "phone_number_id": "1108506122354011",
            }
        }
        message = {
            "from": "5219991234567",
            "id": "wamid.TEST",
            "timestamp": "1710000000",
            "type": "text",
            "text": {"body": "hola prueba real"},
        }
        internal = agente._mensaje_meta_a_interno(message, value)
        self.assertEqual(internal["_source"], "meta")
        self.assertEqual(internal["from"], "5219991234567")
        self.assertEqual(internal["to"], "525638678463")
        self.assertEqual(internal["type"], "text")
        self.assertEqual(internal["text"]["body"], "hola prueba real")

    def test_meta_phone_id_mismatch_ignored(self):
        agente.WHATSAPP_PHONE_NUMBER_ID = "real"
        value = {"metadata": {"phone_number_id": "otro"}}
        message = {
            "from": "5219991234567",
            "id": "wamid.TEST",
            "type": "text",
            "text": {"body": "hola"},
        }
        self.assertIsNone(agente._mensaje_meta_a_interno(message, value))

    def test_meta_send_text_payload(self):
        agente.WHATSAPP_ACCESS_TOKEN = "dummy_token_no_real"
        agente.WHATSAPP_PHONE_NUMBER_ID = "1108506122354011"
        agente.META_MESSAGES_URL = (
            "https://graph.facebook.com/v25.0/1108506122354011/messages"
        )
        calls = []

        class Resp:
            status_code = 200
            text = "{}"
            content = b"{}"

            def json(self):
                return {"messages": [{"id": "wamid.OUT"}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, headers, json, timeout))
            return Resp()

        agente.requests.post = fake_post
        ok, err = agente.meta_enviar_texto("5219991234567", "respuesta prueba")
        self.assertTrue(ok)
        self.assertEqual(err, "")
        url, headers, payload, timeout = calls[0]
        self.assertEqual(
            url, "https://graph.facebook.com/v25.0/1108506122354011/messages"
        )
        self.assertTrue(headers["Authorization"].startswith("Bearer "))
        self.assertEqual(payload["messaging_product"], "whatsapp")
        self.assertEqual(payload["to"], "5219991234567")
        self.assertEqual(payload["type"], "text")
        self.assertEqual(payload["text"]["body"], "respuesta prueba")
        self.assertEqual(timeout, 20)

    def test_meta_descargar_media_via_graph(self):
        agente.WHATSAPP_ACCESS_TOKEN = "dummy_token_no_real"
        agente.META_GRAPH_VERSION = "v25.0"
        calls = []

        class RespMeta:
            status_code = 200
            content = b'{"url":"https://cdn.example.com/file.ogg"}'
            text = '{"url":"https://cdn.example.com/file.ogg"}'
            headers = {"Content-Type": "application/json"}

            def json(self):
                return {"url": "https://cdn.example.com/file.ogg"}

        class RespBin:
            status_code = 200
            content = b"audio-bytes"
            text = ""
            headers = {"Content-Type": "audio/ogg"}

            def json(self):
                return {}

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            return RespMeta() if len(calls) == 1 else RespBin()

        old_get = agente.requests.get
        agente.requests.get = fake_get
        try:
            blob = agente.meta_descargar_media("mid.123")
        finally:
            agente.requests.get = old_get

        self.assertEqual(blob, b"audio-bytes")
        self.assertEqual(calls[0][0], "https://graph.facebook.com/v25.0/mid.123")
        self.assertEqual(calls[1][0], "https://cdn.example.com/file.ogg")

    def test_descargar_media_mensaje_prefiere_meta(self):
        meta_calls = []
        ycloud_calls = []

        old_meta = agente.meta_descargar_media
        old_ycloud = agente.ycloud_descargar_media
        agente.meta_descargar_media = lambda media_id, media_obj=None: (
            meta_calls.append((media_id, media_obj)) or b"meta-bytes"
        )
        agente.ycloud_descargar_media = lambda media_id, media_obj=None: (
            ycloud_calls.append((media_id, media_obj)) or b"ycloud-bytes"
        )
        try:
            blob, obj = agente.descargar_media_mensaje(
                {"audio": {"id": "audio123"}},
                "audio",
                "meta",
            )
        finally:
            agente.meta_descargar_media = old_meta
            agente.ycloud_descargar_media = old_ycloud

        self.assertEqual(blob, b"meta-bytes")
        self.assertEqual(obj["id"], "audio123")
        self.assertEqual(len(meta_calls), 1)
        self.assertEqual(len(ycloud_calls), 0)


class TestAgendaConfig(unittest.TestCase):
    def test_estructura_valida(self):
        cfg = agente.obtener_agenda_config()
        self.assertIn("dias_weekdays", cfg)
        self.assertIn("hora_inicio", cfg)
        self.assertIn("hora_fin", cfg)
        self.assertGreater(cfg["hora_fin"], cfg["hora_inicio"])
        self.assertIsInstance(cfg["dias_weekdays"], set)


class TestSecurityLog(unittest.TestCase):
    def test_evento_persiste(self):
        agente._log_security_event(
            "525500000000", "test_tipo", "mensaje de prueba"
        )
        if agente.SECURITY_LOG_PATH.exists():
            import json
            datos = json.loads(agente.SECURITY_LOG_PATH.read_text(encoding="utf-8"))
            self.assertTrue(any(e["tipo"] == "test_tipo" for e in datos))


class TestSanitizerSalida(unittest.TestCase):
    """Red final anti-leak de tags. Los casos cubren los leaks reales
    vistos en producción 2026-04-22 (screenshot de Eduardo)."""

    def test_tag_bien_formado_se_elimina(self):
        t = agente._sanitizar_salida(
            "[CALENDARIO:CONSULTAR:2026-04-23]\nDéjame ver qué tengo libre."
        )
        self.assertNotIn("CALENDARIO", t.upper())
        self.assertIn("Déjame", t)

    def test_lead_capturado_bien_formado_se_elimina(self):
        t = agente._sanitizar_salida(
            "[LEAD_CAPTURADO: nombre=Juan; negocio=Barber; ciudad=Mérida]\n"
            "Genial, Juan. ¿Qué tal va el negocio?"
        )
        self.assertNotIn("LEAD_CAPTURADO", t.upper())
        self.assertIn("Genial", t)

    def test_leak_inline_sin_corchetes(self):
        """Caso real del screenshot: 'calendario consulta 2024-05-24'
        a mitad de frase."""
        t = agente._sanitizar_salida(
            "Claro, déjame checar. calendario consulta 2024-05-24\n"
            "¿Qué día te acomoda?"
        )
        self.assertNotIn("calendario consulta", t.lower())
        self.assertIn("Claro", t)

    def test_leak_parentesis_es_igual_a(self):
        """Caso real: '(negocio es igual a X, ciudad es igual a Y)'."""
        t = agente._sanitizar_salida(
            "Perfecto (nombre es igual a Juan, negocio es igual a "
            "consultorio, ciudad es igual a Mérida). ¿Qué día?"
        )
        self.assertNotIn("es igual a", t.lower())

    def test_texto_limpio_no_se_toca(self):
        original = "¡Qué onda! Aquí de Digitaliza. ¿En qué te ayudamos?"
        self.assertEqual(agente._sanitizar_salida(original), original)

    def test_evento_quiere_web_se_elimina(self):
        t = agente._sanitizar_salida(
            "Te paso con Eduardo para cotizar bien.\n[EVENTO:QUIERE_WEB]"
        )
        self.assertNotIn("QUIERE_WEB", t.upper())

    def test_fallback_si_quedo_vacio(self):
        """Si Gemini respondió solo con tags, devolvemos texto neutral."""
        t = agente._sanitizar_salida(
            "[LEAD_CAPTURADO: nombre=X; negocio=Y; ciudad=Z]\n"
            "[EVENTO:QUIERE_CONTRATAR]"
        )
        self.assertTrue(len(t) > 0)


class TestExtractoresInteligenciaComercial(unittest.TestCase):
    """Tags de Fase 1 inspirados en bot de Emilio Bustani."""

    def test_alerta_precio_flag(self):
        limpio, flag = agente._extraer_alerta_precio(
            "El precio de lanzamiento es $2500.\n[ALERTA_PRECIO]"
        )
        self.assertTrue(flag)
        self.assertNotIn("ALERTA", limpio.upper())

    def test_intento_futuro_flag(self):
        limpio, flag = agente._extraer_intento_futuro(
            "Va, aquí estamos.\n[INTENTO_FUTURO]"
        )
        self.assertTrue(flag)
        self.assertNotIn("INTENTO", limpio.upper())

    def test_escalacion_flag(self):
        limpio, flag = agente._extraer_escalacion(
            "Ya le aviso a Eduardo.\n[ESCALACION]"
        )
        self.assertTrue(flag)
        self.assertNotIn("ESCALACION", limpio.upper())

    def test_competidor_con_payload(self):
        limpio, datos = agente._extraer_competidor(
            "Entiendo.\n[COMPETIDOR: nombre=ManyChat; precio=1500]"
        )
        self.assertEqual(datos, {"nombre": "ManyChat", "precio": "1500"})
        self.assertNotIn("COMPETIDOR", limpio.upper())

    def test_perdida_con_razon(self):
        limpio, razon = agente._extraer_perdida(
            "Entiendo.\n[PERDIDA: razon=precio]"
        )
        self.assertEqual(razon, "precio")
        self.assertNotIn("PERDIDA", limpio.upper())

    def test_referido_notas_con_coma(self):
        """Las notas pueden contener comas; el separador principal es ';'."""
        limpio, datos = agente._extraer_referido(
            "Genial.\n[REFERIDO: numero=5299887766; "
            "notas=mi cuñada Ale, consultorio dental]"
        )
        self.assertEqual(datos["numero"], "5299887766")
        self.assertEqual(datos["notas"], "mi cuñada Ale, consultorio dental")

    def test_referido_pendiente(self):
        limpio, datos = agente._extraer_referido(
            "[REFERIDO: numero=pendiente; notas=Claudia, vet Progreso]"
        )
        self.assertEqual(datos["numero"], "pendiente")

    def test_evento_web_flag(self):
        limpio, flag = agente._extraer_evento_web(
            "Te conecto con Eduardo.\n[EVENTO:QUIERE_WEB]"
        )
        self.assertTrue(flag)
        self.assertNotIn("WEB", limpio.upper())


class TestClasificacionTipoCliente(unittest.TestCase):
    """Segmentación de Fase 2 — niveles del embudo."""

    PHONE = "525544445555"

    def setUp(self):
        # Cada test parte de un estado limpio para ESTE phone.
        import shutil
        phone_norm = agente.normalizar_numero(self.PHONE)
        for d in (agente.SEGUIMIENTO_DIR, agente.LEADS_DIR):
            for suffix in (".flag", ".json"):
                for p in d.glob(f"{phone_norm}*{suffix}"):
                    p.unlink(missing_ok=True)

    def test_sin_nada_es_nuevo(self):
        self.assertEqual(
            agente._clasificar_tipo_cliente(self.PHONE), "nuevo"
        )

    def test_con_lead_es_prospecto(self):
        agente.guardar_lead(
            self.PHONE,
            {"nombre": "Juan", "negocio": "Barber", "ciudad": "Mérida"},
        )
        self.assertEqual(
            agente._clasificar_tipo_cliente(self.PHONE), "prospecto"
        )

    def test_con_quiere_contratar_es_cliente_activo(self):
        agente.guardar_lead(
            self.PHONE,
            {"nombre": "Juan", "negocio": "Barber", "ciudad": "Mérida"},
        )
        flag = agente.SEGUIMIENTO_DIR / (
            f"{agente.normalizar_numero(self.PHONE)}_quiere_contratar.flag"
        )
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("x")
        self.assertEqual(
            agente._clasificar_tipo_cliente(self.PHONE), "cliente_activo"
        )

    def test_con_referido_y_contrato_es_vip(self):
        ref = agente.SEGUIMIENTO_DIR / (
            f"{agente.normalizar_numero(self.PHONE)}_referido.flag"
        )
        qc = agente.SEGUIMIENTO_DIR / (
            f"{agente.normalizar_numero(self.PHONE)}_quiere_contratar.flag"
        )
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text("x")
        qc.write_text("x")
        self.assertEqual(
            agente._clasificar_tipo_cliente(self.PHONE), "vip"
        )

    def test_phone_vacio_cae_a_nuevo(self):
        self.assertEqual(agente._clasificar_tipo_cliente(""), "nuevo")

    def test_contexto_incluye_tipo_y_guidance(self):
        ctx = agente._contexto_tipo_cliente(self.PHONE)
        self.assertIn("nuevo", ctx)
        self.assertIn("TIPO DE PROSPECTO", ctx)


class TestAliasAdmin(unittest.TestCase):
    """Alias interno (separado del nombre del extractor automático).
    Permite a Eduardo etiquetar clientes con nombres propios sin tocar
    contactos de WhatsApp."""

    PHONE = "525577778888"

    def setUp(self):
        phone_norm = agente.normalizar_numero(self.PHONE)
        # Limpiar perfil y conversación previos del test.
        for d, suffix in (
            (agente.PERFILES_DIR, ".json"),
            (agente.CONVERSACIONES_DIR, ".json"),
        ):
            p = d / f"{phone_norm}{suffix}"
            p.unlink(missing_ok=True)

    def test_set_alias_crea_perfil_minimo(self):
        ok = agente._perfil_set_alias(self.PHONE, "Francisco Castillo")
        self.assertTrue(ok)
        perfil = agente._perfil_cliente(self.PHONE) or {}
        # Sin conversación, _perfil_cliente devuelve {} pero el alias
        # quedó en disco. Lo leemos directo.
        import json as _json
        phone_norm = agente.normalizar_numero(self.PHONE)
        p = agente.PERFILES_DIR / f"{phone_norm}.json"
        self.assertTrue(p.exists())
        data = _json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data.get("alias_admin"), "Francisco Castillo")

    def test_set_alias_sobrescribe(self):
        agente._perfil_set_alias(self.PHONE, "Original")
        agente._perfil_set_alias(self.PHONE, "Nuevo")
        import json as _json
        phone_norm = agente.normalizar_numero(self.PHONE)
        data = _json.loads(
            (agente.PERFILES_DIR / f"{phone_norm}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data["alias_admin"], "Nuevo")

    def test_quitar_alias(self):
        agente._perfil_set_alias(self.PHONE, "Pepe")
        habia = agente._perfil_quitar_alias(self.PHONE)
        self.assertTrue(habia)
        import json as _json
        phone_norm = agente.normalizar_numero(self.PHONE)
        data = _json.loads(
            (agente.PERFILES_DIR / f"{phone_norm}.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("alias_admin", data)

    def test_quitar_alias_inexistente(self):
        habia = agente._perfil_quitar_alias(self.PHONE)
        self.assertFalse(habia)

    def test_buscar_por_alias(self):
        agente._perfil_set_alias(self.PHONE, "Francisco Castillo")
        encontrado = agente._buscar_phone_por_alias_o_nombre("Francisco Castillo")
        self.assertEqual(encontrado, agente.normalizar_numero(self.PHONE))

    def test_buscar_por_alias_parcial_case_insensitive(self):
        agente._perfil_set_alias(self.PHONE, "Francisco Castillo")
        # contains + case-insensitive
        self.assertEqual(
            agente._buscar_phone_por_alias_o_nombre("francisco"),
            agente.normalizar_numero(self.PHONE),
        )
        self.assertEqual(
            agente._buscar_phone_por_alias_o_nombre("CASTILLO"),
            agente.normalizar_numero(self.PHONE),
        )

    def test_buscar_no_encuentra(self):
        self.assertIsNone(
            agente._buscar_phone_por_alias_o_nombre("Nadie Inexistente XYZ")
        )

    def test_buscar_query_vacia(self):
        self.assertIsNone(agente._buscar_phone_por_alias_o_nombre(""))
        self.assertIsNone(agente._buscar_phone_por_alias_o_nombre("   "))


class TestComandosEtiquetar(unittest.TestCase):
    """Regex y handlers de los nuevos comandos admin."""

    def test_regex_etiquetar_match(self):
        m = agente.CMD_ETIQUETAR_RE.search(
            "[CMD_ETIQUETAR: +5219991112233 | Francisco Castillo]"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "+5219991112233")
        self.assertEqual(m.group(2).strip(), "Francisco Castillo")

    def test_regex_etiquetar_alias_con_espacios(self):
        m = agente.CMD_ETIQUETAR_RE.search(
            "[CMD_ETIQUETAR:+529876543210|Ana María del Carmen]"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2).strip(), "Ana María del Carmen")

    def test_regex_quitar_etiqueta_match(self):
        m = agente.CMD_QUITAR_ETIQUETA_RE.search(
            "[CMD_QUITAR_ETIQUETA: +5219995554433]"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "+5219995554433")

    def test_sanitizer_atrapa_etiquetar_leak(self):
        leak = "Listo, voy a [CMD_ETIQUETAR: +52... | Pepe] al cliente."
        limpio = agente._sanitizar_salida(leak)
        self.assertNotIn("ETIQUETAR", limpio.upper())


if __name__ == "__main__":
    unittest.main(verbosity=2)
