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


if __name__ == "__main__":
    unittest.main(verbosity=2)
