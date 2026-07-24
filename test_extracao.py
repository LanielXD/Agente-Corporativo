"""Testes unitarios."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ingestao import _limpar_texto


def test_limpar_texto_remove_paginacao():
    sujo = "Texto normal.\n1 / 5\npágina 3\npág. 10\n- 7 -\nMais texto."
    limpo = _limpar_texto(sujo)
    assert "1 / 5" not in limpo
    assert "página 3" not in limpo
    assert "pág. 10" not in limpo
    assert "- 7 -" not in limpo
    assert "Texto normal" in limpo
    assert "Mais texto" in limpo


def test_limpar_texto_remove_confidencial():
    sujo = "confidencial\nDocumento interno\nTexto real"
    limpo = _limpar_texto(sujo)
    assert "confidencial" not in limpo


def test_ok():
    assert True