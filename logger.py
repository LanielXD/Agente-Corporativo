# ──────────────────────────────────────────────
# LOGGER — registro estruturado em JSON Lines
# Eventos: startup, consulta, erro, ingestao, feedback
# ──────────────────────────────────────────────

from pathlib import Path
import json
from datetime import datetime

LOG_PATH = Path(__file__).parent / "execucao.log"


def _registrar(evento, detalhes=None):
    try:
        entrada = {
            "timestamp": datetime.now().isoformat(),
            "evento": evento,
            "detalhes": detalhes or {},
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception:
        pass


def startup():
    _registrar("startup")


def consulta(pergunta, qtd_docs, departamentos=None):
    _registrar("consulta", {
        "pergunta": pergunta,
        "qtd_documentos_retornados": qtd_docs,
        "departamentos": departamentos or [],
    })


def erro(origem, mensagem):
    _registrar("erro", {"origem": origem, "mensagem": str(mensagem)})


def ingestao(arquivos, chunks, sucesso):
    _registrar("ingestao", {
        "arquivos_processados": arquivos,
        "chunks_gerados": chunks,
        "sucesso": sucesso,
    })


def feedback(avaliacao):
    _registrar("feedback", {"avaliacao": avaliacao})
