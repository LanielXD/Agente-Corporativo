"""Testes estruturais de propensão a alucinação no prompt real da RAG Chain."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.chain import RAGChain


def obter_prompt_system():
    """Extrai o texto do system prompt real criado pela RAGChain."""
    template = RAGChain._create_prompt(None)
    return template.partial_variables.get("system", "")


def test_prompt_permite_rh_financeiro_juridico():
    """Cenário 1: pergunta sobre RH sem documentos — deve permitir resposta"""
    prompt = obter_prompt_system()
    erros = []
    for dept in ["RH", "FINANCEIRO", "JURIDICO"]:
        if f"{dept}:" not in prompt:
            erros.append(f"{dept} ausente dos responsáveis")
    if "conhecimento geral" not in prompt:
        erros.append("'conhecimento geral' ausente")
    if "RESPONSAVEIS PELOS SETORES" not in prompt:
        erros.append("responsáveis ausentes")
    if "ALTERAR ou EXCLUIR" not in prompt:
        erros.append("regra de alteração/exclusão ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] RH/Financeiro/Juridico autorizados sem docs")


def test_prompt_recusa_fora_escopo():
    """Cenário 2: pergunta não corporativa — deve recusar"""
    prompt = obter_prompt_system()
    erros = []
    if "NAO corporativo" not in prompt and "NÃO corporativo" not in prompt:
        erros.append("regra de recusa 'NAO corporativo' ausente")
    if "amenidades" not in prompt:
        erros.append("regra 'amenidades' ausente")
    if "cultura geral" not in prompt:
        erros.append("regra 'cultura geral' ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] Assuntos nao corporativos sao recusados")


def test_prompt_prioriza_documentos():
    """Cenário 3: com documentos disponíveis — deve priorizá-los"""
    prompt = obter_prompt_system()
    erros = []
    if "PRIMEIRAMENTE" not in prompt:
        erros.append("regra 'PRIMEIRAMENTE' ausente")
    if "Sempre cite o nome do arquivo" not in prompt:
        erros.append("instrução 'Sempre cite' ausente")
    if "nao foi encontrado na base de conhecimento" not in prompt:
        erros.append("aviso de base de conhecimento ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] Documentos sao priorizados no prompt")


def test_prompt_tem_contexto_e_pergunta():
    """Cenário 4: template deve ter placeholders de contexto e pergunta"""
    template = RAGChain._create_prompt(None)
    vars_template = template.input_variables
    erros = []
    if "context" not in vars_template:
        erros.append("placeholder {context} ausente")
    if "question" not in vars_template:
        erros.append("placeholder {question} ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] Placeholders {context} e {question} presentes")


if __name__ == "__main__":
    print("Testes de propensão a alucinação:\n")
    test_prompt_permite_rh_financeiro_juridico()
    test_prompt_recusa_fora_escopo()
    test_prompt_prioriza_documentos()
    test_prompt_tem_contexto_e_pergunta()
    print("\n[OK] Todos os 4 testes passaram -- regras anti-alucinacao presentes no prompt.")
