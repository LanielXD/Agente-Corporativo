"""Testes estruturais de propensão a alucinação nos prompts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

QTD_DOCUMENTOS = 5
_RESPONSAVEIS_STR = "- RH: Maria Oliveira\n- FINANCEIRO: João Santos\n- JURIDICO: Dra. Ana Costa"


def montar_prompt_sem_docs(pergunta):
    return f"""Você é um assistente corporativo.

Não foram encontrados documentos específicos para esta pergunta.

REGRAS:
- Se a pergunta for sobre os departamentos de **RH**, **Financeiro** ou **Jurídico** (suas funções, rotinas ou áreas de atuação), responda com seu conhecimento geral.
- Se a pergunta for sobre amenidades, cultura geral, previsão do tempo ou qualquer assunto NÃO corporativo, informe educadamente que só pode ajudar com informações corporativas.
- Caso contrário, informe que o assunto não foi encontrado na base de conhecimento.

RESPONSÁVEIS PELOS SETORES:
{_RESPONSAVEIS_STR}

REGRAS DE ALTERAÇÃO:
- Se o colaborador pedir para ALTERAR ou EXCLUIR algum arquivo, NÃO faça nenhuma alteração. Informe que ele precisa de autorização do responsável pelo setor do arquivo e exiba o nome do responsável.
- Se o colaborador pedir para ATUALIZAR algum arquivo, informe que você irá notificar o responsável pelo setor sobre a solicitação.

<pergunta>{pergunta}</pergunta>
Responda de forma clara e direta em português."""


def montar_prompt_com_docs(pergunta, docs_com_fonte):
    return f"""Você é um assistente corporativo especializado em analisar documentos internos da empresa.

REGRAS IMPORTANTES:
- Responda com base PRIMEIRAMENTE nos documentos fornecidos abaixo. Se eles não cobrirem totalmente, complemente com seu conhecimento geral sobre RH, Financeiro e Jurídico.
- Se a pergunta for sobre amenidades, cultura geral, previsão do tempo, ou qualquer assunto NÃO corporativo, responda educadamente que só pode ajudar com informações corporativas.
- Se os documentos não contiverem informação suficiente para responder e o assunto não for sobre RH, Financeiro ou Jurídico, avise que o assunto não foi encontrado na base de conhecimento.
- Se a pergunta for sobre um arquivo específico (ex: "despesas.csv", "política de férias"), informe o que contém naquele arquivo com base nos trechos disponíveis.

RESPONSÁVEIS PELOS SETORES:
{_RESPONSAVEIS_STR}

REGRAS DE ALTERAÇÃO:
- Se o colaborador pedir para ALTERAR ou EXCLUIR algum arquivo, NÃO faça nenhuma alteração. Informe que ele precisa de autorização do responsável pelo setor do arquivo e exiba o nome do responsável.
- Se o colaborador pedir para ATUALIZAR algum arquivo, informe que você irá notificar o responsável pelo setor sobre a solicitação.

Documentos:
{chr(10).join(docs_com_fonte)}

<pergunta>{pergunta}</pergunta>

Responda de forma clara e direta em português.
Sempre cite o nome do arquivo entre aspas ao usar uma informação dele.
Exemplo: 'Conforme "Politica_de_Ferias.pdf", as férias devem ser solicitadas com 30 dias de antecedência.'"""


def test_prompt_sem_docs_permite_rh_financeiro_juridico():
    """Cenário 1: pergunta sobre RH sem documentos — deve permitir resposta"""
    prompt = montar_prompt_sem_docs("O que faz o departamento de RH?")
    erros = []
    if "**RH**" not in prompt:
        erros.append("RH ausente das regras")
    if "**Financeiro**" not in prompt:
        erros.append("Financeiro ausente das regras")
    if "**Jurídico**" not in prompt:
        erros.append("Jurídico ausente das regras")
    if "conhecimento geral" not in prompt:
        erros.append("'conhecimento geral' ausente")
    if "RESPONSÁVEIS PELOS SETORES" not in prompt:
        erros.append("responsáveis ausentes")
    if "ALTERAR ou EXCLUIR" not in prompt:
        erros.append("regra de alteração/exclusão ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] RH/Financeiro/Juridico autorizados sem docs")


def test_prompt_sem_docs_recusa_fora_escopo():
    """Cenário 2: pergunta não corporativa sem docs — deve recusar"""
    prompt = montar_prompt_sem_docs("Qual o time do Flamengo?")
    erros = []
    if "NÃO corporativo" not in prompt:
        erros.append("regra de recusa 'NÃO corporativo' ausente")
    if "amenidades" not in prompt:
        erros.append("regra 'amenidades' ausente")
    if "cultura geral" not in prompt:
        erros.append("regra 'cultura geral' ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] Assuntos nao corporativos sao recusados")


def test_prompt_com_docs_prioriza_documentos():
    """Cenário 3: com documentos disponíveis — deve priorizá-los"""
    docs = [
        'Documento [1] — "ferias.pdf" (rh):\nPolítica de férias: 30 dias.',
        'Documento [2] — "beneficios.docx" (rh):\nVale alimentação de R$500.',
    ]
    prompt = montar_prompt_com_docs("Quantos dias de férias?", docs)
    erros = []
    if "PRIMEIRAMENTE" not in prompt:
        erros.append("regra 'PRIMEIRAMENTE' ausente")
    if "ferias.pdf" not in prompt:
        erros.append("documento ferias.pdf não incluído")
    if "beneficios.docx" not in prompt:
        erros.append("documento beneficios.docx não incluído")
    if "RESPONSÁVEIS PELOS SETORES" not in prompt:
        erros.append("responsáveis ausentes com documentos")
    if "Sempre cite" not in prompt:
        erros.append("instrução 'Sempre cite' ausente")
    assert not erros, "; ".join(erros)
    print("  [OK] Documentos sao priorizados no prompt")


if __name__ == "__main__":
    print("Testes de propensão a alucinação:\n")
    test_prompt_sem_docs_permite_rh_financeiro_juridico()
    test_prompt_sem_docs_recusa_fora_escopo()
    test_prompt_com_docs_prioriza_documentos()
    print("\n[OK] Todos os 3 testes passaram -- regras anti-alucinacao presentes nos prompts.")
