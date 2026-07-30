# ──────────────────────────────────────────────
# IMPORTAÇÕES
# ──────────────────────────────────────────────

import streamlit as st
import yaml
import requests
import os
from pathlib import Path

import logger
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ──────────────────────────────────────────────
# CONFIGURAÇÃO INICIAL
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

try:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Arquivo config.yaml não encontrado.")
    st.stop()
except yaml.YAMLError as e:
    st.error(f"Erro ao ler config.yaml: {e}")
    st.stop()

if config is None:
    st.error("config.yaml está vazio.")
    st.stop()

CHAVES_OBRIGATORIAS = ["modelo_llm", "modelo_embedding"]
for chave in CHAVES_OBRIGATORIAS:
    if chave not in config:
        st.error(f"Configuração inválida: a chave '{chave}' não foi encontrada em config.yaml.")
        st.stop()
    if not isinstance(config[chave], str) or not config[chave].strip():
        st.error(f"Configuração inválida: a chave '{chave}' está vazia em config.yaml.")
        st.stop()

PASTA_CHROMA = BASE_DIR / "chroma_db"
PASTA_DOCUMENTOS = BASE_DIR / "documentos"
MODELO_LLM = config["modelo_llm"]
MODELO_EMBEDDING = config["modelo_embedding"]
QTD_DOCUMENTOS = config.get("qtd_documentos", 3)
TEMPERATURA_LLM = config.get("temperatura_llm", 0.1)
MAX_TOKENS_LLM = config.get("max_tokens_llm", 2048)
MODELO_RERANKER = config.get("modelo_reranker", "BAAI/bge-reranker-base")
USAR_RERANKER = config.get("usar_reranker", False)
RESPONSAVEIS = config.get("responsaveis", {})
_RESPONSAVEIS_STR = "\n".join(
    f"- {dept.upper()}: {nome}" for dept, nome in RESPONSAVEIS.items()
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY não configurada. Defina como variável de ambiente ou em .streamlit/secrets.toml")
    st.stop()

# Avatares do chat (estilo amigável/moderno)
AVATAR_USER = "🙋"
AVATAR_ASSISTANT = "💬"

st.set_page_config(
    page_title="Chatbot Corporativo",
    page_icon="🤖",
    layout="centered",
)

# ──────────────────────────────────────────────
# INICIALIZAÇÃO DO VECTORSTORE (com auto-indexação no Cloud)
# ──────────────────────────────────────────────

try:
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
except Exception as e:
    st.error(f"Erro ao carregar modelo de embedding '{MODELO_EMBEDDING}': {e}")
    st.stop()

@st.cache_resource
def obter_vectorstore():
    """Retorna ChromaDB; se não existir, indexa documentos na primeira execução."""
    if not PASTA_CHROMA.exists():
        with st.spinner("🔄 Indexando documentos na primeira execução..."):
            import ingestao
            if not ingestao.processar_documentos(log_fn=lambda *a, **k: None):
                raise RuntimeError("Falha ao indexar documentos. Verifique a pasta 'documentos/'.")
    try:
        return Chroma(
            persist_directory=str(PASTA_CHROMA),
            embedding_function=embeddings,
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao conectar ao banco de dados ChromaDB: {e}")

try:
    vectorstore = obter_vectorstore()
    if len(vectorstore.get(limit=1)["ids"]) == 0:
        st.warning("⚠️ O índice ChromaDB está vazio. Reindexando...")
        import ingestao
        if not ingestao.processar_documentos(log_fn=lambda *a, **k: None):
            st.error("Falha ao reindexar documentos.")
            st.stop()
        vectorstore = obter_vectorstore()
except RuntimeError as e:
    st.error(str(e))
    st.stop()
except Exception:
    st.warning("⚠️ Não foi possível verificar o índice ChromaDB. Tentando reindexar...")
    import ingestao
    if not ingestao.processar_documentos(log_fn=lambda *a, **k: None):
        st.error("Falha ao indexar documentos.")
        st.stop()
    vectorstore = obter_vectorstore()

logger.startup()

# ──────────────────────────────────────────────
# ESTADO DA SESSÃO
# ──────────────────────────────────────────────

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "filtro_dept" not in st.session_state:
    st.session_state.filtro_dept = None

# ──────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────

MAX_MENSAGENS_HISTORICO = 30


@st.cache_resource
def carregar_llm():
    return ChatGroq(
        model=MODELO_LLM,
        temperature=TEMPERATURA_LLM,
        max_tokens=MAX_TOKENS_LLM,
        groq_api_key=GROQ_API_KEY,
    )


import hashlib

@st.cache_data(ttl=3600, show_spinner=False)
def resposta_llm_cache(prompt_hash, prompt):
    """Cache de resposta LLM por 1 hora, keyed por hash do prompt."""
    llm = carregar_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


@st.cache_resource
def carregar_reranker():
    if not USAR_RERANKER:
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(MODELO_RERANKER)
    except Exception as e:
        st.warning(f"Não foi possível carregar o reranker ({MODELO_RERANKER}): {e}")
        return None


def _registrar_feedback(pergunta, resposta, avaliacao):
    import json
    from datetime import datetime
    try:
        feedback_path = BASE_DIR / "feedback.log"
        entrada = {
            "timestamp": datetime.now().isoformat(),
            "pergunta": pergunta,
            "resposta": resposta,
            "avaliacao": avaliacao,
        }
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        logger.feedback(avaliacao)
    except Exception:
        pass


@st.cache_data(ttl=300, show_spinner=False)
def buscar_docs(pergunta, departamentos_filtro=None):
    """Busca documentos com cache de 5 minutos."""
    vectorstore = obter_vectorstore()
    filtro = None
    if departamentos_filtro:
        filtro = {"departamento": {"$in": list(departamentos_filtro)}}
    k_busca = QTD_DOCUMENTOS * 2 if USAR_RERANKER else QTD_DOCUMENTOS
    docs = vectorstore.similarity_search(pergunta, k=k_busca, filter=filtro)
    if docs and USAR_RERANKER:
        reranker = carregar_reranker()
        if reranker:
            try:
                pairs = [[pergunta, doc.page_content] for doc in docs]
                scores = reranker.predict(pairs)
                scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                docs = [doc for doc, _ in scored[:QTD_DOCUMENTOS]]
            except Exception:
                docs = docs[:QTD_DOCUMENTOS]
        else:
            docs = docs[:QTD_DOCUMENTOS]
    return docs


def _deduplicar_fontes(fontes):
    unicas = []
    vistas = set()
    for f in fontes:
        chave = f.get("arquivo", f.get("fonte", str(f)))
        if chave not in vistas:
            vistas.add(chave)
            unicas.append(f)
    return unicas


def preparar_prompt(pergunta, vectorstore, departamentos_filtro=None):
    try:
        docs = buscar_docs(pergunta, departamentos_filtro)
    except Exception as e:
        logger.erro("busca_documentos", e)
        return f"Erro ao buscar documentos: {e}", []

    if not docs:
        prompt = (
            "Você é um assistente corporativo.\n\n"
            "Não foram encontrados documentos específicos para esta pergunta.\n\n"
            "REGRAS:\n"
            "- Se a pergunta for sobre os departamentos de **RH**, **Financeiro** ou **Jurídico** "
            "(suas funções, rotinas ou áreas de atuação), responda com seu conhecimento geral.\n"
            "- Se a pergunta for sobre amenidades, cultura geral, previsão do tempo ou qualquer "
            "assunto NÃO corporativo, informe educadamente que só pode ajudar com informações corporativas.\n"
            "- Caso contrário, informe que o assunto não foi encontrado na base de conhecimento.\n\n"
            f"RESPONSÁVEIS PELOS SETORES:\n{_RESPONSAVEIS_STR}\n\n"
            "REGRAS DE ALTERAÇÃO:\n"
            "- Se o colaborador pedir para ALTERAR ou EXCLUIR algum arquivo, NÃO faça nenhuma alteração. "
            "Informe que ele precisa de autorização do responsável pelo setor do arquivo e exiba o nome do responsável.\n"
            "- Se o colaborador pedir para ATUALIZAR algum arquivo, informe que você irá notificar "
            "o responsável pelo setor sobre a solicitação.\n\n"
            f"<pergunta>{pergunta}</pergunta>\n"
            "Responda de forma clara e direta em português."
        )
        logger.consulta(pergunta, 0, departamentos_filtro)
        return prompt, []

    contexto = []
    fontes = []
    for doc in docs:
        contexto.append(doc.page_content)
        fontes.append(doc.metadata)

    docs_com_fonte = []
    for i, (c, m) in enumerate(zip(contexto, fontes)):
        nome_arquivo = m.get("fonte", "desconhecido")
        nome_departamento = m.get("departamento", "desconhecido")
        docs_com_fonte.append(
            f'Documento [{i+1}] — "{nome_arquivo}" ({nome_departamento}):\n{c}'
        )

    prompt = (
        "Você é um assistente corporativo especializado em analisar documentos internos da empresa.\n\n"
        "REGRAS IMPORTANTES:\n"
        "- Responda com base PRIMEIRAMENTE nos documentos fornecidos abaixo. Se eles não cobrirem "
        "totalmente, complemente com seu conhecimento geral sobre RH, Financeiro e Jurídico.\n"
        "- Se a pergunta for sobre amenidades, cultura geral, previsão do tempo, ou qualquer assunto "
        "NÃO corporativo, responda educadamente que só pode ajudar com informações corporativas.\n"
        "- Se os documentos não contiverem informação suficiente para responder e o assunto não for "
        "sobre RH, Financeiro ou Jurídico, avise que o assunto não foi encontrado na base de conhecimento.\n"
        "- Se a pergunta for sobre um arquivo específico (ex: \"despesas.csv\", \"política de férias\"), "
        "informe o que contém naquele arquivo com base nos trechos disponíveis.\n\n"
        f"RESPONSÁVEIS PELOS SETORES:\n{_RESPONSAVEIS_STR}\n\n"
        "REGRAS DE ALTERAÇÃO:\n"
        "- Se o colaborador pedir para ALTERAR ou EXCLUIR algum arquivo, NÃO faça nenhuma alteração. "
        "Informe que ele precisa de autorização do responsável pelo setor do arquivo e exiba o nome do responsável.\n"
        "- Se o colaborador pedir para ATUALIZAR algum arquivo, informe que você irá notificar "
        "o responsável pelo setor sobre a solicitação.\n\n"
        "Documentos:\n"
        f"{chr(10).join(docs_com_fonte)}\n\n"
        f"<pergunta>{pergunta}</pergunta>\n\n"
        "Responda de forma clara e direta em português.\n"
        "Sempre cite o nome do arquivo entre aspas ao usar uma informação dele.\n"
        "Exemplo: 'Conforme \"Politica_de_Ferias.pdf\", as férias devem ser solicitadas "
        "com 30 dias de antecedência.'"
    )

    fontes_unicas = _deduplicar_fontes(fontes)
    logger.consulta(pergunta, len(fontes_unicas), [m.get("departamento", "?") for m in fontes_unicas])
    return prompt, fontes_unicas


def stream_resposta(prompt):
    try:
        # Hash do prompt para cache
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        resposta = resposta_llm_cache(prompt_hash, prompt)
        # Simula streaming token a token para UX
        for chunk in resposta.split():
            yield chunk + " "
    except Exception as e:
        logger.erro("llm_stream", e)
        yield f"Erro ao gerar resposta: {e}. Verifique a conexão com a Groq API."


def _poda_historico():
    if len(st.session_state.mensagens) > MAX_MENSAGENS_HISTORICO:
        st.session_state.mensagens = st.session_state.mensagens[-MAX_MENSAGENS_HISTORICO:]


@st.cache_data(ttl=30)
def verificar_groq():
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=5
        )
        return r.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def listar_metadados():
    try:
        metadados_raw = vectorstore.get()["metadatas"]
        resultado = [m for m in metadados_raw if m is not None]
        if resultado:
            return resultado
    except Exception:
        pass
    return None

with st.sidebar:
    groq_ok = verificar_groq()
    status_groq = "🟢 Groq API OK" if groq_ok else "🔴 Groq API offline"
    st.caption(status_groq)

    st.markdown(
        f"<div style='background:#757575;color:white;"
        f"padding:12px;border-radius:10px;text-align:center;margin:8px 0'>"
        f"<div style='font-size:28px'>💼</div>"
        f"<div style='font-weight:600'>Colaborador</div>"
        f"<div style='font-size:13px;opacity:.9'>Acesso a todos os documentos</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    metadados = listar_metadados() or []
    dept_set = set(m.get("departamento", "desconhecido") for m in metadados)
    dept_ordenados = sorted(dept_set)
    if dept_ordenados:
        st.session_state.filtro_dept = st.multiselect(
            "Filtrar por departamento",
            dept_ordenados,
            default=dept_ordenados,
        )
    else:
        st.session_state.filtro_dept = None
        st.caption("📂 Nenhum departamento disponível para filtro")

    st.divider()
    st.caption("📁 Documentos carregados:")
    fontes_para_mostrar = sorted(set(m.get("fonte", "desconhecido") for m in metadados))
    for nome_fonte in fontes_para_mostrar:
        st.caption(f"  - {nome_fonte}")
    if not fontes_para_mostrar:
        st.caption("  (Nenhum documento disponível)")

    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        for key in list(st.session_state.keys()):
            if key.startswith("fb_"):
                del st.session_state[key]

    st.divider()
    st.markdown("**Tecnologias:**  \nStreamlit  \nLangChain  \nChromaDB  \nGroq API  \nHuggingFace")

# ──────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────

st.title("🤖 Susan AI")
st.markdown("Sua AI corporativa para lhe ajudar no dia a dia")

# ──────────────────────────────────────────────
# CHAT
# ──────────────────────────────────────────────

for i, mensagem in enumerate(st.session_state.mensagens):
    papel = mensagem.get("papel", "user")
    avatar = AVATAR_USER if papel == "user" else AVATAR_ASSISTANT
    with st.chat_message(papel, avatar=avatar):
        st.markdown(mensagem.get("conteudo", ""))
        if "fontes" in mensagem and mensagem["fontes"]:
            with st.expander("📎 Documentos consultados"):
                for f in mensagem["fontes"]:
                    nome_fonte = f.get("fonte", "desconhecido")
                    dept_fonte = f.get("departamento", "desconhecido")
                    st.caption(f"📄 **{nome_fonte}** — {dept_fonte}")
    if mensagem.get("papel") == "assistant" and mensagem.get("conteudo"):
        fb_key = f"fb_{i}"
        if fb_key not in st.session_state:
            st.session_state[fb_key] = None
        st.caption("👍 Útil · 👎 Não útil")
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"like_{i}"):
                _registrar_feedback(
                    mensagem.get("pergunta", ""),
                    mensagem["conteudo"],
                    "positivo",
                )
                st.session_state[fb_key] = "positivo"
        with col2:
            if st.button("👎", key=f"dislike_{i}"):
                _registrar_feedback(
                    mensagem.get("pergunta", ""),
                    mensagem["conteudo"],
                    "negativo",
                )
                st.session_state[fb_key] = "negativo"
        if st.session_state[fb_key] == "positivo":
            st.caption("✅ Feedback registrado — obrigado!")
        elif st.session_state[fb_key] == "negativo":
            st.caption("✅ Feedback registrado — sua opinião nos ajuda a melhorar!")

pergunta = st.chat_input("Faça uma pergunta sobre os documentos...")

# ──────────────────────────────────────────────
# PROCESSAMENTO DA PERGUNTA
# ──────────────────────────────────────────────

if pergunta:
    st.session_state.mensagens.append({"papel": "user", "conteudo": pergunta})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("Consultando documentos..."):
            prompt, fontes = preparar_prompt(
                pergunta, st.session_state.filtro_dept
            )
        if fontes is not None:
            resposta = st.write_stream(stream_resposta(prompt))
            if fontes:
                with st.expander("📎 Documentos consultados"):
                    for f in fontes:
                        dept = f.get("departamento", "desconhecido")
                        nome_fonte = f.get("fonte", "desconhecido")
                        st.caption(f"📄 **{nome_fonte}** — {dept}")
            else:
                st.caption("ℹ️ Nenhum documento específico foi usado.")
        else:
            st.markdown(prompt)
            resposta = prompt
            fontes = []

    st.session_state.mensagens.append({
        "papel": "assistant",
        "conteudo": resposta,
        "fontes": fontes,
        "pergunta": pergunta,
    })

    _poda_historico()