# ──────────────────────────────────────────────
# APP — Streamlit Chatbot com Pinecone + Groq
# ──────────────────────────────────────────────

"""
Susan AI - Chatbot Corporativo
Arquitetura: Streamlit + Pinecone + Groq + LangChain
"""

import streamlit as st
import warnings
import os

# Configuração deve vir ANTES de outros imports que usam config
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*torchvision.*")

import streamlit as st

# Imports da nova arquitetura
from config import get_config
from embeddings.provider import get_embedding_provider
from rag.chain import create_rag_chain
from rag.retriever import create_retriever
from vectorstore.pinecone_client import pinecone_client

# ──────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Susan AI - Chatbot Corporativo",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# ESTADO DA SESSÃO
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "filtro_dept" not in st.session_state:
    st.session_state.filtro_dept = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# ──────────────────────────────────────────────
# FUNÇÕES DE INICIALIZAÇÃO (cached)
# ──────────────────────────────────────────────

@st.cache_resource
def carregar_embeddings():
    """Carrega provider de embeddings (API + fallback local)."""
    try:
        return get_embedding_provider()
    except Exception as e:
        st.error(f"❌ Erro ao carregar embeddings: {e}")
        st.stop()


@st.cache_resource
def inicializar_rag_chain(_embeddings):
    """Inicializa a chain RAG completa."""
    try:
        return create_rag_chain(
            embeddings=_embeddings,
        )
    except Exception as e:
        st.error(f"❌ Erro ao inicializar RAG: {e}")
        st.stop()


@st.cache_data(ttl=30)
def verificar_groq():
    """Health check da Groq API."""
    try:
        import requests
        config = get_config()
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.groq.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.groq.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=300)
def listar_metadados() -> list:
    """Lista metadados disponíveis no Pinecone para sidebar."""
    try:
        stats = pinecone_client.get_stats()
        namespaces = stats.get("namespaces", {})
        metadados = []
        for ns, info in namespaces.items():
            if info.get("vector_count", 0) > 0:
                metadados.append({
                    "namespace": ns,
                    "count": info.get("vector_count", 0),
                })
        return metadados
    except Exception:
        return []


# ──────────────────────────────────────────────
# CARREGAMENTO INICIAL
# ──────────────────────────────────────────────

# Carrega embeddings
if st.session_state.embeddings is None:
    with st.spinner("📥 Carregando modelo de embeddings..."):
        st.session_state.embeddings = carregar_embeddings()

# Inicializa RAG Chain
if st.session_state.rag_chain is None:
    with st.spinner("🔗 Inicializando Susan AI..."):
        st.session_state.rag_chain = inicializar_rag_chain(st.session_state.embeddings)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    # Status da Groq API
    groq_ok = verificar_groq()
    status_groq = "🟢 Groq API OK" if groq_ok else "🔴 Groq API offline"
    st.caption(status_groq)

    # Info do usuário
    st.markdown(
        f"<div style='background:#757575;color:white;"
        f"padding:12px;border-radius:10px;text-align:center;margin:8px 0'>"
        f"<div style='font-size:28px'>💼</div>"
        f"<div style='font-weight:600'>Colaborador</div>"
        f"<div style='font-size:13px;opacity:.9'>Acesso a todos os documentos</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Filtro por departamento
    config = get_config()
    depts = list(config.get_responsaveis_dict().keys())
    if depts:
        st.session_state.filtro_dept = st.multiselect(
            "📂 Filtrar por departamento",
            depts,
            default=depts,
        )
    else:
        st.session_state.filtro_dept = None
        st.caption("📂 Nenhum departamento configurado")

    # Stats do Pinecone
    st.divider()
    st.caption("📊 Estatísticas do Banco")
    try:
        stats = pinecone_client.get_stats()
        st.caption(f"Total de vetores: {stats.get('total_vectors', 0)}")
        st.caption(f"Dimensão: {stats.get('dimension', 'N/A')}")
        st.caption(f"Uso: {stats.get('index_fullness', 0):.1%}")
    except Exception:
        st.caption("Não foi possível obter stats")

    st.divider()
    st.caption("📁 Documentos por departamento")
    for dept in depts:
        st.caption(f"  - {dept.title()}")

    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Tecnologias:**\n"
        "Streamlit  \n"
        "LangChain  \n"
        "Pinecone  \n"
        "Groq API  \n"
        "HuggingFace"
    )

# ──────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────

st.title("🤖 Susan AI")
st.markdown("Sua assistente corporativa para consultas em documentos internos")

# ──────────────────────────────────────────────
# CHAT
# ──────────────────────────────────────────────

# Exibe histórico
for i, msg in enumerate(st.session_state.messages):
    avatar = "🙋" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        # Exibe fontes se houver
        if msg.get("sources"):
            with st.expander("📎 Documentos consultados"):
                for src in msg["sources"]:
                    st.caption(f"📄 **{src.get('fonte', 'desconhecido')}** — {src.get('departamento', 'desconhecido')}")

# Input do usuário
pergunta = st.chat_input("Faça uma pergunta sobre os documentos...")

# ──────────────────────────────────────────────
# PROCESSAMENTO DA PERGUNTA
# ──────────────────────────────────────────────

if pergunta:
    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(pergunta)

    # Gera resposta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Buscando documentos..."):
            # Prepara filtro por departamento
            filtro = None
            if st.session_state.filtro_dept:
                filtro = {"departamento": {"$in": st.session_state.filtro_dept}}

            try:
                # Busca documentos
                retriever = create_retriever(
                    embeddings=st.session_state.embeddings,
                    filter=filtro,
                )
                resultado = retriever.search(pergunta)

                if resultado.documents:
                    st.write(f"📚 Encontrados {len(resultado.documents)} documentos relevantes")
                else:
                    st.warning("⚠️ Nenhum documento encontrado para esta pergunta")

            except Exception as e:
                st.error(f"Erro na busca: {e}")
                resultado = None

        # Gera resposta via RAG Chain
        with st.spinner("🤖 Gerando resposta..."):
            try:
                if resultado and resultado.documents:
                    # Usa RAG Chain com documentos
                    resposta_obj = st.session_state.rag_chain.invoke(pergunta)
                    resposta = resposta_obj.answer
                    fontes = resposta_obj.sources
                else:
                    # Resposta sem documentos (conhecimento geral)
                    resposta = st.session_state.rag_chain.invoke(pergunta).answer
                    fontes = []

                # Streaming da resposta
                resposta_placeholder = st.empty()
                resposta_completa = ""
                for chunk in resposta.split():
                    resposta_completa += chunk + " "
                    resposta_placeholder.markdown(resposta_completa + "▌")

                resposta_placeholder.markdown(resposta_completa)
                resposta = resposta_completa

            except Exception as e:
                st.error(f"Erro ao gerar resposta: {e}")
                resposta = f"Erro ao gerar resposta: {e}"
                fontes = []

        # Exibe fontes se houver
        if fontes:
            with st.expander("📎 Documentos consultados"):
                for src in fontes:
                    st.caption(f"📄 **{src.get('fonte', 'desconhecido')}** — {src.get('departamento', 'desconhecido')}")
        else:
            st.caption("ℹ️ Nenhum documento específico foi usado.")

    # Adiciona ao histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta,
        "sources": fontes,
    })

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Susan AI • Powered by Streamlit + LangChain + Pinecone + Groq"
)