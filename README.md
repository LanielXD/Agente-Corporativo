# 🤖 Susan AI — Agente Corporativo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Deploy](https://img.shields.io/badge/Deploy-Streamlit%20Cloud-brightgreen)

Agente de inteligência artificial para responder perguntas de colaboradores com base em documentos internos da empresa. **Rodando 100% na nuvem** (Streamlit Cloud + Groq + Pinecone) — sem infraestrutura própria.

---

## Funcionalidades

- **Chat interativo** com histórico de conversa por sessão
- **Busca semântica** em todos os documentos indexados via embeddings + Pinecone
- **Filtro por departamento** na sidebar (RH, Financeiro, Jurídico)
- **Citação de fontes** — cada resposta indica o arquivo de origem
- **Curadoria de qualidade**: ignora rascunhos, backups; mantém versão oficial
- **Responsáveis por setor**: RH (Maria Oliveira), Financeiro (João Santos), Jurídico (Dra. Ana Costa)
- **Proteção contra alucinação**: LLM instruído a recusar assuntos fora do escopo corporativo
- **Streaming token-a-token** para UX fluida

---

## Stack (Cloud-First)

| Camada | Tecnologia | Onde roda |
|--------|-----------|-----------|
| Interface | Streamlit | Streamlit Cloud (grátis) |
| Orquestração | LangChain (LCEL) | Cloud |
| **LLM** | **Groq API** (`llama-3.1-8b-instant`) | **Groq Cloud (grátis, ~500ms)** |
| Embeddings | `sentence-transformers/paraphrase-MiniLM-L6-v2` | API HF com fallback local |
| Banco vetorial | **Pinecone** (serverless, grátis) | Pinecone Cloud |
| Extração docs | PyMuPDF, python-docx, openpyxl, python-pptx, pandas, BeautifulSoup | Local/Cloud |
| Configuração | `config.py` (Pydantic Settings) + `.env` / `st.secrets` | Local/Cloud |

---
## Demo do projeto
https://agente-corporativo-susan-ai.streamlit.app/
---
## Pré-requisitos

- Python 3.10+ (recomendado 3.13)
- Conta gratuita no [Groq Console](https://console.groq.com/keys) → pegue `GROQ_API_KEY`
- Conta gratuita no [Pinecone](https://www.pinecone.io/) → pegue `PINECONE_API_KEY`

---

## Instalação Local

```bash
# Clonar o repositório
git clone https://github.com/LanielXD/Agente-Corporativo.git
cd Agente-Corporativo

# (Recomendado) Ambiente virtual com Python 3.13
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/macOS

# Instalar dependências
pip install -r requirements.txt

# Configurar segredos (desenvolvimento)
# Crie .streamlit/secrets.toml com:
#   GROQ_API_KEY = "gsk_sua_chave_aqui"
#   PINECONE_API_KEY = "pcsk_sua_chave_aqui"
```

Para scripts standalone (`index.py`), crie também um arquivo `.env` na raiz:

```bash
GROQ__API_KEY=gsk_sua_chave_aqui
PINECONE__API_KEY=pcsk_sua_chave_aqui
EMBEDDING__PROVIDER=local
```

---

## Uso

### 1. Indexar documentos

Organize os arquivos por departamento:

```
documentos/
  rh/
    Politica_de_Ferias.pdf
  financeiro/
    Despesas.xlsx
  juridico/
    Contratos.docx
```

Execute a indexação (cria o índice no Pinecone se não existir):

```bash
python index.py
```

### 2. Iniciar o chat

```bash
streamlit run app.py
```

Acesse: `http://localhost:8501`

---

## Deploy no Streamlit Cloud (Grátis)

1. **Push** deste repo no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Conecte seu GitHub → selecione o repo → branch `main` → arquivo `app.py`
4. **Advanced settings** → **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
   PINECONE_API_KEY = "pcsk_xxxxxxxxxxxxxxxxxxxxxxxx"
   PINECONE_INDEX = "susan-ai"
   PINECONE_REGION = "us-east-1"
   ```
5. **Deploy!** → em ~2 min estará no ar em `https://seu-app.streamlit.app`

---

## Estrutura do Projeto

```
agente-corporativo/
├── app.py                  # Interface Streamlit (chat + sidebar + cache)
├── config.py               # Configuração centralizada (Pydantic Settings)
├── index.py                # Script de indexação dos documentos no Pinecone
├── embeddings/
│   └── provider.py         # Provider de embeddings (API HF/OpenAI + fallback local)
├── vectorstore/
│   └── pinecone_client.py  # Cliente Pinecone (índice, upsert, busca, stats)
├── rag/
│   ├── retriever.py        # Retriever com filtros e reranker opcional
│   └── chain.py            # Pipeline RAG completo (LCEL + Groq)
├── documentos/             # Documentos fonte por departamento (rh, financeiro, juridico)
├── test_alucinacao.py      # Testes anti-alucinação do prompt real
├── test_extracao.py        # Testes unitários de extração
├── requirements.txt        # Dependências
├── .gitignore
└── README.md
```

Pastas não versionadas (locais):

```
venv/                       # Ambiente virtual
.streamlit/secrets.toml     # Segredos locais (dev)
.env                        # Variáveis de ambiente para scripts standalone
```

---

## Testes

```bash
python test_alucinacao.py  # Verifica regras anti-alucinação no prompt real
python test_extracao.py    # Valida limpeza de texto (paginação, confidencial)
```

---

## Performance (Cloud)

| Métrica | Valor típico |
|---------|--------------|
| Cold start | ~5s (carregar embeddings) |
| Busca vetorial (Pinecone) | ~50ms |
| LLM response (Groq) | **~500ms** |

---

## Licença

MIT License — veja [LICENSE](LICENSE).

---

## Créditos

Desenvolvido como parte do desafio **Alura Agentes — ONE IA FOR TECH**.

**Deploy**: Streamlit Cloud + Groq API + Pinecone (Free Tiers)  
**Autor**: [LanielXD](https://github.com/LanielXD)
