# 🤖 Susan AI — Agente Corporativo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Deploy](https://img.shields.io/badge/Deploy-Streamlit%20Cloud-brightgreen)

Agente de inteligência artificial para responder perguntas de colaboradores com base em documentos internos da empresa. **Rodando 100% na nuvem** (Streamlit Cloud + Groq API) — sem infraestrutura própria.

---

## 🚀 Demo ao vivo

**[https://agente-corporativo-susanai.streamlit.app/](https://agente-corporativo-susan-ai.streamlit.app/)**

---

## Funcionalidades

- **Chat interativo** com histórico de conversa por sessão
- **Busca semântica** em todos os documentos indexados via embeddings + ChromaDB
- **Filtro por departamento** na sidebar (RH, Financeiro, Jurídico)
- **Citação de fontes** — cada resposta indica o arquivo de origem
- **Feedback** 👍/👎 por resposta, registrado em `feedback.log`
- **Curadoria de qualidade**: ignora rascunhos, backups; mantém versão oficial
- **Responsáveis por setor**: RH (Maria Oliveira), Financeiro (João Santos), Jurídico (Dra. Ana Costa)
- **Proteção contra alucinação**: LLM instruído a recusar assuntos fora do escopo corporativo
- **Streaming token-a-token** para UX fluida
- **Cache inteligente** (busca 5min, resposta LLM 1h) — perguntas repetidas são instantâneas

---

## Stack Atualizada (Cloud-First)

| Camada | Tecnologia | Onde roda |
|--------|-----------|-----------|
| Interface | Streamlit | Streamlit Cloud (grátis) |
| Orquestração | LangChain | Cloud |
| **LLM** | **Groq API** (`llama-3.1-8b-instant`) | **Groq Cloud (grátis, ~500ms)** |
| Embeddings | HuggingFace `paraphrase-MiniLM-L6-v2` | Cloud (CPU, 6 layers) |
| Banco vetorial | ChromaDB | Memória (reindexa no startup <5s) |
| Extração docs | PyMuPDF, python-docx, openpyxl, python-pptx, pandas, BeautifulSoup | Local/Cloud |
| Logs | JSON Lines (`execucao.log`) | Cloud |
| Configuração | YAML (`config.yaml`) | Cloud |

---

## Pré-requisitos

- Python 3.10+
- Conta gratuita no [Groq Console](https://console.groq.com/keys) → pegue `GROQ_API_KEY`
- Pip

---

## Instalação Local

```bash
# Clonar o repositório
git clone https://github.com/LanielXD/Agente-Corporativo.git
cd Agente-Corporativo

# (Opcional) Ambiente virtual
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/macOS

# Instalar dependências
pip install -r requirements.txt

# Configurar segredo local (desenvolvimento)
echo 'GROQ_API_KEY = "gsk_sua_chave_aqui"' > .streamlit/secrets.toml
```

---

## Configuração (`config.yaml`)

```yaml
# LLM via Groq (grátis, rápido)
modelo_llm: "llama-3.1-8b-instant"
temperatura_llm: 0.1
max_tokens_llm: 4096

# Embeddings leves (6 layers, multilíngue PT/EN/ES)
modelo_embedding: "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Chunking otimizado
chunk_tamanho: 600
chunk_sobreposicao: 60

# Busca enxuta
qtd_documentos: 3
usar_reranker: false

# Responsáveis por setor
responsaveis:
  rh: "Maria Oliveira"
  financeiro: "João Santos"
  juridico: "Dra. Ana Costa"
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

Execute a ingestão:

```bash
python ingestao.py
```

> No Streamlit Cloud, a indexação roda automaticamente no **startup** (18 chunks, ~5s).

### 2. Iniciar o chat

```bash
python run.py
# ou
streamlit run app.py
```

Acesse: `http://localhost:8501`

---

## Deploy no Streamlit Cloud (Grátis)

1. **Fork** este repo no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Conecte seu GitHub → selecione o repo → branch `main` → arquivo `app.py`
4. **Advanced settings** → **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
   ```
5. **Deploy!** → em ~2 min estará no ar em `https://seu-app.streamlit.app`

> **Zero infra**: sem Docker, sem VM, sem Ollama, sem GPU. O ChromaDB reindexa em memória a cada cold start (documentos são poucos: 9 arquivos, 18 chunks).

---

## Estrutura do Projeto

```
agente-corporativo/
├── app.py              # Interface Streamlit (chat + sidebar + cache)
├── ingestao.py         # Pipeline de extração e indexação
├── logger.py           # Log estruturado em JSON Lines
├── run.py              # Inicializador (valida Python 3.10+)
├── config.yaml         # Configurações centralizadas
├── config.yaml.example # Template de configuração
├── test_alucinacao.py  # Testes anti-alucinação (3 cenários)
├── test_extracao.py    # Testes unitários de extração
├── requirements.txt    # Dependências
├── .gitignore
└── README.md
```

Pastas geradas automaticamente (não versionadas):

```
chroma_db/              # Índice vetorial (recriado no startup no Cloud)
documentos/             # Documentos fonte por departamento
execucao.log            # Logs JSON Lines
feedback.log            # Feedbacks dos usuários
.streamlit/secrets.toml # Segredos locais (dev)
```

---

## Testes

```bash
python test_alucinacao.py  # Verifica regras anti-alucinação nos prompts
python test_extracao.py    # Valida limpeza de texto (paginação, confidencial)
```

---

## Performance (Cloud)

| Métrica | Valor típico |
|---------|--------------|
| Cold start | ~5s (carregar embeddings + indexar 18 chunks) |
| Busca vetorial (cache hit) | **~0ms** (5 min TTL) |
| Busca vetorial (cache miss) | ~15ms |
| LLM response (Groq) | **~500ms** |
| Resposta repetida (LLM cache) | **~0ms** (1h TTL) |

---

## Licença

MIT License — veja [LICENSE](LICENSE).

---

## Créditos

Desenvolvido como parte do desafio **Alura Agentes — ONE IA FOR TECH**.

**Deploy**: Streamlit Cloud + Groq API (Free Tier)  
**Autor**: [LanielXD](https://github.com/LanielXD)
