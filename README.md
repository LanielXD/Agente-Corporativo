# 🤖 Susan AI — Agente Corporativo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

Agente de inteligência artificial para responder perguntas de colaboradores com base em documentos internos da empresa. Suporta múltiplos formatos (PDF, Word, Excel, PowerPoint, Markdown, CSV, JSON, HTML) e domínios organizacionais (RH, Financeiro, Jurídico).

## Funcionalidades

- **Chat interativo** com histórico de conversa por sessão
- **Busca semântica** em todos os documentos indexados via embeddings + ChromaDB
- **Reranker** para refinar resultados por relevância (CrossEncoder)
- **Filtro por departamento** na sidebar
- **Citação de fontes** — cada resposta indica o arquivo de origem
- **Feedback** 👍/👎 por resposta, registrado em `feedback.log`
- **Curadoria de qualidade**: ignora rascunhos, backups; mantém versão oficial
- **Responsáveis por setor**: RH (Maria Oliveira), Financeiro (João Santos), Jurídico (Dra. Ana Costa)
- **Proteção contra alucinação**: LLM instruído a recusar assuntos fora do escopo corporativo

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Interface | Streamlit |
| Orquestração | LangChain |
| LLM | Ollama (modelo local) |
| Embeddings | HuggingFace + sentence-transformers |
| Banco vetorial | ChromaDB |
| Extração de documentos | PyMuPDF, python-docx, openpyxl, python-pptx, pandas, BeautifulSoup |
| Logs | JSON Lines (`execucao.log`) |
| Configuração | YAML (`config.yaml`) |

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.ai) instalado e rodando com um modelo (ex.: `ollama pull llama3`)
- Pip

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/agente-corporativo.git
cd agente-corporativo

# (Opcional) Criar ambiente virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
```

## Configuração

Edite `config.yaml` para ajustar:

```yaml
modelo_llm: "llama3.1:8b"
modelo_embedding: "paraphrase-multilingual-MiniLM-L12-v2"
temperatura_llm: 0.1
chunk_tamanho: 500
chunk_sobreposicao: 50
qtd_documentos: 5
usar_reranker: true
modelo_reranker: "BAAI/bge-reranker-base"

responsaveis:
  rh: "Maria Oliveira"
  financeiro: "João Santos"
  juridico: "Dra. Ana Costa"

curadoria:
  ignorar_se_conter:
    - "rascunho"
    - "draft"
    - "old"
    - "test"
    - "tmp"
    - "copia"
    - "backup"
  ignorar_exatos:
    - "notas.md"
    - "README.md"
    - "pessoal.txt"
    - "anotacoes.txt"
  manter_versao_oficial: true
```

## Uso

### 1. Indexar documentos

Coloque os documentos nas pastas apropriadas:

```
documentos/
  rh/
    Politica_de_Ferias.pdf
  financeiro/
    Despesas.xlsx
  juridico/
    Contratos.docx
```

Depois execute a ingestão:

```bash
python ingestao.py
```

### 2. Iniciar o chat

```bash
python run.py
```

Ou diretamente:

```bash
streamlit run app.py
```

## Estrutura do projeto

```
agente-corporativo/
├── app.py              # Interface Streamlit (chat + sidebar)
├── ingestao.py         # Pipeline de extração e indexação
├── logger.py           # Log estruturado em JSON Lines
├── run.py              # Inicializador do Streamlit
├── config.yaml         # Configurações centralizadas
├── test_alucinacao.py  # Testes estruturais de prompts
├── test_extracao.py    # Testes unitários de extração
├── requirements.txt    # Dependências
├── .gitignore
└── README.md
```

Pastas geradas automaticamente:

```
chroma_db/              # Índice vetorial (embeddings)
documentos/             # Documentos fonte organizados por departamento
execucao.log            # Logs de execução em JSON Lines
feedback.log            # Feedbacks dos usuários
```

## Testes

```bash
python test_alucinacao.py
python test_extracao.py
```

Os testes verificam se os prompts contêm as regras anti-alucinação (recusa de temas não corporativos, priorização de documentos, citação de fontes) e se a limpeza de texto remove paginação e marcas de confidencialidade.

## Geração de documentos de exemplo

Use a própria IA para gerar documentos fictícios para teste:

> *"Gere um PDF fictício com a política de férias do RH da empresa, incluindo regras de solicitação, prazos e cálculo de dias."*

Salve o resultado na pasta `documentos/rh/` e reindexe.

## Licença

Este projeto é parte do desafio **Alura Agentes — ONE IA FOR TECH**.
