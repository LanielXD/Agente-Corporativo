# ──────────────────────────────────────────────
# CHAIN — Pipeline RAG completo
# ──────────────────────────────────────────────

"""
Pipeline RAG: Retrieval + Augmentation + Generation.
Orquestra busca, contexto e geração de resposta via Groq.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

from config import get_config
from rag.retriever import PineconeRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Resposta completa do pipeline RAG."""
    answer: str
    sources: List[Dict[str, str]]
    query: str
    metadata: Dict[str, Any] = None


class RAGChain:
    """Pipeline RAG completo: Retrieve -> Augment -> Generate."""

    def __init__(
        self,
        embeddings: Embeddings,
        llm: Optional[BaseChatModel] = None,
        retriever: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        self.embeddings = embeddings
        self.config = get_config()

        # LLM (Groq)
        self.llm = llm or self._create_llm()

        # Retriever
        self.retriever = retriever or self._create_retriever()

        # Prompt template
        self.prompt = self._create_prompt(system_prompt)

        # Chain
        self.chain = self._create_chain()

    def _create_llm(self) -> BaseChatModel:
        """Cria instância do Groq LLM."""
        config = get_config()
        return ChatGroq(
            model=config.groq.model,
            temperature=config.groq.temperature,
            max_tokens=config.groq.max_tokens,
            api_key=config.groq.api_key,
        )

    def _create_retriever(self):
        """Cria retriever Pinecone."""
        from rag.retriever import create_retriever
        return create_retriever(
            embeddings=self.embeddings,
            k=self.config.retrieval.top_k,
            score_threshold=self.config.retrieval.score_threshold,
            use_reranker=self.config.reranker.enabled,
        )

    def _create_prompt(self, system_prompt: Optional[str] = None) -> ChatPromptTemplate:
        """Cria template de prompt para o LLM."""
        responsaveis = get_config().get_responsaveis_dict()
        responsaveis_str = "\n".join(
            f"- {dept.upper()}: {nome}" for dept, nome in responsaveis.items()
        )

        default_prompt = (
            "Voce e um assistente corporativo especializado em analisar documentos internos da empresa.\n\n"
            "REGRAS IMPORTANTES:\n"
            "- Responda com base PRIMEIRAMENTE nos documentos fornecidos abaixo. Se eles nao cobrirem "
            "totalmente, complemente com seu conhecimento geral sobre RH, Financeiro e Juridico.\n"
            "- Se a pergunta for sobre amenidades, cultura geral, previsao do tempo ou qualquer assunto "
            "NAO corporativo, responda educadamente que so pode ajudar com informacoes corporativas.\n"
            "- Se os documentos nao contiverem informacao suficiente para responder e o assunto nao for "
            "sobre RH, Financeiro ou Juridico, avise que o assunto nao foi encontrado na base de conhecimento.\n"
            "- Se a pergunta for sobre um arquivo especifico (ex: \"despesas.csv\", \"politica de ferias\"), "
            "informe o que contem naquele arquivo com base nos trechos disponiveis.\n\n"
            f"RESPONSAVEIS PELOS SETORES:\n{responsaveis_str}\n\n"
            "REGRAS DE ALTERACAO:\n"
            "- Se o colaborador pedir para ALTERAR ou EXCLUIR algum arquivo, NAO faca nenhuma alteracao. "
            "Informe que ele precisa de autorizacao do responsavel pelo setor do arquivo e exiba o nome do responsavel.\n"
            "- Se o colaborador pedir para ATUALIZAR algum arquivo, informe que voce ira notificar "
            "o responsavel pelo setor sobre a solicitacao.\n\n"
            "Documentos:\n"
            "{context}\n\n"
            "Pergunta: {question}\n\n"
            "Responda de forma clara e direta em portugues.\n"
            "Sempre cite o nome do arquivo entre aspas ao usar uma informacao dele.\n"
            "Exemplo: 'Conforme \"Politica_de_Ferias.pdf\", as ferias devem ser solicitadas "
            "com 30 dias de antecedencia.'"
        )

        prompt_text = system_prompt or default_prompt

        return ChatPromptTemplate.from_template(
            "{system}\n\nContexto:\n{context}\n\nPergunta: {question}\n\nResposta:"
        ).partial(system=prompt_text)

    def _create_chain(self):
        """Cria a chain RAG usando LCEL (LangChain Expression Language)."""

        def format_docs(docs: List[Document]) -> str:
            """Formata documentos para o prompt."""
            if not docs:
                return "Nenhum documento relevante encontrado."
            formatted = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", doc.metadata.get("fonte", "desconhecido"))
                dept = doc.metadata.get("departamento", "desconhecido")
                formatted.append(
                    f"Documento [{i+1}] — \"{source}\" ({dept}):\n{doc.page_content}"
                )
            return "\n\n".join(formatted)

        def format_sources(docs: List[Document]) -> List[Dict[str, str]]:
            """Formata fontes para retorno."""
            sources = []
            seen = set()
            for doc in docs:
                source = doc.metadata.get("source", doc.metadata.get("fonte", "desconhecido"))
                dept = doc.metadata.get("departamento", "desconhecido")
                key = (source, dept)
                if key not in seen:
                    seen.add(key)
                    sources.append({"fonte": source, "departamento": dept})
            return sources

        # Chain de recuperação + formatação
        retrieval_chain = (
            RunnableParallel(
                {
                    "context": lambda x: self.retriever.search(x["question"]),
                    "question": RunnablePassthrough(),
                }
            )
            | RunnableParallel(
                {
                    "answer": lambda x: self.prompt | self.llm | StrOutputParser(),
                    "sources": lambda x: format_sources(x["context"].documents),
                    "retrieval": lambda x: x["context"],
                }
            )
        )

        return retrieval_chain

    def invoke(self, question: str) -> RAGResponse:
        """Executa o pipeline RAG completo."""
        logger.info(f"Processando pergunta: {question}")

        try:
            result = self.chain.invoke({"question": question})

            answer = result["answer"]
            sources = result["sources"]
            retrieval = result["retrieval"]

            logger.info(f"Resposta gerada com {len(sources)} fontes")

            return RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                metadata={"num_sources": len(sources)},
            )

        except Exception as e:
            logger.error(f"Erro no pipeline RAG: {e}")
            raise

    def stream(self, question: str) -> Iterator[str]:
        """Stream da resposta token a token."""
        logger.info(f"Streaming resposta para: {question}")

        try:
            result = self.chain.invoke({"question": question})
            answer = result["answer"]

            # Simula streaming token a token
            for chunk in answer.split():
                yield chunk + " "

        except Exception as e:
            logger.error(f"Erro no streaming: {e}")
            yield f"Erro ao gerar resposta: {e}"


class StreamingRAGChain(RAGChain):
    """Chain RAG com streaming real via callbacks."""

    def stream(self, question: str) -> Iterator[str]:
        """Stream real da resposta."""
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        streaming_llm = ChatGroq(
            model=self.config.groq.model,
            temperature=self.config.groq.temperature,
            max_tokens=self.config.groq.max_tokens,
            api_key=self.config.groq.api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        chain = self._create_chain()
        chain.llm = streaming_llm

        for chunk in chain.stream({"question": question}):
            if "answer" in chunk:
                yield chunk["answer"]


def create_rag_chain(
    embeddings: Embeddings,
    llm: Optional[BaseChatModel] = None,
    k: int = 3,
    use_reranker: bool = False,
) -> RAGChain:
    """Factory para criar chain RAG configurada."""
    return RAGChain(
        embeddings=embeddings,
        llm=llm,
        retriever=None,  # Será criado internamente
    )


# Instância global lazy
_rag_chain: Optional[RAGChain] = None


def get_rag_chain(embeddings: Embeddings) -> RAGChain:
    """Retorna chain RAG singleton."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = create_rag_chain(embeddings)
    return _rag_chain