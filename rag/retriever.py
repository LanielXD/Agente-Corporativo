# ──────────────────────────────────────────────
# RETRIEVER — Busca vetorial com Pinecone
# ──────────────────────────────────────────────

"""
Retriever para busca semântica com Pinecone.
Suporta filtros por metadados, reranking opcional.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from config import get_config
from vectorstore.pinecone_client import pinecone_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Resultado de uma busca."""
    documents: List[Document]
    scores: Optional[List[float]] = None
    query: str = ""
    total_found: int = 0


class PineconeRetriever(BaseRetriever):
    """Retriever usando Pinecone com suporte a filtros e reranking."""

    embeddings: Embeddings = Field(description="Embeddings model for vector search")
    k: int = 3
    score_threshold: float = 0.0
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 5
    _reranker: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        embeddings: Embeddings,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        use_reranker: Optional[bool] = None,
        **kwargs,
    ):
        # Pass embeddings to parent for pydantic validation
        kwargs['embeddings'] = embeddings
        super().__init__(**kwargs)
        self.embeddings = embeddings
        config = get_config()

        if k is not None:
            self.k = k
        else:
            self.k = config.retrieval.top_k

        if score_threshold is not None:
            self.score_threshold = score_threshold
        else:
            self.score_threshold = config.retrieval.score_threshold

        if namespace is not None:
            self.namespace = namespace

        if filter is not None:
            self.filter = filter

        if use_reranker is not None:
            self.use_reranker = use_reranker
        else:
            self.use_reranker = config.reranker.enabled

        if not self.reranker_model and config.reranker.model:
            self.reranker_model = config.reranker.model

        if not self.reranker_top_n and config.reranker.top_n:
            self.reranker_top_n = config.reranker.top_n

    def _get_reranker(self):
        """Lazy load do reranker."""
        if self._reranker is None and self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_model)
                logger.info(f"Reranker carregado: {self.reranker_model}")
            except Exception as e:
                logger.warning(f"Falha ao carregar reranker: {e}")
                self.use_reranker = False
        return self._reranker

    def _apply_score_threshold(
        self,
        docs: List[Document],
        scores: List[float],
    ) -> List[Document]:
        """Filtra documentos abaixo do threshold."""
        if self.score_threshold <= 0:
            return docs
        filtered = [
            doc for doc, score in zip(docs, scores)
            if score >= self.score_threshold
        ]
        logger.debug(f"Threshold {self.score_threshold}: {len(docs)} -> {len(filtered)} docs")
        return filtered

    def _rerank(
        self,
        query: str,
        docs: List[Document],
        scores: Optional[List[float]] = None,
    ) -> List[Document]:
        """Rerank documentos usando CrossEncoder."""
        reranker = self._get_reranker()
        if not reranker:
            return docs[:self.reranker_top_n]

        try:
            pairs = [[query, doc.page_content] for doc in docs]
            rerank_scores = reranker.predict(pairs)
            scored = list(zip(docs, rerank_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:self.reranker_top_n]]
        except Exception as e:
            logger.warning(f"Erro no reranking: {e}")
            return docs[:self.reranker_top_n]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Busca documentos relevantes (interface BaseRetriever)."""
        return self.search(query).documents

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
    ) -> RetrievalResult:
        """
        Busca documentos por similaridade.

        Args:
            query: Pergunta do usuário
            k: Número de documentos a retornar
            namespace: Namespace do Pinecone (opcional)
            filter: Filtro de metadados (ex: {"departamento": "rh"})

        Returns:
            RetrievalResult com documentos e scores
        """
        k = k or self.k
        ns = namespace or self.namespace
        flt = filter or self.filter

        logger.info(f"Busca: '{query}' | k={k} | ns={ns} | filter={flt}")

        try:
            # Busca com scores
            docs_scores = pinecone_client.similarity_search_with_score(
                query=query,
                k=k * 2 if self.use_reranker else k,
                namespace=ns,
                filter=flt,
                embeddings=self.embeddings,
            )

            if not docs_scores:
                logger.warning(f"Nenhum documento encontrado para: {query}")
                return RetrievalResult(documents=[], query=query, total_found=0)

            docs = [doc for doc, _ in docs_scores]
            scores = [score for _, score in docs_scores]

            # Filtra por threshold
            docs = self._apply_score_threshold(docs, scores)

            # Reranking
            if self.use_reranker and len(docs) > 1:
                docs = self._rerank(query, docs)

            # Limita ao k final
            docs = docs[:k]

            logger.info(f"Retornando {len(docs)} documentos")
            return RetrievalResult(
                documents=docs,
                scores=scores[:len(docs)] if scores else None,
                query=query,
                total_found=len(docs_scores),
            )

        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            return RetrievalResult(documents=[], query=query, total_found=0)

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
    ) -> List[tuple[Document, float]]:
        """Busca retornando tuplas (doc, score)."""
        result = self.search(query, k, namespace, filter)
        if result.scores:
            return list(zip(result.documents, result.scores))
        return [(doc, 1.0) for doc in result.documents]


def create_retriever(
    embeddings: Embeddings,
    k: int = 3,
    score_threshold: float = 0.0,
    namespace: Optional[str] = None,
    filter: Optional[Dict] = None,
    use_reranker: bool = False,
) -> PineconeRetriever:
    """Factory para criar retriever configurado."""
    return PineconeRetriever(
        embeddings=embeddings,
        k=k,
        score_threshold=score_threshold,
        namespace=namespace,
        filter=filter,
        use_reranker=use_reranker,
    )


# Instância global lazy
_retriever: Optional["PineconeRetriever"] = None


def get_retriever(embeddings: Embeddings) -> PineconeRetriever:
    """Retorna retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = create_retriever(embeddings)
    return _retriever