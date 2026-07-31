# ──────────────────────────────────────────────
# PINECONE CLIENT — Cliente Pinecone com LangChain
# ──────────────────────────────────────────────

"""
Cliente Pinecone para operações de vector store.
Gerencia conexão, criação de índice e operações CRUD.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PineconeConfig:
    """Configuração do Pinecone."""
    api_key: str
    index_name: str
    region: str = "us-east-1"
    cloud: str = "aws"
    dimension: int = 384
    metric: str = "cosine"


_NS_PADRAO = "default"


class PineconeClient:
    """Cliente para gerenciar operações no Pinecone."""

    _instance: Optional["PineconeClient"] = None
    _vectorstore: Optional[PineconeVectorStore] = None
    _index = None
    _pc = None
    _config: Optional[PineconeConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._initialize_config()

    def _initialize_config(self):
        """Carrega configuração do Pinecone."""
        config = get_config()
        self._config = PineconeConfig(
            api_key=config.pinecone.api_key,
            index_name=config.pinecone.index_name,
            region=config.pinecone.region,
            cloud=config.pinecone.cloud,
            dimension=config.embedding.dimension,
        )

    def get_client(self) -> Pinecone:
        """Retorna cliente Pinecone (cached)."""
        if self._pc is None:
            if not self._config.api_key:
                raise RuntimeError("PINECONE_API_KEY não configurado")
            self._pc = Pinecone(api_key=self._config.api_key)
        return self._pc

    def ensure_index_exists(self) -> bool:
        """
        Verifica se índice existe, cria se necessário.
        Returns True se índice foi criado agora.
        """
        pc = self.get_client()

        existing_indexes = pc.list_indexes().names()
        if self._config.index_name in existing_indexes:
            logger.info(f"Índice '{self._config.index_name}' já existe")
            return False

        logger.info(f"Criando índice '{self._config.index_name}'...")
        pc.create_index(
            name=self._config.index_name,
            dimension=self._config.dimension,
            metric=self._config.metric,
            spec=ServerlessSpec(
                cloud=self._config.cloud,
                region=self._config.region,
            ),
        )
        logger.info(f"Índice '{self._config.index_name}' criado com sucesso")
        return True

    def get_index(self):
        """Retorna objeto de índice Pinecone."""
        if self._index is None:
            self.ensure_index_exists()
            self._index = self.get_client().Index(self._config.index_name)
        return self._index

    def get_vectorstore(
        self,
        embeddings: Any,
        namespace: Optional[str] = None,
    ) -> PineconeVectorStore:
        """
        Retorna PineconeVectorStore do LangChain.
        Namespace permite separar documentos por categoria (opcional).
        """
        namespace = namespace or _NS_PADRAO
        self.ensure_index_exists()
        return PineconeVectorStore(
            index=self.get_index(),
            embedding=embeddings,
            namespace=namespace,
            text_key="content",
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Any,
        namespace: Optional[str] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Adiciona documentos ao índice.
        Returns: número de documentos adicionados.
        """
        if not documents:
            return 0

        vectorstore = self.get_vectorstore(embeddings, namespace)
        vectorstore.add_documents(documents)
        logger.info(f"{len(documents)} documentos adicionados ao namespace '{namespace}'")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        embeddings: Optional[Any] = None,
    ) -> List[Document]:
        """Busca por similaridade."""
        vectorstore = self.get_vectorstore(embeddings, namespace)
        return vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        embeddings: Optional[Any] = None,
    ) -> List[tuple[Document, float]]:
        """Busca por similaridade com scores."""
        vectorstore = self.get_vectorstore(embeddings, namespace)
        return vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def delete_namespace(self, namespace: str) -> bool:
        """Deleta todos os vetores de um namespace."""
        try:
            index = self.get_index()
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Namespace '{namespace}' deletado")
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar namespace '{namespace}': {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do índice."""
        index = self.get_index()
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }

    def delete_index(self) -> bool:
        """Deleta o índice completamente (CUIDADO!)."""
        try:
            pc = self.get_client()
            pc.delete_index(self._config.index_name)
            self._vectorstore = None
            self._index = None
            logger.warning(f"Índice '{self._config.index_name}' DELETADO")
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar índice: {e}")
            return False

    def reset(self):
        """Força reset das conexões cached."""
        self._vectorstore = None
        self._index = None
        self._pc = None


# Instância global
pinecone_client = PineconeClient()