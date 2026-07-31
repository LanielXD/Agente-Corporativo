# ──────────────────────────────────────────────
# EMBEDDINGS PROVIDER — API com fallback local
# ──────────────────────────────────────────────

"""
Provider de embeddings com suporte a múltiplos provedores:
- HuggingFace Inference API (gratuita, recomendada)
- OpenAI API
- Local (HuggingFaceEmbeddings) - fallback apenas

Fluxo:
    try:
        usar API de embeddings (HuggingFace ou OpenAI)
    except:
        usar HuggingFaceEmbeddings local (fallback)
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings

from config import get_config

logger = logging.getLogger(__name__)


class EmbeddingProvider(Embeddings):
    """Interface base para provedores de embeddings."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de documentos."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Gera embedding para uma query."""
        pass


class HuggingFaceAPIEmbeddings(EmbeddingProvider):
    """Embeddings via HuggingFace Inference API (gratuita)."""

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN")
        self.batch_size = batch_size
        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        self._headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        if not self.api_key:
            raise RuntimeError("HF_API_KEY não configurado para HuggingFace API")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import requests

        response = requests.post(
            self._api_url,
            headers=self._headers,
            json={
                "inputs": texts,
                "options": {"wait_for_model": True, "use_cache": True},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                embeddings = self._call_api(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Erro ao gerar embeddings (batch {i//self.batch_size}): {e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbeddings(EmbeddingProvider):
    """Embeddings via OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.dimension = dimension

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY não configurado")

        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension if "3-" in self.model else None,
        )
        return [e.embedding for e in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LocalHuggingFaceEmbeddings(EmbeddingProvider):
    """Embeddings local usando sentence-transformers (fallback)."""

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.model_name = model
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"Embeddings locais carregados: {model}")
        except Exception as e:
            logger.error(f"Falha ao carregar embeddings locais: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


def get_embedding_provider(
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 32,
) -> EmbeddingProvider:
    """
    Factory para criar o provider de embeddings.

    Args:
        provider: "auto" | "huggingface" | "openai" | "local"
        model: Nome do modelo
        api_key: Chave da API
        batch_size: Tamanho do batch para API

    Returns:
        Instância de EmbeddingProvider

    Raises:
        RuntimeError: Se nenhum provider estiver disponível
    """
    config = get_config()
    cfg = config.embedding

    model = model or cfg.model
    api_key = api_key or cfg.api_key
    batch_size = batch_size or cfg.batch_size
    provider = provider or cfg.provider

    # Auto-detecta provedor
    if provider == "auto":
        # Prioridade: HuggingFace API > OpenAI > Local
        if cfg.api_key or os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN"):
            provider = "huggingface"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "local"

    # 1) HuggingFace Inference API
    if provider == "huggingface":
        try:
            return HuggingFaceAPIEmbeddings(
                model=model,
                api_key=api_key,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.warning(f"Falha ao iniciar HuggingFace API: {e}")
            if provider != "auto":
                raise

    # 2) OpenAI API
    if provider == "openai":
        try:
            return OpenAIEmbeddings(
                model=model or "text-embedding-3-small",
                api_key=api_key,
            )
        except Exception as e:
            logger.warning(f"Falha ao iniciar OpenAI API: {e}")
            if provider != "auto":
                raise

    # 3) Local (fallback)
    try:
        return LocalHuggingFaceEmbeddings(model=model)
    except Exception as e:
        logger.error(f"Falha ao carregar embeddings locais: {e}")
        raise RuntimeError(f"Nenhum provider de embeddings disponível: {e}")