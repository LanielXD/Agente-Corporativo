# ──────────────────────────────────────────────
# CONFIG — Configuração centralizada do projeto
# ──────────────────────────────────────────────

"""
Configuração centralizada usando Pydantic Settings.
Carrega de variáveis de ambiente, st.secrets ou arquivo .env.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GroqConfig(BaseSettings):
    """Configuração do Groq LLM."""
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: str = ""


class EmbeddingConfig(BaseSettings):
    """Configuração de embeddings."""
    model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    dimension: int = 384
    provider: str = "auto"  # auto, huggingface, local
    api_key: str = ""
    batch_size: int = 32


class PineconeConfig(BaseSettings):
    """Configuração do Pinecone."""
    api_key: str = ""
    index_name: str = "susan-ai"
    region: str = "us-east-1"
    cloud: str = "aws"


class RerankerConfig(BaseSettings):
    """Configuração do Reranker."""
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 3


class RetrievalConfig(BaseSettings):
    """Configuração de busca."""
    top_k: int = 3
    score_threshold: float = 0.0


class ResponsavelConfig(BaseSettings):
    """Responsáveis por setor."""
    rh: str = "Maria Oliveira"
    financeiro: str = "João Santos"
    juridico: str = "Dra. Ana Costa"


class ChunkingConfig(BaseSettings):
    """Configuração de chunking."""
    chunk_size: int = 600
    chunk_overlap: int = 60


class CuradoriaConfig(BaseSettings):
    """Configuração de curadoria de documentos."""
    ignorar_se_conter: List[str] = [
        "rascunho", "draft", "old", "test", "tmp", "copia", "backup"
    ]
    ignorar_exatos: List[str] = [
        "notas.md", "README.md", "pessoal.txt", "anotacoes.txt"
    ]
    manter_versao_oficial: bool = True


class Settings(BaseSettings):
    """Configuração principal do projeto."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Configurações por seção
    groq: GroqConfig = GroqConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    pinecone: PineconeConfig = PineconeConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    responsaveis: ResponsavelConfig = ResponsavelConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    curadoria: CuradoriaConfig = CuradoriaConfig()

    @field_validator("groq", "embedding", "pinecone", mode="before")
    @classmethod
    def _load_nested(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, dict):
            return v
        return {}

    def get_responsaveis_dict(self) -> Dict[str, str]:
        """Retorna responsáveis como dict."""
        return {
            "rh": self.responsaveis.rh,
            "financeiro": self.responsaveis.financeiro,
            "juridico": self.responsaveis.juridico,
        }


@lru_cache
def get_config() -> Settings:
    """
    Retorna configuração singleton.
    Cache evita recarregar variáveis de ambiente a cada chamada.
    """
    # Carregar de st.secrets se disponível (Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            secrets = st.secrets
            mapeamento = {
                "GROQ_API_KEY": "GROQ__API_KEY",
                "PINECONE_API_KEY": "PINECONE__API_KEY",
                "PINECONE_INDEX": "PINECONE__INDEX_NAME",
                "PINECONE_REGION": "PINECONE__REGION",
                "HF_API_KEY": "EMBEDDING__API_KEY",
                "EMBEDDING_PROVIDER": "EMBEDDING__PROVIDER",
            }
            for chave_origem, chave_destino in mapeamento.items():
                valor = secrets.get(chave_origem)
                if valor:
                    os.environ.setdefault(chave_destino, valor)
    except Exception:
        pass  # st.secrets não disponível fora do Streamlit

    return Settings()


def get_embedding_model() -> str:
    """Retorna modelo de embedding configurado."""
    return get_config().embedding.model


def get_embedding_dimension() -> int:
    """Retorna dimensão do embedding."""
    return get_config().embedding.dimension


def get_pinecone_index() -> str:
    """Retorna nome do índice Pinecone."""
    return get_config().pinecone.index_name


def get_groq_model() -> str:
    """Retorna modelo Groq configurado."""
    return get_config().groq.model


def get_groq_temperature() -> float:
    """Retorna temperatura Groq."""
    return get_config().groq.temperature


def get_groq_max_tokens() -> int:
    """Retorna max tokens Groq."""
    return get_config().groq.max_tokens


def is_reranker_enabled() -> bool:
    """Verifica se reranker está habilitado."""
    return get_config().reranker.enabled


def get_reranker_model() -> str:
    """Retorna modelo do reranker."""
    return get_config().reranker.model


def get_retrieval_top_k() -> int:
    """Retorna top_k para busca."""
    return get_config().retrieval.top_k


def get_chunk_size() -> int:
    """Retorna tamanho do chunk."""
    return get_config().chunking.chunk_size


def get_chunk_overlap() -> int:
    """Retorna overlap do chunk."""
    return get_config().chunking.chunk_overlap


# Instância global para compatibilidade
config = get_config()