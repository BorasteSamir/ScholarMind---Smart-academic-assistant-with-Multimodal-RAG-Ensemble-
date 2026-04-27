"""
rag/llm_client.py
==================
LLM abstraction layer supporting multiple providers:
  - OpenAI (GPT-4o, GPT-4o-mini, etc.)
  - Anthropic (Claude 3 family)
  - Local LLM via Ollama REST API

The client presents a uniform interface regardless of backend:
    client = LLMClient()
    response = client.generate(system_prompt, user_prompt)

Switching providers requires only an env-var change (LLM_PROVIDER).
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

from config.settings import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LOCAL_LLM_URL,
    LOCAL_LLM_MODEL,
    GROK_API_KEY,
    GROK_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BASE INTERFACE
# ─────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Abstract base class for all LLM provider clients."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Generate a response from the LLM."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...


# ─────────────────────────────────────────────
# OPENAI CLIENT
# ─────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client using the openai Python SDK.
    Supports GPT-4o, GPT-4o-mini, GPT-3.5-turbo, etc.
    """

    def __init__(self, model: str = OPENAI_MODEL, api_key: str = OPENAI_API_KEY):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response in {time.time()-start:.2f}s ({len(result)} chars)")
            return result
        except Exception as exc:
            logger.error(f"OpenAI generation failed: {exc}")
            raise


# ─────────────────────────────────────────────
# ANTHROPIC CLIENT
# ─────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client using the anthropic Python SDK.
    Supports Claude 3 Haiku, Sonnet, Opus, etc.
    """

    def __init__(self, model: str = ANTHROPIC_MODEL, api_key: str = ANTHROPIC_API_KEY):
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        start = time.time()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            result = response.content[0].text.strip()
            logger.debug(f"Anthropic response in {time.time()-start:.2f}s")
            return result
        except Exception as exc:
            logger.error(f"Anthropic generation failed: {exc}")
            raise


# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    """
    Groq API client — OpenAI-compatible endpoint.
    Supports llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it, etc.
    """

    def __init__(self, model: str = GROQ_MODEL, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"Groq response in {time.time()-start:.2f}s ({len(result)} chars)")
            return result
        except Exception as exc:
            logger.error(f"Groq generation failed: {exc}")
            raise


# ─────────────────────────────────────────────
# GROK (xAI) CLIENT
# ─────────────────────────────────────────────

class GrokClient(BaseLLMClient):
    """
    xAI Grok API client — OpenAI-compatible endpoint.
    Supports grok-3, grok-3-fast, grok-3-mini, etc.
    """

    def __init__(self, model: str = GROK_MODEL, api_key: str = GROK_API_KEY):
        if not api_key:
            raise ValueError(
                "GROK_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"Grok response in {time.time()-start:.2f}s ({len(result)} chars)")
            return result
        except Exception as exc:
            logger.error(f"Grok generation failed: {exc}")
            raise


# ─────────────────────────────────────────────
# LOCAL (OLLAMA) CLIENT
# ─────────────────────────────────────────────

class LocalLLMClient(BaseLLMClient):
    """
    Client for locally-hosted LLMs via Ollama's REST API.
    Ollama must be running: https://ollama.ai

    Supported models: llama3, mistral, phi3, gemma, etc.
    """

    def __init__(self, model: str = LOCAL_LLM_MODEL, url: str = LOCAL_LLM_URL):
        self._model = model
        self._url = url

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "model": self._model,
            "prompt": combined_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        start = time.time()
        try:
            resp = requests.post(self._url, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            logger.debug(f"Local LLM response in {time.time()-start:.2f}s")
            return result
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self._url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as exc:
            logger.error(f"Local LLM generation failed: {exc}")
            raise


# ─────────────────────────────────────────────
# FACTORY / SINGLETON
# ─────────────────────────────────────────────

def create_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """
    Factory function — creates the appropriate LLM client based on LLM_PROVIDER.

    Args:
        provider: Override provider string. If None, uses LLM_PROVIDER env var.

    Returns:
        BaseLLMClient instance
    """
    provider = (provider or LLM_PROVIDER).lower()
    logger.info(f"Creating LLM client: provider={provider}")

    if provider == "openai":
        return OpenAIClient()
    elif provider in ("anthropic", "claude"):
        return AnthropicClient()
    elif provider in ("local", "ollama"):
        return LocalLLMClient()
    elif provider in ("grok", "xai"):
        return GrokClient()
    elif provider == "groq":
        return GroqClient()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Set LLM_PROVIDER to 'openai', 'anthropic', 'local', 'grok', or 'groq'."
        )


_default_client: Optional[BaseLLMClient] = None


def get_llm_client() -> BaseLLMClient:
    """Return the module-level singleton LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = create_llm_client()
    return _default_client
