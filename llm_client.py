"""
InceptionRAG - LLM Client Abstraction

Supports multiple API providers:
- OpenAI (default)
- Azure OpenAI
- Custom API endpoints (proxy/relay services)

Usage:
    from llm_client import create_llm_client, LLMConfig
    
    # OpenAI
    client = create_llm_client(LLMConfig(
        provider="openai",
        api_key="sk-xxx",
        model="gpt-4o-mini"
    ))
    
    # Azure OpenAI
    client = create_llm_client(LLMConfig(
        provider="azure",
        api_key="your-key",
        api_base="https://your-resource.openai.azure.com/",
        api_version="2024-02-15-preview",
        deployment="your-deployment"
    ))
    
    # Custom API (proxy/relay services)
    client = create_llm_client(LLMConfig(
        provider="custom",
        api_key="your-key",
        api_base="https://api.example.com/v1",
        model="gpt-4o-mini"
    ))
    
    # Call
    response = client.chat("What is 2+2?")
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LLMConfig:
    """LLM API Configuration"""
    provider: str = "openai"           # openai / azure / custom
    api_key: str = ""
    api_base: str = ""                 # Custom base URL
    api_version: str = ""              # Azure only
    deployment: str = ""               # Azure only
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 256
    
    @classmethod
    def from_yaml(cls, path: str) -> "LLMConfig":
        """Load config from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        llm_config = data.get("llm", {})
        return cls(
            provider=llm_config.get("provider", "openai"),
            api_key=llm_config.get("api_key", ""),
            api_base=llm_config.get("api_base", ""),
            api_version=llm_config.get("api_version", ""),
            deployment=llm_config.get("deployment", ""),
            model=llm_config.get("model", "gpt-4o-mini"),
        )
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables"""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", "") or os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_base=os.getenv("OPENAI_API_BASE", "") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )


class LLMClient:
    """Abstract LLM Client"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate client based on provider"""
        if self.config.provider == "azure":
            self._init_azure()
        else:
            # OpenAI or Custom (both use OpenAI client)
            self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI or Custom API client"""
        from openai import OpenAI
        
        kwargs = {"api_key": self.config.api_key}
        
        # Custom base URL (for proxy/relay or other providers)
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base
        
        self._client = OpenAI(**kwargs)
        self._model = self.config.model
    
    def _init_azure(self):
        """Initialize Azure OpenAI client"""
        from openai import AzureOpenAI
        
        self._client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.api_base,
        )
        # Azure uses deployment name instead of model
        self._model = self.config.deployment
    
    def chat(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Send a chat message and get response.
        
        Args:
            user_message: User message content
            system_message: System prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Response text
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        return response.choices[0].message.content or ""
    
    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Send multiple messages and get response.
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Response text
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        return response.choices[0].message.content or ""


def create_llm_client(config: LLMConfig = None, config_path: str = None) -> LLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        config: LLMConfig object
        config_path: Path to config.yaml
        
    Returns:
        LLMClient instance
    """
    if config is None:
        if config_path:
            config = LLMConfig.from_yaml(config_path)
        else:
            # Try default config path
            default_path = Path(__file__).parent / "config.yaml"
            if default_path.exists():
                config = LLMConfig.from_yaml(str(default_path))
            else:
                # Fallback to environment variables
                config = LLMConfig.from_env()
    
    return LLMClient(config)


# Convenience function for backward compatibility
def get_openai_client(api_key: str, api_base: str = None, model: str = "gpt-4o-mini") -> LLMClient:
    """
    Quick helper to create OpenAI-compatible client.
    
    Args:
        api_key: API key
        api_base: Optional custom base URL
        model: Model name
        
    Returns:
        LLMClient instance
    """
    config = LLMConfig(
        provider="custom" if api_base else "openai",
        api_key=api_key,
        api_base=api_base or "",
        model=model,
    )
    return LLMClient(config)
