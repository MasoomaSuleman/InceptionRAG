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

HF_CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or "/tmp/masoo20/hf-cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


@dataclass
class LLMConfig:
    """LLM API Configuration"""
    provider: str = "openai"           # openai / azure / custom / hf
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


def _resolve_local_model_path(model_name: str) -> str:
    """Resolve a repo ID to the cached snapshot directory when running fully offline."""
    if not model_name:
        return model_name

    if os.path.exists(model_name):
        return model_name

    if snapshot_download is not None and "/" in model_name:
        try:
            return snapshot_download(
                repo_id=model_name,
                cache_dir=HF_CACHE_DIR,
                local_files_only=True,
            )
        except Exception:
            pass

    candidate = os.path.join(HF_CACHE_DIR, "models--" + model_name.replace("/", "--"))
    if os.path.isdir(candidate):
        snapshot_root = os.path.join(candidate, "snapshots")
        if os.path.isdir(snapshot_root):
            snapshot_dirs = sorted(os.listdir(snapshot_root))
            if snapshot_dirs:
                return os.path.join(snapshot_root, snapshot_dirs[0])
        return candidate

    return model_name


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
        elif self.config.provider == "hf":
            self._init_hf()
        else:
            # OpenAI or Custom (both use OpenAI client)
            self._init_openai()

    def _init_hf(self):
        """Initialize a local Hugging Face generation model."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = _resolve_local_model_path(self.config.model or "Qwen/Qwen2.5-7B-Instruct")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        try:
            if device == "cuda":
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                self._model = self._model.to(device)
        except RuntimeError as e:
            if device == "cuda" and "out of memory" in str(e).lower():
                print("[Warning] CUDA OOM while loading local HF model; retrying on CPU.")
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                self._model = self._model.to("cpu")
                device = "cpu"
            else:
                raise

        self._model.eval()
        self._device = torch.device(device)

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
        if self.config.provider == "hf":
            return self._chat_hf(user_message, system_message, temperature, max_tokens)

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

    def _chat_hf(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate a response with a local Hugging Face model."""
        import torch

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_temperature = temperature if temperature is not None else self.config.temperature
        gen_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=gen_max_tokens,
                    temperature=gen_temperature,
                    do_sample=gen_temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            if self._device.type == "cuda" and "out of memory" in str(e).lower():
                print("[Warning] CUDA OOM during local HF generation; retrying on CPU.")
                self._model = self._model.to("cpu")
                self._device = torch.device("cpu")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=gen_max_tokens,
                        temperature=gen_temperature,
                        do_sample=gen_temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            else:
                raise

        generated_text = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated_text.strip()
    
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
        if self.config.provider == "hf":
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            gen_temperature = temperature if temperature is not None else self.config.temperature
            gen_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

            import torch
            try:
                with torch.no_grad():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=gen_max_tokens,
                        temperature=gen_temperature,
                        do_sample=gen_temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            except RuntimeError as e:
                if self._device.type == "cuda" and "out of memory" in str(e).lower():
                    print("[Warning] CUDA OOM during local HF generation; retrying on CPU.")
                    self._model = self._model.to("cpu")
                    self._device = torch.device("cpu")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    with torch.no_grad():
                        generated = self._model.generate(
                            **inputs,
                            max_new_tokens=gen_max_tokens,
                            temperature=gen_temperature,
                            do_sample=gen_temperature > 0,
                            pad_token_id=self._tokenizer.eos_token_id,
                        )
                else:
                    raise
            return self._tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

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
