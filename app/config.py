"""Application configuration and config-file loaders.

Settings are sourced from environment variables (prefix ``LLMOPS_``) with safe
local-first defaults. Nothing here requires a paid API key: the default provider
is the deterministic ``mock`` provider.

This module also provides loaders for the YAML files under ``configs/`` that
describe prompt templates, model configs, and evaluation run configs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (app/config.py -> app -> root).
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
DOCUMENTS_DIR = ROOT_DIR / "documents"
DATASETS_DIR = ROOT_DIR / "datasets"


class Settings(BaseSettings):
    """Runtime settings, overridable via environment or a local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="LLMOPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=ROOT_DIR / "data")
    database_url: str = Field(default="")
    log_level: str = Field(default="INFO")

    default_provider: str = Field(default="mock")
    default_model: str = Field(default="mock-small")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Optional real-provider settings (never required).
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # OpenRouter (OpenAI-compatible gateway to many real models, incl. free ones).
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_referer: str | None = Field(default=None, alias="OPENROUTER_REFERER")
    openrouter_title: str | None = Field(default="LLMOpsForge", alias="OPENROUTER_TITLE")

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        """Derive the SQLite URL from ``data_dir`` when not explicitly set."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.database_url:
            self.database_url = f"sqlite:///{self.data_dir / 'llmopsforge.db'}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


# --------------------------------------------------------------------------- #
# Config-file models + loaders
# --------------------------------------------------------------------------- #


class PromptBehavior(BaseModel):
    """Knobs that influence how the generator/mock provider behaves."""

    enforce_grounding: bool = False
    always_cite: bool = False
    strict_json: bool = False
    verbosity: str = "low"


class PromptTemplateConfig(BaseModel):
    id: str
    name: str = ""
    version: int = 1
    description: str = ""
    system: str = ""
    instructions: str = ""
    behavior: PromptBehavior = Field(default_factory=PromptBehavior)


class ModelConfigSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 512
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class RagConfig(BaseModel):
    top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 100
    require_citations: bool = True


class EvalThresholds(BaseModel):
    answer_correctness_score: float = 0.5
    citation_correctness_score: float = 0.5
    grounding_score: float = 0.4
    retrieval_relevance_score: float = 0.5
    allow_hallucination: bool = False


class EvalConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = "default"
    description: str = ""
    rag: RagConfig = Field(default_factory=RagConfig)
    prompt_template_id: str = "prompt_v1"
    model_config_id: str = "mock-small"
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load an evaluation run config from a YAML file."""
    return EvalConfig(**_read_yaml(Path(path)))


def load_prompt_template(template_id: str, configs_dir: Path = CONFIGS_DIR) -> PromptTemplateConfig:
    """Find and load a prompt template by id from any YAML under ``configs/``."""
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        data = _read_yaml(yaml_path)
        if data.get("id") == template_id and "system" in data:
            return PromptTemplateConfig(**data)
    raise KeyError(f"Prompt template '{template_id}' not found in {configs_dir}")


def load_all_prompt_templates(configs_dir: Path = CONFIGS_DIR) -> list[PromptTemplateConfig]:
    templates: list[PromptTemplateConfig] = []
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        data = _read_yaml(yaml_path)
        if data.get("id") and "system" in data:
            templates.append(PromptTemplateConfig(**data))
    return templates


def load_model_config(model_id: str, configs_dir: Path = CONFIGS_DIR) -> ModelConfigSpec:
    """Load a single model config by id from ``model_configs.yaml``."""
    for spec in load_all_model_configs(configs_dir):
        if spec.id == model_id:
            return spec
    raise KeyError(f"Model config '{model_id}' not found in {configs_dir}")


def load_all_model_configs(configs_dir: Path = CONFIGS_DIR) -> list[ModelConfigSpec]:
    data = _read_yaml(configs_dir / "model_configs.yaml")
    return [ModelConfigSpec(**m) for m in data.get("models", [])]
