from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """单个模型的配置"""
    name: str                          # 模型名称
    provider: str                      # 提供商：openai, ollama等
    env_key_api: str                   # API_KEY的环境变量名
    env_key_base_url: str              # BASE_URL的环境变量名
    model_name: str                    # 模型标识符
    temperature: float = 0.7
    max_tokens: int = 1000
    extra_params: Dict[str, Any] = field(default_factory=dict)


# 定义所有可用的模型配置
MODELS_CONFIG = {
    "claude": ModelConfig(
        name="claude",
        provider="openai",
        env_key_api="CLAUDE_API_KEY",
        env_key_base_url="CLAUDE_BASE_URL",
        model_name="claude-3.5-sonnet",
        temperature=0.7,
        max_tokens=2000,
    ),
    "deepseek": ModelConfig(
        name="deepseek",
        provider="openai",
        env_key_api="DEEPSEEK_API_KEY",
        env_key_base_url="DEEPSEEK_BASE_URL",
        model_name="deepseek-chat",
        temperature=0.3,
        max_tokens=1000,
    ),
    "ollama": ModelConfig(
        name="ollama",
        provider="ollama",
        env_key_api="OLLAMA_API_KEY",
        env_key_base_url="OLLAMA_BASE_URL",
        model_name="llama2",
        temperature=0.5,
        max_tokens=500,
    ),
}
