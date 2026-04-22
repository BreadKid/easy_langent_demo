import os
from typing import Dict, Optional, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from config.models_config import MODELS_CONFIG


class LLMFactory:
    """LLM模型工厂 - 统一管理多个模型的生命周期"""

    def __init__(self, env_file: str = ".env"):
        """初始化工厂"""
        # 确保加载的是项目根目录的.env
        self.env_path = Path(__file__).parent.parent / env_file
        load_dotenv(self.env_path)
        self._llm_cache: Dict[str, Any] = {}  # 缓存已初始化的模型
        self._load_and_validate()

    def _get_default_model_from_env(self) -> str:
        """
        手动解析 .env 文件以获取 DEFAULT_LLM=true 的模型名称。
        """
        if not self.env_path.exists():
            return "deepseek"

        try:
            content = self.env_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            current_section = "deepseek"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    comment = line.lstrip("#").strip().lower()
                    for model_key in MODELS_CONFIG.keys():
                        if model_key in comment:
                            current_section = model_key
                            break
                    continue
                clean_line = line.replace('"', '').replace("'", "").replace(" ", "")
                if clean_line == "DEFAULT_LLM=true":
                    return current_section
        except Exception as e:
            print(f"⚠️ 解析 .env 获取默认模型失败: {e}")
            
        return "deepseek"

    def _load_and_validate(self):
        """验证所有配置的环境变量是否存在"""
        missing_configs = []
        for model_name, config in MODELS_CONFIG.items():
            api_key = os.getenv(config.env_key_api)
            # Ollama 可能不需要 API_KEY，但为了统一逻辑，我们根据 provider 判断
            if config.provider == "openai" and not api_key:
                missing_configs.append(f"  {model_name}: 缺少 {config.env_key_api}")

        if missing_configs:
            print("⚠️  警告 - 以下模型缺少配置：\n" + "\n".join(missing_configs))

    def get_llm(self, model_name: Optional[str] = None) -> Union[ChatOpenAI, OllamaLLM]:
        """
        获取已初始化的LLM模型（单例模式，缓存重用）

        Args:
            model_name: 模型名称 (claude, deepseek, ollama)

        Returns:
            ChatOpenAI 或 OllamaLLM 实例
        """
        if model_name is None:
            model_name = self._get_default_model_from_env()

        if model_name in self._llm_cache:
            return self._llm_cache[model_name]

        if model_name not in MODELS_CONFIG:
            available = ", ".join(MODELS_CONFIG.keys())
            raise ValueError(f"未知的模型: {model_name}\n可用模型: {available}")

        config = MODELS_CONFIG[model_name]
        api_key = os.getenv(config.env_key_api)
        base_url = os.getenv(config.env_key_base_url)

        # 根据提供商初始化不同的类
        if config.provider == "ollama":
            # Ollama 初始化逻辑
            llm = OllamaLLM(
                model=config.model_name,
                base_url=base_url or "http://localhost:11434",
                temperature=config.temperature,
                **(config.extra_params or {})
            )
        else:
            # 默认使用 OpenAI 兼容接口 (ChatOpenAI)
            if not api_key:
                raise ValueError(f"模型 '{model_name}' 的 API_KEY 未配置")
            
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )

        self._llm_cache[model_name] = llm
        print(f"✅ 初始化模型: {model_name} (Provider: {config.provider})")
        return llm

    def get_llm_with_custom_params(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[ChatOpenAI, OllamaLLM]:
        """
        获取模型并覆盖部分参数（不缓存）
        """
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"未知的模型: {model_name}")

        config = MODELS_CONFIG[model_name]
        api_key = os.getenv(config.env_key_api)
        base_url = os.getenv(config.env_key_base_url)

        if config.provider == "ollama":
            return OllamaLLM(
                model=config.model_name,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature or config.temperature,
                **kwargs
            )
        else:
            if not api_key:
                raise ValueError(f"模型 '{model_name}' 缺少 API_KEY")
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=config.model_name,
                temperature=temperature or config.temperature,
                max_tokens=max_tokens or config.max_tokens,
                **kwargs
            )

    def list_available_models(self) -> list:
        """列出所有可用的模型"""
        return list(MODELS_CONFIG.keys())

    def clear_cache(self):
        """清空缓存"""
        self._llm_cache.clear()


# 全局工厂实例
llm_factory = LLMFactory()
