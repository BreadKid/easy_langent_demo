import os
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config.models_config import MODELS_CONFIG


class LLMFactory:
    """LLM模型工厂 - 统一管理多个模型的生命周期"""

    def __init__(self, env_file: str = ".env"):
        """初始化工厂"""
        # 确保加载的是项目根目录的.env
        env_path = Path(__file__).parent.parent / env_file
        load_dotenv(env_path)
        self._llm_cache: Dict[str, ChatOpenAI] = {}  # 缓存已初始化的模型
        self._load_and_validate()

    def _load_and_validate(self):
        """验证所有配置的环境变量是否存在"""
        missing_configs = []
        for model_name, config in MODELS_CONFIG.items():
            api_key = os.getenv(config.env_key_api)
            if not api_key:
                missing_configs.append(
                    f"  {model_name}: 缺少 {config.env_key_api}"
                )

        if missing_configs:
            print("⚠️  警告 - 以下模型缺少配置：\n" + "\n".join(missing_configs))

    def get_llm(self, model_name: str = "deepseek") -> ChatOpenAI:
        """
        获取已初始化的LLM模型（单例模式，缓存重用）

        Args:
            model_name: 模型名称 (claude, deepseek, ollama)

        Returns:
            ChatOpenAI 实例

        Raises:
            ValueError: 如果模型未配置或缺少API密钥
        """
        # 如果已缓存，直接返回
        if model_name in self._llm_cache:
            return self._llm_cache[model_name]

        # 获取模型配置
        if model_name not in MODELS_CONFIG:
            available = ", ".join(MODELS_CONFIG.keys())
            raise ValueError(
                f"未知的模型: {model_name}\n"
                f"可用模型: {available}"
            )

        config = MODELS_CONFIG[model_name]

        # 从环境变量读取凭证
        api_key = os.getenv(config.env_key_api)
        base_url = os.getenv(config.env_key_base_url)

        if not api_key:
            raise ValueError(
                f"模型 '{model_name}' 的 API_KEY 未配置\n"
                f"请检查 .env 文件中的 {config.env_key_api}"
            )

        # 初始化模型
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **(config.extra_params or {})
        )

        # 缓存
        self._llm_cache[model_name] = llm
        print(f"✅ 初始化模型: {model_name}")
        return llm

    def get_llm_with_custom_params(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatOpenAI:
        """
        获取模型并覆盖部分参数（不缓存，每次创建新实例）

        Args:
            model_name: 模型名称
            temperature: 自定义温度参数
            max_tokens: 自定义最大token数
            **kwargs: 其他参数

        Returns:
            ChatOpenAI 实例
        """
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"未知的模型: {model_name}")

        config = MODELS_CONFIG[model_name]
        api_key = os.getenv(config.env_key_api)
        base_url = os.getenv(config.env_key_base_url)

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
        """清空缓存（用于切换环境或重新加载配置）"""
        self._llm_cache.clear()


# 全局工厂实例
llm_factory = LLMFactory()
