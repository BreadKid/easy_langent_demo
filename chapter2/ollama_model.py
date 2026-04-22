# 原生调用ollama的模型
# from langchain_ollama import OllamaLLM
# llm = OllamaLLM(model="gemma4-e4b_local:v1", base_url="http://localhost:11434")

# 调用封装的工厂模型类
import sys
from pathlib import Path
# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

llm=llm_factory.get_llm("ollama")

response = llm.invoke("请用3句话解释什么是LangChain？")
print("Ollama模型回复：")
print(response)
