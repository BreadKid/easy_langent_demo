#文本输出
import sys
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

llm=llm_factory.get_llm_with_custom_params("deepseek", temperature=0.3)

#创建格式化对象
parsor=StrOutputParser()

#调用链
chain=llm|parsor
result=chain.invoke("请用一句话介绍一下什么是LangChain？")

print("生成的类型：\n"+type(result).__name__)
print("\n生成的内容：\n"+result)