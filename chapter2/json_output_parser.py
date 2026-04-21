#json输出，便于业务处理
import sys
from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

llm = llm_factory.get_llm_with_custom_params("deepseek", temperature=0.3)

#创建prompt
prompt=PromptTemplate(template="请介绍1个LangChain开发工具，输出工具名和核心功能。{{format_instructions}}")

#创建parsor
parsor=JsonOutputParser()

#调用链
chain=prompt|llm|parsor
response=chain.invoke({})

print("生成的内容：\n",response)
print("指定字段：",response.get("tool_name",None))