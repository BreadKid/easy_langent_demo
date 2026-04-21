#自定义模型输出
import sys
from pathlib import Path
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

# 定义输出数据模型
class SpecInfo(BaseModel):
    tool_name:str=Field(description="工具名称")
    function:str=Field(description="核心功能")
    difficulty:str=Field(description="学习难度，分为简单、一般、困难")

# 定义输出解析器
parser=PydanticOutputParser(pydantic_object=SpecInfo)

# 定义提示词模板
prompt = PromptTemplate(
    template="{user_input}，严格按照要求输出。\n{format_instructions}",
    input_variables=["user_input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

#LLM
llm = llm_factory.get_llm_with_custom_params("deepseek", temperature=0.3)

#调用链
chain=prompt|llm|parser
response=chain.invoke({"user_input":"请介绍1个LangChain开发工具，输出工具"})

print("生成的内容：\n",response)
print("\n指定字段function=",response.function)
print("\n转成字典：", response.model_dump())