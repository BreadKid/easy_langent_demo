#输出基类
import sys
from pathlib import Path
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

llm = llm_factory.get_llm("deepseek")

#自定义parser
class CustomToolParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        """将模型输出按 '工具名@核心功能@学习难度' 解析为字典"""
        text = text.strip().replace("\n", "").replace(" ", "")
        parts = text.split("@")
        if len(parts) != 3:
            raise ValueError(f"输出格式错误！需满足「工具名@核心功能@学习难度」，当前输出：{text}")
        return {
            "tool_name": parts[0].strip(),
            "function": parts[1].strip(),
            "difficulty": parts[2].strip()
        }

    def get_format_instructions(self) -> str:
        """生成提示词，引导模型按自定义格式输出"""
        return "请严格按照「工具名@核心功能@学习难度」格式输出，不添加多余内容。示例：LangSmith@全链路调试监控@中等"
    
parser=CustomToolParser()

prompt=PromptTemplate(
    template="请介绍1个LangChain开发工具。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain=prompt|llm|parser
response=chain.invoke({})

print("自定义解析器解析结果：")
print(response)
print("解析结果类型：", type(response).__name__)