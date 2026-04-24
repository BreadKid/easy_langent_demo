import sys
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory


# 初始化 LLM
llm = llm_factory.get_llm()

# 组件1:提取卖点
# prompt
prompt1=PromptTemplate(
    input_variables=["product_intro"],
    template="从产品介绍中提取2个最大的卖点，用简洁语言列出{product_intro}"
)
# chain
chain1=prompt1|llm

# 格式化结果
product_intro_info=RunnableLambda(lambda info:{"sell_points": info})

# 组件2:生成话术
# prompt
prompt2=PromptTemplate(
    input_variables=["sell_points"],
    template="根据以下卖点生成一段吸引人的话术：{sell_points}"
)
# chain
chain2=prompt2|llm

# 串联组件
all_chain=chain1|product_intro_info|chain2

# 测试调用
product_intro="这款智能手表具有心率监测和睡眠分析功能，适合运动爱好者和健康管理者。"
response=all_chain.invoke({"product_intro": product_intro})
print("生成的话术：", response)