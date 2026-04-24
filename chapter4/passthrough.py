import sys
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap,RunnablePassthrough

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

llm=llm_factory.get_llm()

prompt1=PromptTemplate(
    input_variables=["product_intro"],
    template="从产品介绍中提取2个最大的卖点，用简洁语言列出{product_intro}"
)
prompt2=PromptTemplate(
    input_variables=["sell_points","target_audience"],
    template="根据以下卖点和目标受众生成一段吸引人的话术(100字以内)：{sell_points}，目标受众：{target_audience}"
)

all_chain=(
    # 步骤1卖点
    RunnableMap({
        "sell_points":prompt1|llm|(lambda x:x),
        "target_audience":RunnablePassthrough()
    })

    # 步骤2话术
    |prompt2|llm
)

# 测试调用
input={"product_intro":"这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟",
       "target_audience":"大学生群体（喜欢运动、预算有限、注重性价比）"}

response=all_chain.invoke(input)
print("生成的话术：", response)