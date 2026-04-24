import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

llm=llm_factory.get_llm()

# 场景：查订单
order_prompt=ChatPromptTemplate.from_messages([
    ("system","你是智能客服，负责解答用户的订单查询问题。"),
    ("human","用户问题：{query}\n请引导用户提供订单号，并告知查询流程：1. 提供订单号；2. 系统验证；3. 反馈订单状态。")
])
order_chain=order_prompt|llm|StrOutputParser()

# 场景：退货
refund_prompt=ChatPromptTemplate.from_messages([
    ("system","你是智能客服，负责解答用户退货的问题。"),
    ("human","用户问题：{query}\n请说明退款流程：1. 申请退款（订单页面点击退款）；2. 等待审核（1-3个工作日）；3. 退款到账（原路返回，3-5个工作日）。如果用户问退款进度，引导提供退款申请单号。")
])
refund_chain=refund_prompt|llm|StrOutputParser()

# 场景：保修
warranty_prompt=ChatPromptTemplate.from_messages([
    ("system","你是智能客服，负责解答用户报修的问题。"),
    ("human","用户问题：{query}\n请说明保修政策：本产品保修期限为1年，保修范围包括质量问题（非人为损坏），保修流程：1. 联系客服；2. 提供购买凭证；3. 寄回检测维修。")
])
warranty_chain=warranty_prompt|llm|StrOutputParser()

# 场景：默认 无法回答
default_prompt=ChatPromptTemplate.from_messages([
    ("system","你是默认回答生成器，当用户问题无法归类到任何特定场景时，提供通用帮助信息。"),
    ("human","用户问题{query}\n请生成合适的回复。")
])
default_chain=default_prompt|llm|StrOutputParser()

# 路由判断
router_prompt=ChatPromptTemplate.from_messages([
    ("system","""
    你是路由选择器，需根据用户问题判断所属场景，仅输出以下标准化标识之一：
- order：订单查询相关（含订单状态、订单号）
- refund：退货款相关（含退款进度、退款申请）
- warranty：保修相关（含维修、售后保障）
- default：以上均不匹配
无需输出任何其他内容，仅返回标识字符串。
"""),
    ("human","用户问题{query}")
])
router_chain=router_prompt|llm|StrOutputParser()

# 路由分发
all_chain=RunnableLambda(lambda x: x)|(RunnableBranch(
    (lambda x: x["scene"]=="order",order_chain),
    (lambda x: x["scene"]=="refund",refund_chain),
    (lambda x: x["scene"]=="warranty",warranty_chain),
    default_chain
)).with_config(run_name="all_chain")

# 封装整个逻辑
def process_query(query:str):
    # 获取场景
    scene=router_chain.invoke({"query":query})
    # 传入参数并分发处理
    response=all_chain.invoke({"query":query,"scene":scene})
    return response

# 测试调用
test_query=[
    "我的订单什么时候发货？",
    "怎么申请退款呀？",
    "这个产品保修多久？",
    "你们家有什么新品？"  # 无法匹配，触发默认链
]

for q in test_query:
    print(f"用户问题：{q}")
    print("客服回复：", process_query(q))
    print("-"*50)