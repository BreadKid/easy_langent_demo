# 1. 导入需要的模块
import sys
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

# 2. 初始化大模型（通过工厂获取）
llm = llm_factory.get_llm("deepseek")

# 5. 定义 State
class WorkflowState(TypedDict, total=False):
    user_role: str  # 存储用户角色
    original_advice: str  # 存储原始学习建议
    simplified_advice: str  # 存储精简后的建议
    translated_advice: str  # 存储翻译后的建议

# 6. 定义节点
# 6.1 原始学习建议
def generate_advice(state: WorkflowState):
    prompt = f"给{state['user_role']}写一段50字左右的 AI 学习建议。"
    result = llm.invoke(prompt)
    return {"original_advice": result.content}

# 6.2 精简学习建议
def simplify_advice(state: WorkflowState):
    prompt = f"把下面的学习建议精简到30字以内：{state['original_advice']}"
    result = llm.invoke(prompt)
    return {"simplified_advice": result.content}

# 6.3 翻译
def translated_advice(state: WorkflowState):
    prompt = f"把学习建议翻译成英文：{state['simplified_advice']}"
    result = llm.invoke(prompt)
    return {"translated_advice": result.content}

# 7. 构建工作流
workflow = StateGraph(WorkflowState)

workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)
workflow.add_node("translated_advice", translated_advice)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", "translated_advice")
workflow.add_edge("translated_advice", END)

app = workflow.compile()

# 8. 执行
result = app.invoke({"user_role": "高校学生"})

# 9. 输出
print("原始学习建议：")
print(result["original_advice"])
print("\n精简后学习建议：")
print(result["simplified_advice"])
print("\n英文版：")
print(result["translated_advice"])