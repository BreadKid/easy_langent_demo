import sys
from pathlib import Path
# 导入PromptTemplate
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import llm_factory

# 初始化大模型
llm = llm_factory.get_llm_with_custom_params("deepseek", temperature=0.3, max_tokens=200)

# 1. 定义提示词模板
examples = [
    {
        "subject": "Python编程",
        "method": "核心目标：掌握基础语法和常用库；学习步骤：1. 学习变量、函数等基础语法 2. 实操小项目（如计算器） 3. 学习Pandas、Matplotlib库；注意事项：多动手实操，遇到错误及时调试。"
    },
    {
        "subject": "机器学习",
        "method": "核心目标：理解基础算法原理和应用场景；学习步骤：1. 复习数学基础（线性代数、概率） 2. 学习经典算法（线性回归、决策树） 3. 用Scikit-learn实操；注意事项：先理解原理，再动手实现，避免死记硬背。"
    }
]
# input_variables：动态参数列表（这里是user_role和subject）
example_template = """
学科：{subject}
学习方法：{method}
"""
# template：提示词模板字符串，用{参数名}表示动态参数
prompt_template = PromptTemplate(
    input_variables=["subject", "method"],
    template=example_template
)
few_shot_prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = prompt_template,
    suffix = "学科：{new_subject}\n学习方法：",
    input_variables = ["new_subject"]
)

# 2. 格式化模板（传入具体参数，生成完整提示词）
# 给“高校学生”生成“LangChain”学习建议
# formatted_prompt = prompt_template.format(
#     user_role="开发者",
#     subject="AI Agent"
# )
# print("格式化后的提示词：")
# print(formatted_prompt)
formatted_prompt = few_shot_prompt.format(new_subject="LangChain")
print("格式化后的Few-Shot提示词：")
print(formatted_prompt)


# 3. 调用模型生成结果
result = llm.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的内容")
print(result.content)