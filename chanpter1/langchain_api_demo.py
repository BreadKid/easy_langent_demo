# 1. 导入模块
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

# 2. 初始化大模型（通过工厂获取，无需手动配置）
# 支持的模型: 'deepseek', 'claude', 'ollama'
# 或使用自定义参数：llm_factory.get_llm_with_custom_params("deepseek", temperature=0.5)
llm_openai = llm_factory.get_llm("deepseek")

# 5. 构造 Prompt（教学阶段用字符串更直观）
prompt = "请写一段50字左右的 LangChain学习建议，语言简洁、实用，适合初学者。"

# 6. 调用模型
response = llm_openai.invoke(prompt)

# 7. 输出结果
print("生成的学习建议：")
print(response.content)