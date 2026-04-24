'''
使用封装后的 Redis 记忆管理器的 Demo
'''
from langchain_core.runnables.history import RunnableWithMessageHistory
from core.memory_manager import get_redis_history  # 导入封装后的函数
from memory_prompt import prompt
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory


# 初始化 LLM
llm = llm_factory.get_llm()

# 1. 构建基础链
base_chain = prompt | llm

# 2. 构建带全量记忆的对话链
# 直接使用导入的 get_redis_history 即可
full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_redis_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

# 测试多轮对话
config = {"configurable": {"session_id": "user_redis_001"}}

print("--- 开始对话 (使用 core.memory_manager 封装) ---")
# 第一轮
response1 = full_memory_chain.invoke({"user_input": "我叫大白，我的爱好是修机器人"}, config=config)
print(f"助手回复1：{response1.content}")

# 第二轮
response2 = full_memory_chain.invoke({"user_input": "我刚才说我的名字是什么？"}, config=config)
print(f"助手回复2：{response2.content}")

# 查看历史
print("\nRedis 中的完整历史：")
for msg in get_redis_history("user_redis_001").messages:
    print(f"[{msg.type}]: {msg.content}")
