import sys
from pathlib import Path

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core import llm_factory

# 初始化llm
class LangChainChatBot:
    
    def __init__(self, model_name: str = "deepseek", system_prompt: str = None):
        # 指定模型
        self.llm = llm_factory.get_llm(model_name)
        # 系统提示词
        self.system_prompt = system_prompt or "你是一个耐心的AI学习助手，回复简洁易懂，适合高中生理解。"

    # 对话
    def chat(self, user_input: str) -> str:
        # 消息内容
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        # 调用LLM
        result = self.llm.invoke(messages)
        #结果返回
        return result.content

# ============ 业务代码示例 ============

if __name__ == "__main__":
    # LLM 单次生成
    # bot = LangChainChatBot(model_name="deepseek")
    # response = bot.chat("请用3句话解释什么是LangChain？")
    # print("ChatModel回复：")
    # print(response)

    # chat多轮对话
    llm = llm_factory.get_llm("deepseek")
    history = [
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"}
    ]

    """第一轮对话"""
    history.append({"role": "user", "content": "请用3句话解释什么是LangChain？"})
    response = llm.invoke(history)
    print("【第一轮对话回复】："+response.content)

    # 前文结果加入
    history.append({"role": "assistant", "content": response.content})

    """第二轮对话"""
    history.append({"role": "user", "content": "核心组件哪些？只要组件名字就行，不要解释。"})
    response = llm.invoke(history)
    print("【第二轮对话回复】："+response.content)

    # 前文结果加入
    history.append({"role": "assistant", "content": response.content})

    """第三轮对话"""
    history.append({"role":"user","content":"给一个简单场景"})
    response = llm.invoke(history)
    print("【第三轮对话回复】："+response.content)