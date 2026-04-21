"""
项目入口 - main.py
在项目根目录运行: python main.py
"""

from chapter2 import LangChainChatBot


def main():
    """主函数"""
    # 示例1: 使用默认配置的DeepSeek
    print("=" * 50)
    print("示例1: DeepSeek")
    print("=" * 50)
    bot = LangChainChatBot(model_name="deepseek")
    response = bot.chat("请用3句话解释什么是LangChain？")
    print(f"回复：\n{response}\n")

    # 示例2: 使用Claude
    print("=" * 50)
    print("示例2: Claude")
    print("=" * 50)
    bot = LangChainChatBot(model_name="claude")
    response = bot.chat("什么是大语言模型？")
    print(f"回复：\n{response}\n")


if __name__ == "__main__":
    main()
