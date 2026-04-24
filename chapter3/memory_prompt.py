from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# 1. 定义提示词模板（与全量记忆通用，可复用）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于最近的对话历史回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])