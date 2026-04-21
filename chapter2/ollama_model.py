from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma4-e4b_local:v1", base_url="http://localhost:11434")
response = llm.invoke("请用3句话解释什么是LangChain？")
print("Ollama模型回复：")
print(response)
