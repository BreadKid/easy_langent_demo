# 导入Hugging Face的LLM接口
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 加载Hugging Face的模型和tokenizer
model_name = "moonshotai/Kimi-K2.6"  # 模型名 这是本地路径 这里替换成自己的路径！！
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 构建pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.3
)

# 3. 初始化LangChain的LLM接口（统一接口）
hf_llm = HuggingFacePipeline(pipeline=pipe)

# 4. 调用模型（和之前的LLM调用方式完全一样）
prompt = "请用3句话解释什么是LangChain？"
result = hf_llm.invoke(prompt)

print("Hugging Face模型回复：")
print(result)