import sys
from pathlib import Path
# 导入PromptTemplate
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_core.example_selectors import BaseExampleSelector
from typing import List, Dict
import json

# 添加项目根目录到 Python 路径（允许从任何地方运行此脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import llm_factory

# 初始化大模型
llm=llm_factory.get_llm_with_custom_params("deepseek", temperature=0.3, max_tokens=300)
#读取json数据
json_path = Path(__file__).parent / "learning_method_examples.json"
with open(json_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

class ExampleSelector(BaseExampleSelector):
    def __init__(self,examples:List[Dict[str,str]]):
        self.examples=examples

    def add_example(self,example:Dict[str,str]) -> None:
        self.examples.append(example)

    def select_examples(self,input_varibles:Dict[str,str]) ->List[Dict]:
        # 输入难度，默认easy
        target_difficulty=input_varibles.get("difficulty","easy")
        #过滤匹配内容
        return [ex for ex in self.examples if ex.get("difficulty")==target_difficulty]
#使用自定义匹配选择器
example_selector=ExampleSelector(examples=examples)

#少样本模板
few_shot_prompt=FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["subject","difficulty","method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
        ),
    example_separator="\n",
    prefix="少样本提示：",
    suffix="参考以上示例，回答：\n学科：{new_subject}\n难度：{difficulty}\n学习方法：",
    input_variables=["new_subject","difficulty"]
)

#提示词模板生成
formatted_prompt=few_shot_prompt.format(new_subject="LangChain",difficulty="easy")
print("提示词：\n"+formatted_prompt)

#调用模型生成结果
response=llm.invoke([{"role":"user","content":formatted_prompt}])
print("\n生成的内容：\n"+response.content)
