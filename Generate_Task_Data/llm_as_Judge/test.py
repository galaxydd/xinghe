
from llm_as_Judge.eva import JUDGE_PROMPT_TEMPLATE
from llm_as_Judge.tool import read_json

data_file = "../prediction_result/zero_shot/deepseek-reasoner/legal_recitation.json"
data_dict = read_json(data_file)


for i, entry in enumerate(data_dict):
    question = entry["instruction"]
    answer =  entry["reference"]
    model_answer = entry["prediction"]
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        model_answer=model_answer)
    print(prompt)

