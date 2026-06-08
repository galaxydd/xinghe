import json
import jsonlines
import random
from PyQt5.QtCore import QLineF


def process_legal_concept(input_file: str, output_file: str, sample_size: int = 500):
    json_data = []
    with jsonlines.open(input_file, mode='r') as reader:
        for obj in reader:
            option_A = "A: " + obj["option_list"]["A"] + "\n"
            option_B = "B: " + obj["option_list"]["B"] + "\n"
            option_C = "C: " + obj["option_list"]["C"] + "\n"
            option_D = "D: " + obj["option_list"]["D"]
            question = obj["statement"] + "\n" + option_A + option_B + option_C + option_D

            answer_str = "".join(obj["answer"])

            formatted_obj = {
                "instruction": "请阅读一下法律概念选择题，给出正确选项，不需要做出解释。只给出答案。例如：AB\n",
                "question": question,
                "answer": answer_str
            }
            json_data.append(formatted_obj)

    # 随机抽取 其中sample_size 个数据作为任务数据集
    sampled_data = random.sample(json_data, min(sample_size, len(json_data)))

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    print(f"JSON文件已保存至：{output_file}")


if __name__ == "__main__":
    input_file = './JEC_QA/0_train.json'
    output_file = 'legal_concept_concise.json'

    process_legal_concept(input_file, output_file, sample_size=500)