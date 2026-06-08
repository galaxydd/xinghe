import json
import random

from jsonlines import jsonlines

def process_legal_concept(input_file: str, output_file: str, sample_size: int = 500):
    json_data = []
    with jsonlines.open(input_file, mode='r') as reader:
        for obj in reader:

            formatted_obj = {
                "instruction": "根据下面的法律问题,给出相关的法律解释\n",
                "question": obj["input"],
                "answer": obj["output"]
            }
            json_data.append(formatted_obj)

    # 随机抽取 其中sample_size 个数据作为任务数据集
    sampled_data = random.sample(json_data, min(sample_size, len(json_data)))

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    print(f"JSON文件已保存至：{output_file}")
    print(f"最终抽取了 {len(sampled_data)} 条数据")

if __name__ == "__main__":
    input_file = '../DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl'
    output_file = '../Process_json/DISC-Law-QA.json'

    process_legal_concept(input_file, output_file, sample_size=400)