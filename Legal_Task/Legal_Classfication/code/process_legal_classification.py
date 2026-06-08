import json
import jsonlines
import random


def process_legal_question_classfication(input_file: str, output_file: str, sample_size: int = 500):
    json_data = []
    subjects = set()
    with jsonlines.open(input_file, mode='r') as reader:
        for obj in reader:
            subjects.add(obj.get("subject", "未知"))



    # 随机抽取 其中sample_size 个数据作为任务数据集
    sampled_data = random.sample(json_data, min(sample_size, len(json_data)))


    print(f"JSON文件已保存至：{output_file}")
    print(f"共有 {len(subjects)} 种不同的 subject 类别：{subjects}")

if __name__ == "__main__":
    input_file = '../JEC_QA/0_train.json'
    output_file = '../legal_question_classfication.json'

    process_legal_question_classfication(input_file, output_file, sample_size=500)