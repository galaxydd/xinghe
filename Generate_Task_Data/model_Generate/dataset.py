import json

def read_json(input_file):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = []
    for item in data:
        dataset.append({
            "instruction": item["instruction"],
            "question": item["question"],
            "answer": item["answer"]
        })
    return dataset

if __name__ == "__main__":
    dataset = read_json("../data/legal_question_classification.json")
    for i, entry in enumerate(dataset):
        print(f"Entry {i + 1}:")
        print(f"  Instruction: {entry['instruction']}")
        print(f"  Question: {entry['question']}")
        print(f"  Answer: {entry['answer']}")
        print("-" * 50)

