import os
import json


def extract_questions_and_answers(input_dir, output_file):
    """
    从指定目录读取所有 JSON 文件，提取 question 和 candidate_answer 中第一个答案，
    并将结果保存为新的 JSON 文件。

    参数：
    - input_dir: JSON 文件所在的目录路径
    - output_name: 结果保存的输出文件路径
    """
    # 用于存储所有结果
    results = []

    # 遍历目录中的所有 JSON 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)

            # 打开并加载 JSON 文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 获取 question 和第一个 candidate_answer
                question = data.get('question')
                candidate_answer = data.get('candidate_answer', [])

                if question and candidate_answer:
                    # 只提取第一个答案
                    answer = candidate_answer[0]

                    # 格式化成新的结构并加入结果列表
                    result = {
                        "instruction": question,
                        "input": "",
                        "answer": answer
                    }
                    results.append(result)

    # 将结果转换为 JSON 格式
    json_result = json.dumps(results, ensure_ascii=False, indent=4)

    # 保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_result)

    print(f"Processed {len(results)} questions and answers.")

# 替换为你的文件夹路径
input_dir = r'D:\law\laws_data\test'
output_dir = r'D:\law\process_Data'

# 输出的统一 JSON 文件路径
output_file = os.path.join(output_dir, 'lawzhidao2.3w_test.json')
extract_questions_and_answers(input_dir, output_file)