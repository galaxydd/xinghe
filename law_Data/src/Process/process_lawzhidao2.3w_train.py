import os
import json

def extract_questions_and_answers(input_dir,output_file):

    result = []
    # 遍历指定目录中的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # 只处理 JSON 文件
        if os.path.isfile(file_path) and filename.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 读取 JSON 文件
                    data = json.load(file)

                    # 假设每个 JSON 文件中都包含 'question' 和 'answer' 字段
                    if 'question' in data and 'answer' in data:
                        result.append({
                            'instruction': data['question'],
                            'input': "",
                            'answer': data['answer']
                        })
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # 将结果转换为 JSON 格式
    json_result = json.dumps(result, ensure_ascii=False, indent=4)

    # 保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_result)

    print(f"Processed {len(result)} questions and answers.")


# 替换为你的文件夹路径
input_dir = r'D:\law\laws_data\train'
output_dir = r'D:\law\process_Data'

# 输出的统一 JSON 文件路径
output_name = 'lawzhidao2.3w_train.json'
output_file = os.path.join(output_dir, output_name)
extract_questions_and_answers(input_dir,output_file)
