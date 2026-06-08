import pandas as pd
import json
import os

# 读取xls文件
df = pd.read_excel('D:\law\lawzhidao_best_reply.xls')

# 假设列标题为 'question' 和 'reply'
result = []

# 遍历每一行并提取 question 和 reply
for index, row in df.iterrows():
    instruction = row['question']
    answer = row['reply']

    # 创建一个字典并添加到结果列表
    result.append({
        "instruction": instruction,
        "input": "",
        "answer": answer
    })

# 将结果转换为 JSON 格式
json_result = json.dumps(result, ensure_ascii=False, indent=2)


# 指定导出路径
output_dir = r'D:\law'  # 请根据需要修改为你的目录
output_file = os.path.join(output_dir, 'lawzhidao_best_reply.json')

# 确保目标目录存在
os.makedirs(output_dir, exist_ok=True)

# 保存为 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_result)



print("JSON 文件已成功导出！")