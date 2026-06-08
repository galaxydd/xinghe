import csv
import json

import csv
import json
import os.path
import random

def process_csv(input_file:str ,output_file: str):
    # 读取CSV文件并分类存储
    constitution = []      # 存储宪法
    laws = []               # 存储所有法律（"法律"类）
    admin_regulations = []  # 存储所有行政法规（"行政法规"类）

    with open(input_file, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            category = row.get('法律性质', '').strip()  # 分类列名为"法律性质"
            if category == '宪法':
                constitution.append(row)
            elif category == '法律':
                laws.append(row)
            elif category == '行政法规':
                admin_regulations.append(row)


    # 随机筛选条法律法规
    selected_laws = random.sample(laws, min(100, len(laws)))
    selected_admins = random.sample(admin_regulations,min(49,len(admin_regulations)))
    # 合并所有需要处理的数据
    # 宪法1条  法律 100条  地方性法规49条
    all_data = constitution + selected_laws + selected_admins

    print(f"总计生成 {len(all_data)} 条有效的的法律条文")
    # 构建JSON结构
    json_data = []
    for item in all_data:
        entry = {
            "instruction": "下面是一个法律或法规，判断当前的时效性类别。包括有效、已废止、尚未生效，例如[类别]有效<eoa>请严格按照这个格式回答。法律： ",
            "question": item.get('信息', '').strip(),  # 假设名称列名为"信息"
            "answer": f"回答:{item.get('时效性', '有效').strip()}"
        }
        json_data.append(entry)


    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"JSON文件已保存至：{output_file}")



if __name__ == "__main__":
    file_path = './result'
    input_file = os.path.join(file_path,'valid.csv')
    process_csv(input_file,'./result_json/valid.json')

