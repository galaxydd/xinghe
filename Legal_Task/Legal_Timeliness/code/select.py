import json
import pandas as pd

# 读取 JSON 文件
with open("../result_json/valid.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 提取 JSON 文件中的 "question" 字段
json_questions = {entry["question"] for entry in json_data}

# 读取 CSV 文件（假设 CSV 文件中有 "question" 和 "时效性" 等字段）
csv_data = pd.read_csv("../result/valid.csv")

# 筛选出 CSV 文件中 JSON 文件没有的条目
missing_data = csv_data[~csv_data["信息"].isin(json_questions)]

# 如果只需要部分信息，比如只保留 "question" 和 "时效性" 字段
selected_columns = ["信息", "时效性"]
partial_data = missing_data[selected_columns]

# 保存结果到新的 CSV 文件
output_file = "t.csv"
partial_data.to_csv(output_file, index=False, encoding="utf-8")

print(f"已保存部分信息到 {output_file}")