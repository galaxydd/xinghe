import json

import json

# 读取 JSON 文件
with open("../legal_summary_whole.json", "r", encoding="utf-8") as infile:
    data = json.load(infile)

# 确保数据是列表
if not isinstance(data, list):
    raise ValueError("JSON 数据应为列表格式")

# 按 question + answer 的总长度排序（从短到长）
sorted_data = sorted(data, key=lambda x: len(x.get("question", "") + x.get("answer", "")))

# 取最短的 10 条记录
shortest_ten = sorted_data[:10]

# 生成新的 JSON 数据
result = [{"question": item.get("question", ""), "answer": item.get("answer", "")} for item in shortest_ten]

# 保存到新的 JSON 文件
with open("../output.json", "w", encoding="utf-8") as outfile:
    json.dump(result, outfile, ensure_ascii=False, indent=4)

print("✅ 任务完成：已将最短的 10 条记录按长度排序后保存到 output.json")
