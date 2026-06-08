import os
import json

def merge_json_files(input_dir: str, output_file: str):
    """
    将 input_dir 目录下所有 .json 文件中的数组内容合并到一个列表中，并输出到 output_file。
    """
    merged_data = []

    # 遍历目录下的所有 JSON 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 如果文件内容是列表，则将其合并到 merged_data 中
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"警告：{file_name} 中的数据不是列表，已跳过。")

    # 将合并后的数据写入 output_file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(merged_data, out_f, ensure_ascii=False, indent=4)

    print(f"合并完成，共合并 {len(merged_data)} 条记录。结果已保存到 {output_file}")


if __name__ == "__main__":
    input_directory = "result_filter"        # 需要合并的 JSON 文件所在目录
    output_file = "legal_recitation.json"    # 合并后的输出文件
    merge_json_files(input_directory, output_file)
