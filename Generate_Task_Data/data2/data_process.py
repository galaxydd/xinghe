import json
import random


def convert_format_random(input_filename, output_filename, sample_size=500):
    # 1. 读取原始的所有数据
    with open(input_filename, 'r', encoding='utf-8') as f:
        # 去除空白行并把每一行存入列表
        lines = [line.strip() for line in f if line.strip()]

    # 2. 如果数据总量大于 sample_size，则随机抽取 500 条；否则全部保留
    if len(lines) > sample_size:
        sampled_lines = random.sample(lines, sample_size)
    else:
        sampled_lines = lines

    converted_data = []

    # 3. 遍历随机抽取的这些行，解析并转换格式
    for line in sampled_lines:
        item = json.loads(line)

        # 构建新的字典，将原有的 'input' 映射为 'question'
        new_item = {
            "instruction": item.get("instruction", ""),
            "question": item.get("input", ""),
            "answer": item.get("answer", "")
        }
        converted_data.append(new_item)

    # 4. 将转换后的列表作为一个完整的 JSON 数组写入新文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文字符正常显示，indent=4 提供良好的缩进格式
        json.dump(converted_data, f, ensure_ascii=False, indent=4)

    print(f"转换完成！共随机抽取并转换了 {len(converted_data)} 条数据，文件已保存至 {output_filename}")


if __name__ == "__main__":
    # 指定输入和输出文件名
    input_file = './legal_inner/mulit_hop.json'
    output_file = 'legal_inner/mulit_hop.json'

    # 执行转换，指定只随机抽取 500 条
    convert_format_random(input_file, output_file, sample_size=500)