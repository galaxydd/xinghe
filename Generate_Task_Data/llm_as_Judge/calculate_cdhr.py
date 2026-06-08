import os
import json

RESULTS_DIR = "../diagnostic_results"

HALLUCINATION_KEYS = [
    "H1_Outdated_Provisions",
    "H2_Irrelevant_Citations",
    "H3_Contradictory_Advice",
    "H4_Fabricated_Provisions",
    "H5_Incorrect_Article_Numbers"
]

# 映射文件名到四大客观矛盾场景
SCENARIO_MAPPING = {
    "legal_timeliness.json": "C1_法律规范矛盾",
    "legal_recitation.json": "C2_内部逻辑矛盾",
    "legal_summary.json": "C3_法律事实矛盾",
    "legal_open_qa.json": "C4_解释主体矛盾"
}


def load_json_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return data

        if content.startswith('['):
            data = json.loads(content)
        else:
            # 如果是上一步改写成的 JSONL 逐行写入格式
            for line in content.split('\n'):
                if line.strip():
                    data.append(json.loads(line))
    return data


def calculate_cdhr():
    if not os.path.exists(RESULTS_DIR):
        print(f"[错误] 找不到结果目录: {RESULTS_DIR}")
        print(f"当前工作目录是: {os.getcwd()}")
        return

    # 用于存储最终 4x5 矩阵的数据
    cdhr_matrix = {}
    for scenario in SCENARIO_MAPPING.values():
        cdhr_matrix[scenario] = {k: 0 for k in HALLUCINATION_KEYS}

    print("==================================================")
    print("📊 开始计算 CDHR (矛盾驱动的幻觉触发率)...")
    print("==================================================\n")

    # 使用 os.walk 遍历可能存在的多级子目录（模型文件夹）
    for root, dirs, files in os.walk(RESULTS_DIR):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(root, filename)
            model_name = os.path.basename(root)  # 获取模型文件夹的名称

            # 根据文件名匹配场景
            scenario_name = SCENARIO_MAPPING.get(filename)
            print(scenario_name)
            if not scenario_name:
                print(f"[跳过] 未知任务文件: {filename}")
                continue

            print(f"正在分析: [{model_name}] 的 {filename}...")

            # 加载数据
            data = load_json_data(filepath)
            total_valid = 0
            h_counts = {k: 0 for k in HALLUCINATION_KEYS}

            # 遍历每一条诊断数据进行统计
            for item in data:
                diagnosis = item.get("CDHR_Diagnosis")
                if isinstance(diagnosis, str):
                    try:
                        clean_str = diagnosis.replace("```json", "").replace("```", "").strip()
                        diagnosis = json.loads(clean_str)
                    except json.JSONDecodeError:
                        print(f"[警告] 数据格式异常无法解析，已跳过。内容片段: {diagnosis[:50]}...")
                        continue

                if not diagnosis or not isinstance(diagnosis, dict):
                    continue


                total_valid += 1
                for key in HALLUCINATION_KEYS:
                    # 兼容 Python 原生 True 以及被转成字符串的 "true"
                    val = diagnosis.get(key)
                    if val is True or str(val).lower() == 'true':
                        h_counts[key] += 1

                # 计算触发率百分比
            if total_valid > 0:
                # 确保矩阵的这个场景已经被初始化为一个字典
                if scenario_name not in cdhr_matrix:
                    cdhr_matrix[scenario_name] = {}

                for key in HALLUCINATION_KEYS:
                    rate = (h_counts[key] / total_valid) * 100
                    cdhr_matrix[scenario_name][key] = round(rate, 2)

                print(f"  └─ ✅ 分析完成 | 有效样本: {total_valid} 条\n")
            else:
                print(f"  └─ ⚠️ 无有效诊断数据\n")


    print("\n4x5 CDHR 评估矩阵结果 (%)")
    print("| 矛盾场景 | H1:引用失效 | H2:无关引用 | H3:自相矛盾 | H4:捏造法条 | H5:编号错误 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for scenario in ["C1_法律规范矛盾", "C2_内部逻辑矛盾", "C3_法律事实矛盾", "C4_解释主体矛盾"]:
        rates = cdhr_matrix.get(scenario, {k: 0 for k in HALLUCINATION_KEYS})
        row = f"| {scenario} | {rates['H1_Outdated_Provisions']}% | {rates['H2_Irrelevant_Citations']}% | {rates['H3_Contradictory_Advice']}% | {rates['H4_Fabricated_Provisions']}% | {rates['H5_Incorrect_Article_Numbers']}% |"
        print(row)

    return cdhr_matrix



if __name__ == "__main__":
    matrix_data = calculate_cdhr()