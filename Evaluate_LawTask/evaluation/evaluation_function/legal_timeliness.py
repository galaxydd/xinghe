import re
from evaluation.utils import read_json, is_dropout

"""
Task:  法律时效性判断
Metric: Accuracy
Class: Single
"""


def evaluate_legal_timeliness(data_file: str):
    """
    读取给定文件中的 JSON 数据，统计正确率和无法提取时效性的数量，
    并打印结果。
    """
    data_dict = read_json(data_file)
    correct = 0  # 预测正确的数量
    total = 0  # 样本总数
    dropout = 0  # 无法提取时效性的数量

    for entry in data_dict:
        # 从 reference 和 prediction 中提取法律时效性类别,并判断dropout
        dropout_flag,  prediction = is_dropout(entry["prediction"],'legal_timeliness')
        if dropout_flag:
            dropout+=1
            continue

        reference = entry["reference"].replace("回答:", "").strip()
        # 计算正确预测数
        correct += int(reference == prediction)
        total+=1
    accuracy = correct / max(total, 1)
    dropout_prob = dropout / len(data_dict)

    # 格式化输出
    print(f"Accuracy: {accuracy:.2%}\nDropout probability: {dropout_prob:.2%}")
    return {"score": accuracy},{"dropout_prob":dropout_prob}
if __name__ == "__main__":

    evaluate_legal_timeliness("../../prediction_result/zero-shot/baichuan-7b/legal_timeliness.json")

    evaluate_legal_timeliness("../../prediction_result/zero-shot/deepseek-v3/legal_timeliness.json")