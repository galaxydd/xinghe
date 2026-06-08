from nltk import accuracy

from evaluation.utils import read_json, normalize_zh_answer, is_dropout

"""
Task:  法律焦点分类
Metric: Accuracy
Class: Single
"""
#   类别列表
categories = ["交通事故", "医疗纠纷", "婚姻家庭", "劳动纠纷",
              "债权债务", "刑事辩护", "合同纠纷", "房产纠纷"]
def evaluate_legal_identification(data_file):

    data_dict = read_json(data_file)
    correct = 0
    dropout = 0
    total = 0
    for i, entry in enumerate(data_dict):
        dropout_flag, prediction = is_dropout(entry["prediction"])
        if dropout_flag:
            dropout += 1
            continue
        reference = entry["reference"].strip()
        predicted_category = None
        for category in categories:
            if category in prediction:
                predicted_category = category
                break  # 找到第一个匹配的类别就停止

        # 计算正确预测数
        correct += int(reference == predicted_category)
        total+=1

    accuracy = correct / max(total, 1)
    dropout_prob = dropout / len(data_dict)
    print(f"Accuracy: {accuracy:.2%}\nDropout probability: {dropout_prob:.2%}")



if __name__ == "__main__":

    #evaluate_legal_identification("../../prediction_result/zero-shot/glm-4-9b-chat/legal_focus_identification.json")
    #evaluate_legal_identification("../../prediction_result/zero-shot/baichuan-7b/legal_focus_identification.json")
    evaluate_legal_identification("../../prediction_result/zero-shot/deepseek-v3/legal_focus_identification.json")
