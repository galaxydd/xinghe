import re

from torchvision.transforms.v2.functional import normalize

from evaluation.evaluation_fuction.legal_timeliness import evaluate_legal_timeliness
from evaluation.utils import read_json, is_dropout, process_legal_concept, normalize_zh_answer, \
    extract_uppercase_letters

"""
Task:  法律时效性判断
Metric: Accuracy
Class: 多分类
"""


#  baichuan-7b模型遵循指令的能力很差，无法按照正常格式进行输出。
#  并且还会出现没有的答案如E：....
def evaluate_legal_concept_concise(data_file: str,model_name=None):
    """
    读取给定文件中的 JSON 数据，统计正确率和无法提取时效性的数量，
    并打印结果。
    """
    data_dict = read_json(data_file)
    correct = 0  # 预测正确的数量
    total = 0  # 样本总数
    dropout = 0  # 无法提取时效性的数量
    for entry in data_dict:
        #从 reference 和 prediction 中提取法律时效性类别,并判断dropout
        dropout_flag = False
        prediction = entry["prediction"]
        #对baichuan-7b特殊处理
        if model_name == 'baichuan-7b':
            dropout_flag,  prediction = is_dropout(entry["prediction"],'legal_concept_concise')
        elif model_name is None:
            prediction = extract_uppercase_letters(normalize_zh_answer(prediction))
            if not prediction:
                dropout_flag = True
        if dropout_flag:
            dropout+=1
            continue
        reference = entry["reference"].strip()
        correct += int(reference == prediction)
        total += 1
    print(total)
    print(dropout)
    accuracy = correct / max(total, 1)
    dropout_prob = dropout / len(data_dict)

    # 格式化输出
    print(f"Accuracy: {accuracy:.2%}\nDropout probability: {dropout_prob:.2%}")
    return {"score": accuracy},{"dropout_prob":dropout_prob}
if __name__ == "__main__":

    #evaluate_legal_concept_concise("../../prediction_result/zero-shot/baichuan-7b/legal_concept_concise.json","baichuan-7b")
    evaluate_legal_concept_concise("../../prediction_result/zero-shot/chatglm3-6b/legal_concept_concise.json")
    #evaluate_legal_concept_concise("../../prediction_result/zero-shot/deepseek-v3/legal_concept_concise.json")