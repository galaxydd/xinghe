import re
from evaluation.utils import read_json, normalize_zh_answer, compute_rouge, is_dropout

"""
Task:  法条背诵
Metric: ROUGE-L
Class: Generate
"""

def evaluate_legal_recitation(data_file):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    data_dict = read_json(data_file)
    dropout = 0
    pres = []
    refs = []
    for i, entry in enumerate(data_dict):
        #  模型未能成功回答 计算dropout概率
        dropout_flag, prediction = is_dropout(entry["prediction"], 'legal_recitation')
        if dropout_flag:
            dropout += 1
            continue

        reference = normalize_zh_answer(entry["reference"].replace("回答:", ""))
        pres.append(prediction)
        refs.append(reference)

    # ROUGE-L计算分数,Drop_pro
    return compute_rouge(pres, refs,dropout,len(data_dict))



if __name__ == "__main__":

    #evaluate_legal_recitation("../../prediction_result/zero-shot/glm-4-9b-chat/legal_recitation.json")
    evaluate_legal_recitation("../../prediction_result/zero-shot/baichuan-7b/legal_recitation.json")
    evaluate_legal_recitation("../../prediction_result/zero-shot/deepseek-v3/legal_recitation.json")

