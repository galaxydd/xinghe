from evaluation.utils import read_json, normalize_zh_answer, compute_rouge, is_dropout
import re
"""
Task:  法律摘要
Metric: ROUGE-L
Class: Generate
"""

def evaluate_legal_summary(data_file):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    data_dict = read_json(data_file)
    total = len(data_dict)
    print(total)
    dropout = 0
    pres = []
    refs = []
    for i, entry in enumerate(data_dict):
        #  模型未能成功回答 计算dropout概率
        bool_drop, prediction = is_dropout(entry["prediction"],'legal_summary')
        if bool_drop:
            dropout+=1
            continue
        reference = normalize_zh_answer(entry["reference"].replace("回答:", ""))
        pres.append(prediction)
        refs.append(reference)

    # ROUGE-L计算分数
    rouge_scores = compute_rouge(pres, refs)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    dropout_prob = dropout / total if total > 0 else 0
    print(f"Dropout probability: {dropout_prob:.2%}")
    print(f"rouge_l_score: {average_rouge_l:.2%}")

    return {"score": average_rouge_l},{"dropout_prob":dropout_prob}




if __name__ == "__main__":

    #evaluate_legal_summary("../../prediction_result/zero-shot/glm-4-9b-chat/legal_summary.json")
    evaluate_legal_summary("../../prediction_result/zero-shot/baichuan-7b/legal_summary.json")
    #evaluate_legal_summary("../../prediction_result/zero-shot/deepseek-v3/legal_summary.json")