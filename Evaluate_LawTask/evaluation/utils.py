import json
import re
import string

from rouge_chinese import Rouge
import jieba


"""
    读取prediction_result中的答案
"""
def read_json(input_file):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = []
    for key, item in data.items():
        # 提取 origin_prompt 数组的第一个元素，并获取其中的 "prompt" 字段
        prompt_text = ""
        if "origin_prompt" in item and isinstance(item["origin_prompt"], list) and item["origin_prompt"]:
            first_prompt = item["origin_prompt"][0]  # 获取列表中的第一个元素
            prompt_text = first_prompt.get("prompt", "").strip()  # 确保获取 "prompt" 字段并去除首尾空格

        # 获取预测结果和参考答案，并去除换行符或空格
        prediction = item.get("prediction", "").strip()
        reference = item.get("reference", "").strip()

        dataset.append({
            "instruction": prompt_text,
            "prediction": prediction,  # 预测结果
            "reference": reference  # 参考答案
        })

    return dataset

"""
    第一步预处理在evaluation_fuction内部
    1. 用于generate问题 处理计算ROUGE-L分数的 第二步预处理
    2. 用于Accuracy问题 的预处理
"""
def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    return white_space_fix(remove_punc(s))

"""
    计算rouge_L分数
"""
def compute_rouge(pres, refs):
    assert(len(pres) == len(refs))
    pres= [' '.join(jieba.cut(p)) for p in pres]
    refs = [' '.join(jieba.cut(r)) for r in refs]
    return Rouge().get_scores(pres, refs)
"""
      判断给定的预测结果是否为无效回答（dropout），并根据任务名称对预测结果进行处理。
        prediction (str): 模型的预测结果。
        task_name (str): 任务名称，根据不同任务调用相应的处理函数。
"""
def is_dropout(prediction, task_name=None):

    if prediction == "未成功回答":
        return True, prediction
    # 定义任务名称与对应处理函数的映射
    process_funcs = {
        'legal_summary': process_legal_summary,
        'legal_recitation': process_legal_headers,
        'legal_timeliness': extract_keyword,
    }
    # 根据任务名称进行处理，如果有对应的处理函数
    if task_name in process_funcs:
        prediction = normalize_zh_answer(process_funcs[task_name](prediction))
    else:
        prediction = normalize_zh_answer(prediction)

    # 如果处理后为空，则视为 dropout
    return (not prediction), prediction
"""
    法律时效性抽取函数
"""
def extract_keyword(text: str) -> str:

    pattern = re.compile(r'(有效|已废止|尚未生效)')
    match = pattern.search(text)
    return match.group(1) if match else ""

"""
        法律摘要抽取函数
处理prediction中 常见的开头型式
包括 摘要:
    摘要如下:
    该法律文本摘要为: ....
"""
def process_legal_summary(text):
    # 主匹配模式
    main_pattern = r'''
        \s*                            # 空格
        (?:摘要|法律文本摘要|该法律文本摘要)  # 核心匹配词
        (?:如下|为|内容为|的?内容)?     # 后缀词
        [\s：:]*                       # 分隔符号
    '''
    # 二次清理模式
    suffix_pattern = r'^[\s的：:、]+'
    entries = re.split(r'\n+', text)
    processed = []
    for entry in entries:
        # 进行文本清洗
        stage1 = re.sub(main_pattern, '', entry, flags=re.X | re.U)
        stage2 = re.sub(suffix_pattern, '', stage1.strip())
        if stage2:
            processed.append(stage2)

    return "\n".join(processed)

"""
        法条背诵抽取函数
处理prediction中 常见的开头型式
包括 《法律》第xx条的内容/规定如下/的内容是....
"""
def process_legal_headers(text):
    # 第一阶段：主清洗正则
    main_pattern = r'''
        (?:^《[^》]+》)?          # 可选法律名称（含书名号）
        第[零一二三四五六七八九十百]+条  # 必须包含的条款序号
        (?:[\s：:]*[的]?)         # 中文冒号/空格等
        (?:内容如下|规定如下|规定|内容是)?  # 常见引导语
        [：:。\s]*                # 结尾标点
    '''
    # 第二阶段：清理残留前缀
    suffix_pattern = r'^[\s的、是：:]+'
    entries = re.split(r'\n{2,}', text)
    processed = []
    for entry in entries:
        # 进行文本清洗
        stage1 = re.sub(main_pattern, '', entry, flags=re.X | re.U)
        stage2 = re.sub(suffix_pattern, '', stage1.strip())
        if stage2:
            processed.append(stage2)

    return "\n".join(processed)

if __name__ == "__main__":
    dataset = read_json("../prediction_result/zero-shot/baichuan-7b/legal_open_qa.json")
    for i, entry in enumerate(dataset):
        print(f"Entry {i + 1}:")
        print(f"  Instruction: {entry['instruction']}")
        print(f"  Prediction: {entry['prediction']}")
        print(f"  Reference: {entry['reference']}")
        print("-" * 50)
