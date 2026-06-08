import csv
import json
import os

from vllm import LLM, SamplingParams
from typing import Dict, List



class model_generator:

    def __init__(self, is_few_shot, device, is_vllm, model_path, few_shot_path):
        '''
        Args:
            f_path: The path for original data file in json format
            is_few_shot: Whether to utilize few-shot setting
            device: device number of cuda
            model_path: The path of the model
            few_shot_path: The path for few-shot data file
        '''
        self.model_path = model_path
        self.is_few_shot = is_few_shot
        if is_few_shot and few_shot_path == None:
            raise ValueError("Cannot find few-shot path")
        self.few_shot_path = few_shot_path
        self.device = device
        self.is_vllm = is_vllm

    def model_init_vllm(self):
        '''
        Init the model using vllm
        '''
        model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="half",
            # gpu_memory_utilization=0.75,  # 降低显存利用率（默认0.9）
            enforce_eager=True,
            # swap_space = 16  # 启用8GB磁盘交换空间缓解显存压力
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer


ACCURACY_FILES: List[str] = [
    "legal_concept_concise.json",
    "legal_focus_identification.json",
    "legal_question_classification.json",
    "legal_timeliness.json",
    "sentence_prediction_500.json",
    "mulit_hop.json"
]

Q_TYPE_MAX_TOKENS_MAP: Dict[str, int] = {
    "accuracy": 20,
    "generate": 300
}

def read_few_shot_by_task(file_path, task_name):
    """
    从给定的CSV文件中读取指定 task_name 的所有行，返回一个列表，每个元素是一个字典（包含 question 和 answer）。
    """
    results = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['task_name'] == task_name:
                results.append(row)
    return results


def get_fewshot_examples(task_name, few_shot_path):
    """
    根据给定的 task_name 从 CSV 读取对应的 few-shot 示例，并返回格式化的示例字符串。
    """
    task_rows = read_few_shot_by_task(few_shot_path, task_name)
    if not task_rows:
        return f"未找到 task_name 为 {task_name} 的任何示例数据。"

    # 定义不同任务的示例格式
    if task_name == 'legal_summary':
        row = task_rows[0]
        return f"以下是一个例子:\n文本: {row['question']}\n摘要: {row['answer']}\n"

    if task_name == 'legal_open_qa':
        row = task_rows[0]
        return f"以下是一个例子:\n问题: {row['question']}\n答案: {row['answer']}\n"

    # 其他任务，展示 3 条示例
    examples_list = [
        f"问题: {row['question']} 回答: {row['answer']}\n"
        for row in task_rows[:3]
    ]
    return "以下是三个例子:\n" + "\n".join(examples_list) +"\n"


def process_prompt(task_name, instruction, question, is_few_shot=False, few_shot_instruction=None):
    """
    处理输入的任务，并返回格式化的提示词 (Prompt)。
    """
    if is_few_shot :
        instruction += f"\n{few_shot_instruction}\n请你回答下面所给出的问题:"

    prompt_map = {
        'legal_summary': f"{instruction}\n文本: {question}",
        'legal_focus_identification': f"{instruction}\n咨询: {question}",
        'legal_timeliness': f"{instruction}\n法律: {question}"
    }

    return prompt_map.get(task_name, f"{instruction}\n问题: {question}")
def get_question_type(data_file):
    print(data_file)
    if not data_file.endswith(".json"):
        raise ValueError("Invalid file format, expected .json")
    return "accuracy" if data_file in ACCURACY_FILES else "generate"


def truncate_long(prompt, context_length, tokenizer, q_type):
    '''
    For question with long context, truncate it.
    Args:
        prompt: Original prompt for the model
        context_length: Maximum context length for the model
        tokenizer: tokenizer for the model
        q_type: Must be 'generate' or 'accuracy'
    '''
    ori_prompt = tokenizer.encode(prompt)
    if q_type == 'generate':
        if len(ori_prompt) > context_length - 512:
            print(f"Input tokens too long, cut to {context_length - 512} tokens!")
            half = int((context_length - 512) / 2)
            prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                ori_prompt[-half:], skip_special_tokens=True)
    elif q_type == 'accuracy':
        if len(ori_prompt) > context_length - 20:
            print(f"Input tokens too long, cut to {context_length - 20} tokens!")
            half = int((context_length - 20) / 2)
            prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                ori_prompt[-half:], skip_special_tokens=True)
    else:
        raise ValueError(f"Wrong question type, q_type must be 'generation' or 'multiple_choice' but get {q_type}")
    return prompt


def save_law_results(output_dir, model_name, data_file, result_dict):
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    output_file = os.path.join(model_output_dir, data_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print(f"-------------{output_file} 文件保存成功 ------------------")