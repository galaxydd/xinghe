import csv
import json
import os
from typing import Dict, List
import requests
from openai import OpenAI
from zhipuai import ZhipuAI

from model_Generate.API_Model.api_gen import truncate_long
from model_Generate.dataset import read_json


class BaseAPI:
    """API 客户端基类，封装通用逻辑"""

    # 类常量
    Q_TYPE_MAX_TOKENS_MAP: Dict[str, int] = {
        "accuracy": 20,
        "generate": 200
    }

    ACCURACY_FILES: List[str] = [
        "legal_concept_concise.json",
        "legal_focus_identification.json",
        "legal_question_classification.json",
        "legal_timeliness.json",
        "multi_hop.json",
        "sentence_prediction_500.json"
    ]

    def __init__(self, api_key: str,base_url: str,is_few_shot: bool,few_shot_path:str,is_zhipu:bool) -> None:
        """初始化API客户端
        :param api_key: 平台颁发的认证密钥
        :param is_few_shot 启用few-shot模式
        """

        self.api_key = api_key
        self.base_url = base_url
        if is_zhipu:
            self.client =ZhipuAI(
                api_key= self.api_key
            )
        else:
            self.client = OpenAI(
                base_url= self.base_url,
                api_key= self.api_key
             )
        self.is_few_shot =is_few_shot
        self.few_shot_path = few_shot_path

    def get_question_type(self, data_file: str) -> str:
        """根据数据文件判断问题类型
        :param data_file: 输入数据文件名
        :return: 问题类型标识符
        :raises ValueError: 当文件后缀不符合预期格式时
        """
        if not data_file.endswith(".json"):
            raise ValueError("Invalid file format, expected .json")

        return "accuracy" if data_file in self.ACCURACY_FILES else "generate"

    def get_max_tokens(self, q_type: str) -> int:
        """获取指定问题类型的token限额

        :param q_type: 问题类型标识符
        :return: 最大token数
        :raises KeyError: 当q_type不存在于映射表时
        """
        if q_type not in self.Q_TYPE_MAX_TOKENS_MAP:
            raise KeyError(f"Unsupported question type: {q_type}")

        return self.Q_TYPE_MAX_TOKENS_MAP[q_type]

    def generate_output(self,data_dir, output_dir, data_file, model_name, q_type):

        os.makedirs(output_dir, exist_ok=True)
        """ 处理单个 JSON 文件 """
        file_name = os.path.splitext(data_file)[0]  # 去除文件扩展名
        file_path = os.path.join(data_dir, data_file)

        dataset = read_json(file_path)  # 读取 JSON 数据
        if not dataset:
            print(f"文件 {data_file} 为空或无法解析")
            return
        print(f"----- 开始处理 {file_name} -----")

        num_un_success = 0  # 记录为成功数量
        result_dict = {}  # 存储所有结果
        if self.is_few_shot:
            few_shot_instruction = self.get_few_shot_examples(file_name, self.few_shot_path)
        else:
            few_shot_instruction = None  # 不使用 few-shot 时，设为空
        for i, data in enumerate(dataset):
            prompt = self.process_prompt(
                file_name,
                data['instruction'],
                data['question'],
                self.is_few_shot,
                few_shot_instruction
            )
            if i % 50 == 0:
                print(f"Question {i}: {prompt}")
            messages = [{"role": "user", "content": prompt}]

            try:

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    #max_tokens= self.get_max_tokens(q_type)

                )
                prediction = response.choices[0].message.content
                print(f"Answer {i }: {prediction}")

            except Exception as e:

                prediction = "未成功回答"
                num_un_success += 1
                print(e)

            # 组织数据
            result_dict[str(i)] = {
                "origin_prompt": [{"role": "HUMAN", "prompt": prompt}],
                "prediction": prediction,
                "reference": data.get('answer', '')
            }
        print(f"---------{file_name} 处理结束 --------------")
        return result_dict, num_un_success

    def save_law_results(self,output_dir, model_name, data_file, result_dict):
        """
        保存处理结果到 JSON 文件。

        参数：
        - output_dir: str，输出目录
        - model_name: str，模型名称
        - data_file: str，保存的文件名（不含扩展名）
        - result_dict: dict，要保存的数据
        返回：
        - output_file: str，保存的文件路径
        """
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, data_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        print(f"-------------{output_file} 文件保存成功 ------------------")

    def read_few_shot_by_task(self,file_path, task_name):
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

    def get_few_shot_examples(self,task_name, few_shot_path):
        """
        根据给定的 task_name 从 CSV 读取对应的 few-shot 示例，并返回格式化的示例字符串。
        """
        task_rows = self.read_few_shot_by_task(few_shot_path, task_name)
        if not task_rows:
            return f"未找到 task_name 为 {task_name} 的任何示例数据。"
        # 定义不同任务的示例格式
        if task_name == 'legal_summary':
            row = task_rows[0]
            return f"以下是一个例子:\n文本: {row['question']}\n摘要: {row['answer']}"
        if task_name == 'legal_open_qa':
            row = task_rows[0]
            return f"以下是一个例子:\n问题: {row['question']}\n答案: {row['answer']}"
        # 其他任务
        examples_list = [
            f"问题: {row['question']} 回答: {row['answer']}"
            for row in task_rows[:3]
        ]
        return "以下是三个例子:\n" + "\n".join(examples_list)

    def process_prompt(self,task_name, instruction, question, is_few_shot=False, few_shot_instruction=None):
        """
        处理输入的任务，并返回格式化的提示词 (Prompt)。
        """
        "如果是few_shot模式，需要添加问答示例"
        "few_shot_instruction由get_fewshot_examples函数而来"
        if is_few_shot:
            instruction += f"\n{few_shot_instruction}\n请你回答:"
        "根据任务类型辅助提示词"
        prompt_map = {
            'legal_summary': f"{instruction}\n文本: {question}",
            'legal_focus_identification': f"{instruction}\n咨询: {question}",
            'legal_timeliness': f"{instruction}\n法律: {question}"
        }

        return prompt_map.get(task_name, f"{instruction}\n问题: {question}")

    def generate_output_gpt(self,data_dir, output_dir, data_file, model_name, q_type):
        os.makedirs(output_dir, exist_ok=True)
        """ 处理单个 JSON 文件 """
        file_name = os.path.splitext(data_file)[0]  # 去除文件扩展名
        file_path = os.path.join(data_dir, data_file)
        dataset = read_json(file_path)  # 读取 JSON 数据
        if not dataset:
            print(f"文件 {data_file} 为空或无法解析")
            return
        print(f"----- 开始处理 {file_name} -----")

        num_un_success = 0  # 记录为成功数量
        result_dict = {}  # 存储所有结果
        if self.is_few_shot:
            few_shot_instruction = self.get_few_shot_examples(file_name, self.few_shot_path)
        else:
            few_shot_instruction = None  # 不使用 few-shot 时，设为空

        for i, data in enumerate(dataset):
            prompt = self.process_prompt(
                file_name,
                data['instruction'],
                data['question'],
                self.is_few_shot,
                few_shot_instruction
            )

            if i % 50 == 0:
                print(f"Question {i}: {prompt}")
            messages = [{"role": "user", "content": prompt}]

            url = "https://api.deerapi.com/v1/chat/completions"

            try:
                payload = json.dumps({
                    "model":  model_name,
                    "messages": messages,
                    "stream": False
                })
                headers = {
                    'Authorization': 'Bearer sk-Aw0l4yiaErXb7dcGykKm1gtI7wLBbXxA3XHJqy3UJCCX0sDj',
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                pre = json.loads(response.text)
                # 提取 content
                prediction = pre["choices"][0]["message"]["content"]
                print(f"Answer {i }: {prediction}")

            except Exception as e:

                prediction = "未成功回答"
                num_un_success += 1
                print(e)

            # 组织数据
            result_dict[str(i)] = {
                "origin_prompt": [{"role": "HUMAN", "prompt": prompt}],
                "prediction": prediction,
                "reference": data.get('answer', '')
            }
        print(f"---------{file_name} 处理结束 --------------")
        return result_dict, num_un_success