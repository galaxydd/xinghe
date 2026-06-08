import os
import json
import torch
import time
from typing import Dict
import torch
from IPython.sphinxext.ipython_directive import OUTPUT
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from dataset import read_json
from model_ge.model_gen import model_generator, ACCURACY_FILES, truncate_long, get_fewshot_examples, process_prompt


class Fuzi_generator(model_generator):
    def __init__(self, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Fuzi_generator, self).__init__(is_few_shot, device, is_vllm, model_path, few_shot_path)
        self.model_name = model_name

    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().eval().cuda()
        return model, tokenizer

    def generate_output(self, model, tokenizer, data_file, data_dir,  q_type):
        """ 处理单个 JSON 文件 """
        file_name = os.path.splitext(data_file)[0]  # 去除文件扩展名
        file_path = os.path.join(data_dir, data_file)

        dataset = read_json(file_path)  # 读取 JSON 数据
        if not dataset:
            print(f"文件 {data_file} 为空或无法解析")
            return
        print(f"-----{self.model_name} 开始处理 {file_name} -----")

        num_unsuccess = 0  # 记录为成功数量
        result_dict = {}  # 存储所有结果

        if self.is_few_shot:
            few_shot_instruction = get_fewshot_examples(file_name, self.few_shot_path)
        else:
            few_shot_instruction = None  # 不使用 few-shot 时，设为空
        for i, data in enumerate(dataset):
            prompt = process_prompt(
                file_name,
                data['instruction'],
                data['question'],
                self.is_few_shot,
                few_shot_instruction
            )
            try:
                # 输出响应
                prompt = truncate_long(prompt, 2048, tokenizer, q_type)
                if i % 50 == 0:
                    print(f"问题 {i + 1}: {prompt}")
                response, _ = model.chat(tokenizer, prompt, history=[], do_sample=False)
                if data_file not in ACCURACY_FILES:
                    print(f"回答 {i + 1}: {response}")
                else:
                    if i % 50 == 0:
                        print(f"回答 {i + 1}: {response}")
            except:
                num_unsuccess += 1
                response = "未成功回答"

            # 组织数据
            result_dict[str(i)] = {
                "origin_prompt": [{"role": "HUMAN", "prompt": prompt}],
                "prediction": response,
                "reference": data.get('answer', '')
            }

        return result_dict,num_unsuccess
