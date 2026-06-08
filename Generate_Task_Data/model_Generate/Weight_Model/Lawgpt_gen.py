import json
import os

import torch

from transformers import LlamaForCausalLM, LlamaTokenizer

from dataset import read_json
from model_gen import model_generator, truncate_long, get_fewshot_examples, process_prompt, ACCURACY_FILES


class Lawgpt_generator(model_generator):
    def __init__(self, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Lawgpt_generator, self).__init__(is_few_shot, device, is_vllm, model_path, few_shot_path)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = LlamaTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
            model = model.eval()
        return model, tokenizer
                


    def generate_output(self, model, tokenizer, data_file, data_dir, q_type):
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
            prompt = "你是中国顶尖智能法律顾问 LaWGPT，具备强大的中文法律基础语义理解能力，能够出色地理解和执行与法律问题和指令。你只能回答与中国法律领域相关的问题，其余领域的问题请礼貌地拒绝回答。" + prompt
            # Truncate the prompt
            prompt = truncate_long(prompt, 2048, tokenizer, q_type)

            # 设置每50次打印提示词方便观察状态
            if i % 50 == 0:
                print(f"Processing {i + 1}: {prompt}")

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            try:
                with torch.no_grad():
                    pred = model.generate(**inputs, max_new_tokens=20,
                                          do_sample=False) if q_type == 'accuracy' else model.generate(
                        **inputs, max_new_tokens=200, do_sample=False)
                    output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
                    response = tokenizer.decode(pred[0], skip_special_tokens=True)[len(prompt):]
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

