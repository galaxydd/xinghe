import os
import json
import torch
import time
from typing import Dict
import torch
from IPython.sphinxext.ipython_directive import OUTPUT

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import read_json
from model_ge.model_gen import model_generator


class Baichuan_base_generator(model_generator):
    def __init__(self, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Baichuan_base_generator, self).__init__(is_few_shot, device, is_vllm, model_path, few_shot_path)
        self.model_name = model_name

    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().eval().cuda()
        return model, tokenizer

    # def generate_output(self,model,tokenizer,data_file,data_dir,output_dir):

    #     """ 处理单个 JSON 文件 """
    #     file_name = os.path.splitext(data_file)[0]  # 去除文件扩展名
    #     file_path = os.path.join(data_dir, data_file)

    #     dataset = read_json(file_path)  # 读取 JSON 数据
    #     if not dataset:
    #         print(f"文件 {data_file} 为空或无法解析")
    #         return

    #     print(f"-----{self.model_name} 开始处理 {file_name} -----")

    #     result_dict = {}  # 存储所有结果
    #     for i, data in enumerate(dataset):
    #         input_text = data['instruction'] + data['question']
    #         # 设置每50次打印提示词方便观察状态
    #         if i % 50 == 0:
    #             print(f"Processing {i + 1}: {input_text}")

    #         inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 送入 GPU 计算
    #          # 生成文本
    #         with torch.no_grad():
    #             output = model.generate(**inputs, max_length=200)
    #         #解码输出
    #         response = tokenizer.decode(output[0], skip_special_tokens=True)
    #       # 组织数据
    #         result_dict[str(i)] = {
    #             "prompt": [{"role": "HUMAN", "prompt": input_text}],
    #             "prediction": response,
    #             "reference": data.get('answer', '')
    #         }

    #     # 保存结果
    #     model_output_dir = os.path.join(output_dir, self.model_name)

    #     os.makedirs(model_output_dir, exist_ok=True)

    #     output_file = os.path.join(model_output_dir, f"{file_name}.json")

    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(result_dict, f, ensure_ascii=False, indent=4)

    #     print(f"-----{self.model_name} 处理完成 {file_name}，结果已保存至 {output_file} -----")
    def generate_output(self, model, tokenizer, data_file, data_dir, output_dir):
        """ 处理单个 JSON 文件 """
        file_name = os.path.splitext(data_file)[0]  # 去除文件扩展名
        file_path = os.path.join(data_dir, data_file)

        dataset = read_json(file_path)  # 读取 JSON 数据
        if not dataset:
            print(f"文件 {data_file} 为空或无法解析")
            return

        print(f"-----{self.model_name} 开始处理 {file_name} -----")

        model_output_dir = os.path.join(output_dir, self.model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, f"{file_name}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[')  # JSON 数组开始
            for i, data in enumerate(dataset):
                input_text = data['instruction'] + data['question']

                if i % 50 == 0:
                    print(f"Processing {i + 1}: {input_text}")

                start_time = time.time()

                inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 送入 GPU 计算
                with torch.no_grad():
                    output = model.generate(**inputs, max_length=200)
                response = tokenizer.decode(output[0], skip_special_tokens=True)

                end_time = time.time()

                elapsed_time = end_time - start_time
                print(f"一次响应时间为: {elapsed_time:.2f} seconds")

                result = {
                    "prompt": [{"role": "HUMAN", "prompt": input_text}],
                    "prediction": response,
                    "reference": data.get('answer', ''),
                }

                json.dump(result, f, ensure_ascii=False, indent=4)
                f.write(',\n')  # 添加逗号换行，保持 JSON 格式
                f.flush()  # 立即写入磁盘

            f.seek(f.tell() - 2, os.SEEK_SET)  # 移除最后一个逗号
            f.write(']')  # JSON 数组结束

        print(f"-----{self.model_name} 处理完成 {file_name}，结果已保存至 {output_file} -----")

