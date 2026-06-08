import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from vllm import SamplingParams
from dataset import read_json
from model_ge.model_gen import model_generator, truncate_long, ACCURACY_FILES, get_fewshot_examples, process_prompt


class Wisdom_generator(model_generator):
    def __init__(self, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Wisdom_generator, self).__init__(is_few_shot, device, is_vllm, model_path, few_shot_path)
        self.model_name = model_name

    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16,
                                                         trust_remote_code=True)
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
        if self.is_vllm:
            sampling_params = SamplingParams(temperature=0, max_tokens=20) \
                if q_type == 'accuracy' else SamplingParams(temperature=0, max_tokens=300)
            prompt = []
            for data_index, data in enumerate(dataset):
                input_text = data['instruction'] + data['question']
                current_prompt = '</s>Human:' + input_text + '</s>Assistant: '
                # Truncate the prompt
                truncated_prompt = truncate_long(current_prompt, 4096, tokenizer, q_type)
                prompt.append(truncated_prompt)

                # 设置每50次打印提示词方便观察状态
                if data_index % 50 == 0:
                    print(f"Processing -------{data_index + 1}: {input_text}")
                try:
                    # 输出响应
                    response_ls = model.generate(prompt,sampling_params)
                    for i, out in enumerate(response_ls):
                        response = out.outputs[0].text
                        result_dict[str(i)] = {
                            "origin_prompt": [{"role": "HUMAN", "prompt": prompt[i]}],
                            "prediction": response,
                            "reference": dataset[i].get('answer', '')
                        }
                except:
                    for i in range(len(dataset)):
                        num_unsuccess += 1
                        print("Fail to answer")
                        response = "未成功回答"
                        result_dict[str(i)] = {
                            "origin_prompt": [{"role": "HUMAN", "prompt": prompt[i]}],
                            "prediction": response,
                            "reference": dataset[i].get('answer', '')
                        }
        else:
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
                prompt = '</s>Human:' + prompt + '</s>Assistant: '
                # Truncate the prompt
                prompt = truncate_long(prompt, 4096, tokenizer, q_type)

                # 设置每50次打印提示词方便观察状态
                if i % 50 == 0:
                    print(f"Processing {i + 1}: {prompt}")

                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # 送入 GPU 计算
                try:
                    with torch.no_grad():
                        pred = model.generate(**inputs, max_new_tokens=20, do_sample=False) if q_type == 'accuracy' else model.generate(
                            **inputs, max_new_tokens=200, do_sample=False)
                        output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
                        response = output.split("Assistant: ")[1]
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














