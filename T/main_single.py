import os
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_ge.baichuan_gen import Baichuan_base_generator

def main():
    print("hello")
    data_file = 'legal_timeliness.json'
    is_few_shot = False  # 如果需要开启 few-shot 学习，设置为 True
    is_vllm = False  # 是否使用 VLLM 模式
    few_shot_path = None  # 如果有少量训练数据路径，可以提供
    model_path = "/mnt/workspace/model/baichuan-7b"  # 模型路径
    model_name = "Baichuan-7b"  # 模型名称
    device = "cuda"  # 设备（"cuda" cpu)
    data_dir = "/mnt/workspace/Generate_Law_Task/data"

    # 设置输出目录
    output_dir = './prediction_result/zero_shot'
    if is_few_shot:
        output_dir = './prediction_result/few_shot'

    os.makedirs(output_dir, exist_ok=True)

    llm_generator = Baichuan_base_generator(
        is_few_shot=is_few_shot,
        device=device,
        is_vllm=is_vllm,
        few_shot_path=few_shot_path,
        model_path=model_path,
        model_name=model_name,
    )

    # 初始化模型
    model, tokenizer = llm_generator.model_init()
    llm_generator.generate_output(model=model,
                                  tokenizer=tokenizer,
                                  data_file=data_file,
                                  data_dir=data_dir,
                                  output_dir=output_dir)


if __name__ == "__main__":
    main()