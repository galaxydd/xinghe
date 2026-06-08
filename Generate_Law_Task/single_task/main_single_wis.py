import os
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_ge.baichuan_gen import Baichuan_base_generator
from model_ge.fuzi_gen import Fuzi_generator
from model_ge.model_gen import save_law_results, get_question_type
from model_ge.wisdom_gen import Wisdom_generator


def main(model_name, data_file,is_few_shot,is_vllm):
    # 设置固定参数列表
    few_shot_path = None  # 如果有少量训练数据路径，可以提供
    model_path = f"/home/featurize/model/{model_name}"  # 模型路径
    device = "cuda"  # 设备（"cuda" 或 "cpu"）
    data_dir = "/home/featurize/work/Generate_Law_Task/data"

    # 设置输出目录
    output_dir = '/home/featurize/work/Generate_Law_Task/prediction_result/zero_shot'
    if is_few_shot:
        output_dir = '/home/featurize/work/Generate_Law_Task/prediction_result/few_shot'
    os.makedirs(output_dir, exist_ok=True)

    # Fuzi_generator Wisdom_generator Baichuan_Base_generator
    llm_generator = Wisdom_generator(
        is_few_shot=is_few_shot,
        device=device,
        is_vllm=is_vllm,
        few_shot_path=few_shot_path,
        model_path=model_path,
        model_name=model_name,
    )

    # 初始化模型
    model, tokenizer = llm_generator.model_init()
    q_type = get_question_type(data_file)

    un_success_dict = {}
    result_dict, un_success = llm_generator.generate_output(
        model=model,
        tokenizer=tokenizer,
        data_file=data_file,
        data_dir=data_dir,
        q_type=q_type
    )

    un_success_dict[data_file] = un_success
    save_law_results(output_dir, model_name, data_file=data_file, result_dict=result_dict)
    save_law_results(output_dir, model_name, data_file='un_success_counts.json',
                                   result_dict=un_success_dict)

if __name__ == "__main__":
    model_name = "wisdom"  # 传入模型名称
    data_file = 'legal_timeliness.json'  # 传入数据文件名称
    is_few_shot = False  # 如果需要开启 few-shot 学习，设置为 True
    is_vllm = True  # 是否使用 VLLM 模式
    main(model_name, data_file,is_few_shot,is_vllm)
