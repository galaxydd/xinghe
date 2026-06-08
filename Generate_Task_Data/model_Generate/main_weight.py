import os

from Weight_Model.LexiLaw_gen import LexiLaw_generator
from Weight_Model.baichuan_13b_gen import Baichuan_13B_generator
from Weight_Model.baichuan_gen import Baichuan_base_generator
from Weight_Model.chatlaw_gen import Chatlaw_generator
from Weight_Model.disc_law_gen import Disc_generator
from Weight_Model.fuzi_gen import Fuzi_generator
from Weight_Model.model_gen import save_law_results, get_question_type
from Weight_Model.wisdom_gen import Wisdom_generator


def main(model_name, is_few_shot,is_vllm):
    # 设置固定参数列表
    global llm_generator
    few_shot_path = './few_shot.csv'  # few_shot路径
    model_path = f"/home/featurize/model/{model_name}"  # 模型路径
    device = "cuda"  # 设备（"cuda" 或 "cpu"）
    data_dir = "/home/featurize/work/Generate_Law_Task/data"

    # 设置输出目录
    output_dir = '/home/featurize/work/Generate_Law_Task/prediction_result/zero_shot'
    if is_few_shot:
        output_dir = '/home/featurize/work/Generate_Law_Task/prediction_result/few_shot'
    os.makedirs(output_dir, exist_ok=True)

    if model_name == "baichuan-7b":
        llm_generator = Baichuan_base_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )
    elif model_name == "baichuan-13b":
        llm_generator = Baichuan_13B_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )
    elif model_name == "fuzi-mingcha":
        llm_generator = Fuzi_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )
    elif model_name == "lexilaw":
        llm_generator = LexiLaw_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )
    elif model_name == "wisdom":
        llm_generator = Wisdom_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )
    elif model_name =="disc-law":
        llm_generator = Disc_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )

    elif model_name == "chatlaw-13b":
        llm_generator = Chatlaw_generator(
            is_few_shot=is_few_shot,
            device=device,
            is_vllm=is_vllm,
            model_path=model_path,
            model_name=model_name,
            few_shot_path=few_shot_path
        )


    # 初始化模型
    model, tokenizer = llm_generator.model_init()

    items = os.listdir(data_dir)
    # 过滤出里面全部的法律任务文件名
    file_list = [f for f in items if os.path.isfile(os.path.join(data_dir, f))]
    un_success_dict = {}
    for data_file in file_list:

        q_type = get_question_type(data_file)
        result_dict, un_success = llm_generator.generate_output(
            model=model,
            tokenizer=tokenizer,
            data_file=data_file,
            data_dir=data_dir,
            q_type=q_type,
        )

        un_success_dict[data_file] = un_success
        save_law_results(output_dir, model_name, data_file=data_file, result_dict=result_dict)

    save_law_results(output_dir, model_name, data_file='un_success_counts.json',
                                   result_dict=un_success_dict)

if __name__ == "__main__":
    model_name = "chatlaw-13b"  # 传入模型名称
    is_few_shot = False  # 如果需要开启 few-shot 学习，设置为 True
    is_vllm = False  # 是否使用 VLLM 模式
    main(model_name, is_few_shot,is_vllm)
    ###lawgpt两个 chatlaw disc-law7b 这四个
