import os
from model_Generate.API_Model.api_gen import decide_api
from model_Generate.API_Model.base_Api import BaseAPI

def process_files(api_name, model_path,is_few_shot,few_shot_path):
    # 决定 API 相关参数
    api, base = decide_api(api_name)
    api_client = BaseAPI(api, base,is_few_shot,few_shot_path,is_zhipu=False)
    output_dir = '../prediction_result/zero_shot'
    if is_few_shot:
        output_dir = '../prediction_result/few_shot'
    # 参数配置

    data_dir = "../data"
    model_name = os.path.basename(model_path)  # 获取模型名称

    items = os.listdir(data_dir)
    # 过滤出里面全部的法律任务文件名
    file_list = [f for f in items if os.path.isfile(os.path.join(data_dir, f))]
    un_success_dict = {}

    for data_file in file_list:
        q_type = api_client.get_question_type(data_file)
        # 处理单个文件（可以扩展为批量处理）---见 all_task

        if api_name == 'gpt':
            result_dict, un_success = api_client.generate_output_gpt(data_dir, output_dir, data_file, model_path, q_type)
        else:
            result_dict, un_success = api_client.generate_output(data_dir, output_dir, data_file, model_path, q_type)
        un_success_dict[data_file] = un_success

        # 保存生成结果
        api_client.save_law_results(output_dir, model_name, data_file=data_file, result_dict=result_dict)

    # 保存法律任务未成功数量
    api_client.save_law_results(output_dir, model_name, data_file='un_success_counts.json', result_dict=un_success_dict)

if __name__ == "__main__":
    is_few_shot = False
    few_shot_path = '../few_shot.csv'
    api_name = "deepseek"
    process_files(api_name,'deepseek-v4-pro',is_few_shot,few_shot_path)

