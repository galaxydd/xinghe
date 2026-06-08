import os
from model_Generate.API_Model.api_gen import decide_api
from model_Generate.API_Model.base_Api import BaseAPI


def main(api_name,data_dir,model_path,data_file,is_few_shot,few_shot_path):
    # 决定 API 相关参数
    api, base = decide_api(api_name)
    api_client = BaseAPI(api, base,is_few_shot,few_shot_path,is_zhipu=False)

    # 设置输出目录
    output_dir = '../prediction_result/zero_shot'
    if is_few_shot:
        output_dir = '../prediction_result/few_shot'

    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(model_path)  # 获取模型名称

    # 获取问题类型
    q_type = api_client.get_question_type(data_file)
    un_success_dict = {}
    # 处理单个文件
    result_dict, un_success = api_client.generate_output(data_dir, output_dir, data_file, model_path, q_type)
    un_success_dict[data_file] = un_success

    # 保存结果
    api_client.save_law_results(output_dir, model_name, data_file=data_file, result_dict=result_dict)
    api_client.save_law_results(output_dir, model_name, data_file='un_success_counts.json', result_dict=un_success_dict)

if __name__ == "__main__":
    is_few_shot = False
    few_shot_path = '../few_shot.csv'

    #api_name = "siliconCloud"
    #model_path = "Pro/THUDM/glm-4-9b-chat"

    api_name = "deepseek"
    model_path = "deepseek-reasoner"

    # 单一任务处理......
    data_dir = "../data2/legal_inner/"
    data_file = "mulit_hop.json"
    main(api_name,data_dir,model_path,data_file,is_few_shot,few_shot_path)

