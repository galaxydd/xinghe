from llm_as_Judge.tool import read_json, JUDGE_PROMPT_TEMPLATE, BEST_MODEL_MAPPING
from model_Generate.API_Model.api_gen import decide_api
import os
import json
import re
from openai import OpenAI

def call_judge_llm(client,question, answer, model_answer):
    """请求大模型 API 进行裁判"""

    try:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            model_answer=model_answer)

        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "你是一个输出 JSON 格式的法律审查专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API异常] 调用裁判模型失败: {e}")
        return None


def judge_generate(client,prediction_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(prediction_dir)
    # 遍历筛选好的最优模型与任务映射
    for task_file, best_model in BEST_MODEL_MAPPING.items():
        file_path = os.path.join(prediction_dir, best_model, task_file)
        output_path = os.path.join(output_dir,best_model,task_file)
        model_output_dir = os.path.dirname(output_path)
        # 2. 判断父目录是否存在，不存在则一次性创建多级目录
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        print(file_path)

        if not os.path.exists(file_path):
            print(f"[警告] 找不到预测文件: {file_path}，已跳过。")
            continue

        print(f"\n 开始诊断任务: {task_file} | 评测模型: {best_model}")
        final_results = []
        data_dict = read_json(file_path)
        # 读取答案中的每一条数据
        for i, entry in enumerate(data_dict):
            question = entry["instruction"]
            answer = entry["reference"]
            model_answer = entry["prediction"]
            judge_result = call_judge_llm(client,question, answer, model_answer)
            print(judge_result)
            if (i+1) % 50 == 0:
                print(f"当前已处理 {i+1} 条数据...\n")
                print(judge_result)
            processed_entry = {
                "id": i,
                "CDHR_Diagnosis": judge_result
            }
            final_results.append(processed_entry)

        print(f"\n 成功处理了 {len(final_results)} 条数据。")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        print(f"所有诊断结果已带序号成功保存至: {output_path}")


def judge_generate_2(client, prediction_dir, output_dir):
    # 确保输出总目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for task_file, best_model in BEST_MODEL_MAPPING.items():
        file_path = os.path.join(prediction_dir, best_model, task_file)
        output_path = os.path.join(output_dir, best_model, task_file)
        model_output_dir = os.path.dirname(output_path)
        # 2. 判断父目录是否存在，不存在则一次性创建多级目录
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        print(file_path)
        # 确保每个模型的子文件夹存在
        model_output_dir = os.path.join(output_dir, best_model)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)


        print(f"读取路径: {file_path}")

        if not os.path.exists(file_path):
            print(f"[警告] 找不到预测文件: {file_path}，已跳过。")
            continue

        print(f"\n开始诊断任务: {task_file} | 评测模型: {best_model}")

        data_dict = read_json(file_path)
        count = 0

        # 使用 'w' 模式打开文件，准备逐行写入
        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(data_dict):
                question = entry["instruction"]
                answer = entry["reference"]
                model_answer = entry["prediction"]

                # 调用裁判模型
                judge_result = call_judge_llm(client, question, answer, model_answer)

                # 构造单条结果
                processed_entry = {
                    "id": i + 1,
                    "CDHR_Diagnosis": judge_result
                }
                f.write(json.dumps(processed_entry, ensure_ascii=False) + "\n")
                f.flush()  # 确保数据立即写入磁盘
                count += 1

                if (i + 1) % 50 == 0:
                    print(f"当前已处理 {i + 1} 条数据...")
                    # print(judge_result) # 如果需要看实时结果可以取消注释
        print(f"\n任务完成：成功处理并保存了 {count} 条数据至 {output_path}")

if __name__ == "__main__":
    api_name = "deepseek"
    judge_model = "deepseek-v4-pro"
    output_dir = "../diagnostic_results"
    api, base = decide_api(api_name)
    client = OpenAI(
        api_key=api,
        base_url=base
    )
    prediction_dir = "../prediction_result/zero_shot"
    #judge_generate(client, prediction_dir, output_dir)
    judge_generate_2(client,prediction_dir,output_dir)