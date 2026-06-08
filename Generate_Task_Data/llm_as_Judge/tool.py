import json


def read_json(input_file):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = []
    for key, item in data.items():
        # 提取 origin_prompt 数组的第一个元素，并获取其中的 "prompt" 字段
        prompt_text = ""
        if "origin_prompt" in item and isinstance(item["origin_prompt"], list) and item["origin_prompt"]:
            first_prompt = item["origin_prompt"][0]  # 获取列表中的第一个元素
            prompt_text = first_prompt.get("prompt", "").strip()  # 确保获取 "prompt" 字段并去除首尾空格

        # 获取预测结果和参考答案，并去除换行符或空格
        prediction = item.get("prediction", "").strip()
        reference = item.get("reference", "").strip()

        dataset.append({
            "instruction": prompt_text,
            "prediction": prediction,  # 预测结果
            "reference": reference  # 参考答案
        })

    return dataset
# 根据前置的“法律得分表格”，为每个生成类任务手动指定表现最好的模型
# 键为对应的任务文件名，值为该任务下得分最高的模型文件夹名
# 目前先这样配置得分最好得模型，后续增加模型后进一步进行调整
BEST_MODEL_MAPPING = {
    #"legal_open_qa.json": "deepseek-reasoner",   # 对应 法律规范矛盾
    "legal_recitation.json": "deepseek-reasoner",# 对应 内部逻辑矛盾
    #"legal_summary.json": "deepseek-chat",       # 对应 法律事实矛盾
    #"judicial_analysis.json": "chatglm3-6b"      # 对应 解释主体矛盾
}

# ==========================================
# 3. 裁判提示词模板
# ==========================================
JUDGE_PROMPT_TEMPLATE = """
# 角色设定
你是一个法律逻辑与事实审查专家。你的任务是对比【标准答案】，审查【待测模型输出】中是否存在特定的“法律幻觉”。

- 原始任务上下文 (Input_Context)：{question}
- 标准答案与解析 (Ground_Truth)：{answer}
- 待测模型输出 (Model_Output)：{model_answer}

# 审查规则
请严格按照以下五项法律幻觉现象进行独立核查：
1. 引用失效法律（H1）：核查引用的法条是否已废止，或版本是否旧于标准答案适用的版本。
2. 无关法条引用（H2）：核查提取的法律焦点或引用的法条是否脱离标准答案的核心争议，被次要事实误导。
3. 自相矛盾建议（H3）：核查回答内部是否存在逻辑突变，即最终结论与前文自身设定的前提相互冲突。
4. 捏造法条（H4）：核查是否凭空编造了现行法律及标准答案中均不存在的罪名、概念或条款内容。
5. 法条编号错误（H5）：核查引用的法条内容语义是否正确，但却为其标注了错误的数字序号。

# 输出格式要求 (严格输出纯 JSON 格式)
请对上述规则进行判定，并输出包含 5 个布尔值（true/false）和 1 个简短原因的 JSON 结构。true 代表检测到了该类幻觉，false 代表未检测到。

```json
{{
  "H1_Outdated_Provisions": false, 
  "H2_Irrelevant_Citations": false, 
  "H3_Contradictory_Advice": false, 
  "H4_Fabricated_Provisions": false, 
  "H5_Incorrect_Article_Numbers": false,
  "Reasoning": "请用一句话简述判定为 true 的原因，若全为 false 则填无。"
}}


"""