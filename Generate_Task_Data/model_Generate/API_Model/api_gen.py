import os
from dotenv import load_dotenv
from openai import base_url


def decide_api(api_name):
    # 指定 .env 文件路径（确保路径正确）
    env_file = r"E:\PycharmProjects\Generate_Task_Data\config.env"

    # 加载 .env 文件
    load_dotenv(dotenv_path=env_file)

    # 选择 API 并获取密钥和 URL
    if api_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = 'https://api.deepseek.com'
    elif api_name == "siliconCloud":
        api_key = os.getenv("SILICON_CLOUD_API_KEY")
        base_url = 'https://api.siliconflow.cn/v1'
    elif api_name == "zhipu":
        api_key = os.getenv("ZhiPu_API_KEY")
        base_url =None
    elif api_name == "gpt":
        api_key = os.getenv("GPT_API_KEY")
        base_url = 'https://api.deerapi.com'
    else:
        raise ValueError(f"Unsupported API name: {api_name}")
    # 检查是否成功获取 API Key
    if api_key is None:
        raise EnvironmentError(f"API key for {api_name} not found in {env_file}")

    return api_key, base_url

def truncate_long(prompt, context_length, tokenizer, q_type):
    '''
    For question with long context, truncate it.
    Args:
        prompt: Original prompt for the model
        context_length: Maximum context length for the model
        tokenizer: tokenizer for the model
        q_type: Must be 'generate' or 'accuracy'
    '''
    ori_prompt = tokenizer.encode(prompt)
    if q_type == 'generate':
        if len(ori_prompt) > context_length - 512:
                print(f"Input tokens too long, cut to {context_length - 512} tokens!")
                half = int((context_length - 512) / 2)
                prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    ori_prompt[-half:], skip_special_tokens=True)
    elif q_type == 'accuracy':
        if len(ori_prompt) > context_length - 20:
                print(f"Input tokens too long, cut to {context_length - 20} tokens!")
                half = int((context_length - 20) / 2)
                prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    ori_prompt[-half:], skip_special_tokens=True)
    else:
        raise ValueError(f"Wrong question type, q_type must be 'generation' or 'multiple_choice' but get {q_type}")
    return prompt