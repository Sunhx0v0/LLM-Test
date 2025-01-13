import openai
import requests
from openai import OpenAI

# 配置 OpenAI API 的基础信息
client = OpenAI(
    api_key = "xxx",
    base_url = "https://api.agicto.cn/v1",
)

# deepseek API:sk-cbde9b07247641f6a44c45f23dadb947

# 系统提示
system_prompt = "You are a helpful assistant."

def call_openai_api(prompt, model_name, **kwargs):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": prompt}
    ]
    max_retry = kwargs.pop("max_retry", 3)
    retry_count = 0
    while True:
        try:
            # 创建 ChatCompletion 请求
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,  # 此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
                **kwargs
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"Exception: {e} / Prompt: {prompt} / Model: {model_name} / Retry: {retry_count}")
            if retry_count >= max_retry:
                return None
            retry_count += 1
            continue

