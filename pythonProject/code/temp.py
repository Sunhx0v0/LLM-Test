import openai
import requests
from openai import OpenAI


# 系统提示
system_prompt = "You are a helpful assistant."


def call_openai_api(prompt, model_name, **kwargs):
    """
    调用 OpenAI API 进行文本生成。

    参数:
    - prompt: 用户输入的提示文本。
    - model_name: 使用的模型名称。
    - kwargs: 其他传递给 API 的参数（如温度、最大令牌数等）。

    返回:
    - 生成的响应文本，如果发生不可恢复的错误则返回 None。
    """
    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # 设置最大重试次数
    max_retry = kwargs.pop("max_retry", 3)
    retry_count = 0

    try:
        # 创建 ChatCompletion 请求
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",  # 此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
        )
        # chat_completion = openai.ChatCompletion.create(
        #     model=model_name,
        #     messages=messages,
        #     **kwargs
        # )

        # 获取生成的响应内容
        response = chat_completion.choices[0].message.content
        return response

    except Exception as e:
        # 打印异常信息
        print(f"Exception: {e} / Prompt: {prompt} / Model: {model_name} / Retry: {retry_count}")

        # 如果重试次数超过最大限制，返回 None
        if retry_count >= max_retry:
            return None

        # 增加重试计数，继续重试
        retry_count += 1


# #示例调用
# if __name__ == "__main__":
#     prompt = "我要以“LLM对复杂中文的理解水平”为评测维度构建LLM的benchmark，请说明选择的维度及其意义，阐述其创新性或重要性。这是论文的一个部分，请给出一段完整的文字，不要分点回答"
#     model_name = "gpt-4o-mini"  # 根据需要更换模型
#     response = call_openai_api(prompt, model_name)
#     if response:
#         print("Generated Response:", response)
#     else:
#         print("Failed to get a response after multiple retries.")
