# coding=utf-8
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import copy
import random

random.seed(0)

model_name = "ERNIE-3.5-8K"
model = None


data_file = "../data/FLUB-filtered.jsonl"
output_dir = "outputs"

prompt_dir = "prompts"
os.makedirs(output_dir, exist_ok=True)
with open(data_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
candidates = "，".join(sorted(set([item["type"] for item in data if isinstance(item["type"], str)])))

tasks = []
tasks += ["selection"]
tasks += ["explanation_cot"]

decoding_params = dict(temperature=0.7, top_p=0.8, max_tokens=1024)

from openai_api import call_openai_api

print(f"Output: {output_dir} | Decoding Params: {decoding_params} | Tasks: {tasks}")

shot_templates = {
    "selection": "输入：{sentence}\n选项：\n{options}\n答案：{answer}",
    "explanation_q": "输入问题：{sentence}\n回答：{answer}",
    "explanation_nq": "输入句子：{sentence}\n解释：{answer}"
}

fewshot_data = {key: {} for key in shot_templates}
for item in data:
    _id = item["id"]
    _type = item["type"]
    sentence = item["text"]
    options = "\n".join([f"{option}: {content}" for option, content in item["options"].items()])
    answer = item["answer"]
    explanation = item["explanation"]

for task in tasks:
    task_name = task
    print(f"********** Task: {task} | Model: {model_name} **********")
    prompt_files = {
        "": f"{task}.txt",
        "q": f"{task}_q.txt",
        "nq": f"{task}_nq.txt"
    }
    prompt_templates = {}
    for key, file in prompt_files.items():
        file = os.path.join(prompt_dir, file)
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                prompt_templates[key] = f.read()
    print("Prompt Templates:")
    print(prompt_templates)
    ids = set()

    output_file = os.path.join(output_dir, f"{task}_output_{model_name.lower()}.jsonl")
    task_data = copy.deepcopy(data)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        ids = set([item["id"] for item in items])
        # task_data = [item for item in data if item["id"] not in ids]
    prompts = []
    outputs = []
    for i, item in tqdm(enumerate(task_data), total=len(task_data), desc="Prompt"):
        _id = item["id"]
        # print(prompt_templates)
        fs_match_name = task_name
        if task_name.startswith("explanation"):
            fs_match_name = fs_match_name + ("_q" if item["is_question"] else "_nq")
        shot_template = shot_templates.get(fs_match_name)
        if item["is_question"]:
            prompt_template = prompt_templates.get("q", prompt_templates.get("", None))
        else:
            prompt_template = prompt_templates.get("nq", prompt_templates.get("", None))
        slots = dict(
            sentence=item["text"]
        )
        answer = None

        if task.startswith("selection"):
            options = [f"{option}: {content}" for option, content in item["options"].items()]
            slots["options"] = "\n".join(options)
            answer = item["answer"]
        else:
            answer = item["explanation"]
        if prompt_template is None:
            prompt_templates = ""
        prompt = prompt_template.format(**slots)
        prompts.append(prompt)
        output = {
            "id": _id,
            "text": item["text"],
            "prompt": prompt,
            "response": None,
            "answer": answer
        }
        outputs.append(output)

    prompts = [prompt for i, prompt in enumerate(prompts) if outputs[i]["id"] not in ids]
    outputs = [output for output in outputs if output["id"] not in ids]
    print(f"Origin {len(task_data)} Samples | Remain {len(outputs)} Samples")
    if len(outputs) <= 0:
        continue

    fw = open(output_file, "a+", encoding="utf-8")
    # 单线程调用 API
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Model Inference"):
        output = outputs[i]
        response = call_openai_api(prompt, model_name, **decoding_params)
        output["response"] = response
        fw.write(json.dumps(output, ensure_ascii=False) + "\n")
    fw.close()
