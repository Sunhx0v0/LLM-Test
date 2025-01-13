import json

# 读取原始的JSON文件
input_file_path = '../data/FLUB-filtered.jsonl'  # 替换为你的输入文件路径
output_file_path = '../data/FLUB-filtered.jsonl'  # 替换为你的输出文件路径

# 准备一个空列表来保存过滤后的数据
filtered_data = []

# 逐行读取并解析文件内容
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            item = json.loads(line.strip())  # 解析每一行的JSON对象
            if item.get('type') != '冷笑话':  # 过滤掉 type 为 "冷笑话" 的项目
                filtered_data.append(item)
        except json.JSONDecodeError as e:
            print(f"解析JSON失败: {e}, 在这一行: {line}")

# 将过滤后的数据逐个写入新的文件，每个对象占一行
with open(output_file_path, 'w', encoding='utf-8') as file:
    for item in filtered_data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"过滤完成，结果已保存至: {output_file_path}")