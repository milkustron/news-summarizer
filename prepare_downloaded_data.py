import os
import json
import pandas as pd

def remove_invalid_lines(file_path):
    valid_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        try:
            json.loads(line)
            valid_lines.append(line)
        except json.JSONDecodeError:
            continue

    with open(file_path, 'w') as file:
        file.writelines(valid_lines)


for file_name in os.listdir("data"):
    if file_name.startswith(("test_", "train_", "dev_")):
        print(file_name)
        remove_invalid_lines(os.path.join("data", file_name))

schema = ["url", "archive", "title", "date", "text", "summary", "compression", "coverage", "density", "compression_bin", "coverage_bin", "density_bin"]
records = pd.DataFrame(columns=schema)

for file_name in os.listdir("data"):
    if file_name.startswith(("test_", "train_", "dev_")):
        path = os.path.join("data", file_name)

        with open(path, 'r', encoding='utf-8') as file:
            print(file_name)
            jsonObj = pd.read_json(path_or_buf=path, lines=True)
            records = pd.concat([records, jsonObj], ignore_index=True)

records.to_csv('data.csv')
