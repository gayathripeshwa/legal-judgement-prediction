import pandas as pd
import json

def load_cjpe_files(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                text = example['text']  # Check exact field name
                data.append(text)
    return data

texts = load_cjpe_files([
    'data/cjpe_multi_train_0000',
    'data/cjpe_multi_train_0001'
])

with open('data/ildc_texts.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(text.replace('\n', ' ') + '\n\n')
