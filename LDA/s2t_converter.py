import os
from opencc import OpenCC

cc = OpenCC('s2t')

input_folder = 'literature'  # Replace with your actual folder path
output_folder = 'literature_traditional'  # Output folder for traditional Chinese text

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            simplified_text = f.read()

        traditional_text = cc.convert(simplified_text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(traditional_text)

        print(f'轉換完成：{filename}')