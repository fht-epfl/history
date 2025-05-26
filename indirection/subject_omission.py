from subject_omission_prompt import *
from openai import OpenAI
import ast
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
import chinese_converter
import re


def post(messages: list) -> list:

    client = OpenAI(
        base_url = "https://www.dmxapi.com/v1",
        api_key = "sk-Di1qvKyoz9ZrfGHZu44m6LH4o0zz8ehDUFsiU10a9FGmSM2q"
    )

    # client = OpenAI(
    #     base_url = "https://api.kpi7.cn/v1",
    #     api_key = "sk-fE6FPaxFFRTcufPpvuuZjGZIi7rXVCDvkqSBIRMteoqs2KNk"
    # )

    completion = client.chat.completions.create(
        model="deepseek-v3",
        # model="meta/llama-3.1-405b-instruct",
        messages=messages,
        # temperature=0.2,
        # top_p=0.7,
        # max_tokens=1024,
        # stream=True
    )

    return completion.choices[0].message.content

def detect_imageries(text) -> dict:

    text_chunks = [text[i:i+200] for i in range(0, len(text), 500)]

    answer_list = []

    for chunk in tqdm(text_chunks):
        messages = prompt_formatter(chunk)
        while True:
            answer = post(messages)
            print(answer)
            try:
                if answer.startswith("```json"):
                    match = re.search(r"```json(.*?)```", answer, re.DOTALL)
                    json_part = match.group(1).strip()
                    answer = ast.literal_eval(json_part)
                    answer_list += answer
                    break
                elif answer.startswith("[") and answer.endswith("]"):
                    answer = ast.literal_eval(answer)
                    answer_list += answer
                    break
                # answer_str += answer
            except:
                print("Error in parsing the answer. Retrying...")
                continue

    # img_count = dict(Counter(answer_list))

    return answer_list


df_book = pd.read_pickle("passive_voice.pkl")
# print(df_book.head())

df_zhutianxing = df_book[(df_book['author'] == '朱天心') & (df_book['year'] >= 1987)].iloc[:, :]

for i, row in df_zhutianxing.iterrows():
    text_chunk = row['text_chunk_smallest']
    text_chunk = "".join([sent for sent in text_chunk])
    # print(text_chunk)
    img_count = detect_imageries(text_chunk)
    print(f"Processing {row['title']}")
    with open(f"./omission/{row['title']}.json", "w", encoding="utf-8") as f:
        json.dump(img_count, f, ensure_ascii=False, indent=4)

# df_book['imageries'] = df_book['text_chunk_smallest'].apply(lambda x: detect_imageries("".join([sent for sent in x])))
# df_book['imageries']