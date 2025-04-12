import hanlp
import os
import chinese_converter
import random
import re
from tqdm import tqdm
import hanlp.utils

MALE_WORDS = set(["他", "他们", "男", "男士", "男孩", "男子", "男性", "先生", "男人", "爸爸", "父亲", "姥爷", "儿子", "男友",\
    "叔叔", "哥哥", "弟弟", "爷爷", "外公", "公公", "舅舅", "伯伯", "大哥", "小弟", "男神", "男生",\
    "阿公", "爸比", "父親", "祖父", "外祖父", "堂哥", "堂弟", "表哥", "表弟"])

FEMALE_WORDS = set(["她", "她们", "女", "女士", "女孩", "女子", "女性", "小姐", "女人", "妈妈", "母亲", "姥姥", "女儿", "女友",\
    "阿姨", "姐姐", "妹妹", "奶奶", "外婆", "婆婆", "舅妈", "大姐", "小妹", "女神", "女生",\
    "阿嬷", "妈咪", "母親", "祖母", "外祖母", "堂姐", "堂妹", "表姐", "表妹"])


text = ""
for book_name in os.listdir('./book'):
    if '朱天心' in book_name:
        with open(os.path.join('./book', book_name), 'r', encoding='utf-8') as f:
            text_ = f.read()
            text_ = chinese_converter.to_simplified(text_)
            text += text_ + "\n"

text_chunks = text.replace("\u3000", "").split("\n")
# Remove empty strings from the list
text_chunks = [chunk for chunk in text_chunks if chunk.strip()]
# split sentences
text_chunks = [chunk for text_chunk in text_chunks for chunk in hanlp.utils.rules.split_sentence(text_chunk)]
# Remove senteces that only has non-Chinese characters
text_chunks = [chunk for chunk in text_chunks if re.search(r'[\u4e00-\u9fff]', chunk)]

# randamly select 10 chunks
text_samples = random.sample(text_chunks, len(text_chunks))
# print(text_samples)

# HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # multi-task learning model
HanLP = hanlp.pipeline() \
    .append(hanlp.load('COARSE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
    .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok')\
    .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')

def extract_verbs(toks, dep):
    records = []
    for i, tok in enumerate(toks):
        if tok in MALE_WORDS:
            records.append((tok, toks[dep[i][0]-1], dep[i][1], 'male'))
        elif tok in FEMALE_WORDS:
            records.append((tok, toks[dep[i][0]-1], dep[i][1], 'female'))
    return records

all_records = []
for text_sample in tqdm(text_samples):
    res = HanLP(text_sample)
    # print(res)
    # verbs = extract_verbs(res['tok'], res['dep'])
    # print(verbs)
    all_records.extend(extract_verbs(res['tok'], res['dep']))
    # print(all_records)
    # break

import pandas as pd
df_gender = pd.DataFrame(all_records, columns=['gender_word', 'verb', 'relation', 'gender'])
df_gender.to_csv('gender_coarse.csv')

