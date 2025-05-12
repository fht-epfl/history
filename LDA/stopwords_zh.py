from opencc import OpenCC
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word: 
                stopwords.add(word)
    return stopwords

hit_stopwords = load_stopwords('./stopwords/hit_stopwords.txt')
scu_stopwords = load_stopwords('./stopwords/scu_stopwords.txt')
cn_stopwords  = load_stopwords('./stopwords/cn_stopwords.txt')

combined_stopwords = hit_stopwords.union(scu_stopwords, cn_stopwords)
cc = OpenCC('s2t')  
traditional_stopwords = { cc.convert(word) for word in combined_stopwords }
sorted_stopwords = sorted(traditional_stopwords)
with open('./stopwords/stopwords_zh.txt', 'w', encoding='utf-8') as f:
    for word in sorted_stopwords:
        f.write(word + '\n')