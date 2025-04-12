from gensim.models import Word2Vec
import sys
import hanlp
import os
import chinese_converter
from tqdm import tqdm


def tokenize_line(line, tokenizer):
    # Load a HanLP tokenizer model. Adjust the model name as needed.
    return tokenizer(line.strip())

def load_corpus(filenames, tokenizer):
    sentences = []
    for filename in tqdm(filenames):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = chinese_converter.to_simplified(line.strip())
                tokens = tokenize_line(line, tokenizer)
                # print(tokens)
                if tokens:  # make sure token list is not empty
                    sentences.append(tokens)
    return sentences

def train_word2vec(corpus, vector_size=100, window=5, min_count=1, workers=4, epochs=5):
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, 
                     min_count=min_count, workers=workers)
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    return model

def train():

    cutoff_year = 1990
    # input_corpus = ['./book/' + filename for filename in os.listdir('./book/') if int(filename.split('-')[-1].split('.')[0]) > cutoff_year]
    input_corpus = ['./book/' + filename for filename in os.listdir('./book/')]
    output_model = './word2vec/all.model'
    tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
    
    print("Loading and tokenizing corpus...")
    corpus = load_corpus(input_corpus, tokenizer)
    
    print("Training word2vec model...")
    model = train_word2vec(corpus)
    
    print("Saving model to", output_model)
    model.save(output_model)
    
if __name__ == "__main__":
    
    if True:
        train()
    if False:
        print("Loading word2vec model...")

        query_word = '女人'

        for model_name in ['before1990', 'after1990', 'all']:
            model = Word2Vec.load(f'./word2vec/{model_name}.model')
            print(f'{model_name}:')
            
            print(f"Finding words most similar to {query_word}'...")
            similar_words = model.wv.most_similar(query_word)
            for word, similarity in similar_words:
                print(f"{word}: {similarity:.4f}")