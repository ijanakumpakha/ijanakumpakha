import os
import re

import pandas as pd

from gensim.models import Word2Vec, FastText

all_sentences = list()

all_words = list()


def pre_process_text(sentence):
    sentence = sentence.split()

    clean_sentences = list()

    for word in sentence:
        word = re.sub(r'[,/–-—()?;:’\'"‘“”`.।]', ' ', word)
        word = re.sub(r'\s+', ' ', word)
        word = word.replace('[', '')
        word = word.replace(']', '')

        all_words.append(word.strip())

        clean_sentences.append(word.strip())

    clean_sentences = ' '.join(clean_sentences).split()

    while '' in clean_sentences:
        clean_sentences.remove('')
    while ' ' in clean_sentences:
        clean_sentences.remove(' ')

    all_sentences.append(clean_sentences)

    return ' '.join(clean_sentences).strip()


if __name__ == '__main__':
    df = pd.read_csv('files/dataset.csv', encoding='utf-8', dtype=object).fillna('')
    category = pd.read_csv('files/nepali_category.csv', encoding='utf-8', dtype=object).fillna('')

    category_dict = dict(zip(category['english'], category['nepali']))
    df['category'] = df['category'].apply(lambda x: category_dict.get(x.lower()))
    print(df['category'])

    df['title'].apply(pre_process_text, 1)
    df['category'].apply(pre_process_text, 1)
    df['date'].apply(pre_process_text, 1)

    # model_fast_text = FastText(all_sentences, size=300, window=5, min_count=5, workers=4, sg=1, word_ngrams=3)
    # model_fast_text.wv.save_word2vec_format(Config.fast_text_embedding, binary=False)

    model = Word2Vec(all_sentences, min_count=1, size=300, sg=0, negative=15, window=10)
    model.wv.save_word2vec_format('files/word2vec_cbow.txt', binary=False)

    all_sentences = [' '.join(sent) for sent in all_sentences]

    vocabulary = pd.DataFrame()
    vocabulary['article_sentence'] = all_sentences
    vocabulary.to_csv('files/vocabulary.csv', encoding='utf-8', index=False)
