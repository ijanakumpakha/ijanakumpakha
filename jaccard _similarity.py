import numpy as np
import pandas as pd

import re
import tensorflow as tf
from gensim.models import KeyedVectors
from numpy import zeros
from sklearn.metrics.pairwise import jaccard_score

Tokenizer = tf.keras.preprocessing.text.Tokenizer

vocabulary = pd.read_csv('files/vocabulary.csv', dtype=object, encoding='utf-8').fillna('')
vocabulary = vocabulary[['article_sentence']]
vocabulary = vocabulary['article_sentence'].tolist()
vocabulary = np.array(vocabulary)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(vocabulary)
vocab_size = len(tokenizer.word_index) + 1


def pre_process_text(sentence):
    if sentence is None:
        return ''

    sentence = sentence.split()

    clean_sentences = list()

    for word in sentence:
        word = re.sub(r'[,/–-—()?;:’\'"‘“”`.।]', ' ', word)
        word = re.sub(r'\s+', ' ', word)
        word = word.replace('[', '')
        word = word.replace(']', '')

        clean_sentences.append(word.strip())

    clean_sentences = ' '.join(clean_sentences).split()

    while '' in clean_sentences:
        clean_sentences.remove('')
    while ' ' in clean_sentences:
        clean_sentences.remove(' ')

    return ' '.join(clean_sentences).strip()


def tokenize(feature, feature_date, feature_category):
    feature = tokenizer.texts_to_sequences(feature)
    feature_date = tokenizer.texts_to_sequences(feature_date)
    feature_category = tokenizer.texts_to_sequences(feature_category)

    feature_vector = list()

    embedding_matrix = get_embedding()

    for i in range(0, len(feature)):
        length = len(feature[i])
        _temp = np.sum(embedding_matrix[feature[i]], 0)
        _temp = np.divide(_temp, length)

        feature_vector_date = np.sum(embedding_matrix[feature_date[i]], 0)
        feature_vector_date = np.divide(feature_vector_date, len(feature_date[i]))

        feature_vector_category = np.sum(embedding_matrix[feature_category[i]], 0)
        feature_vector_category = np.divide(feature_vector_category, len(feature_category[i]))

        _temp += feature_vector_date
        _temp += feature_vector_category
        _temp = np.divide(_temp, 3)

        feature_vector.append(_temp)

    feature_vector = np.array(feature_vector)

    return feature_vector


def get_embedding():
    embedding_model = get_embedding_model()
    embedding_matrix = zeros((vocab_size, 300))

    for word, index in tokenizer.word_index.items():
        try:
            embedding_vector = embedding_model.get_vector(word)
            embedding_matrix[index] = embedding_vector
        except KeyError:
            pass

    return embedding_matrix


def get_embedding_model():
    return KeyedVectors.load_word2vec_format('files/word2vec_cbow.txt', binary=False)


dataset = pd.read_csv('files/dataset.csv', dtype=object, encoding='utf-8').fillna('')
dataset = dataset[['title', 'category', 'date']]

df_category = pd.read_csv('files/nepali_category.csv', encoding='utf-8', dtype=object).fillna('')
category_dict = dict(zip(df_category['english'], df_category['nepali']))

dataset['title'] = dataset['title'].apply(pre_process_text, 1)
dataset['category'] = dataset['category'].apply(pre_process_text, 1)
dataset['date'] = dataset['date'].apply(pre_process_text, 1)

dataset['category'] = dataset['category'].apply(lambda x: category_dict.get(x.lower()), 1)

date = dataset['date'].tolist()
category = dataset['category'].tolist()
dataset = dataset['title'].tolist()

tokenized_dataset = np.array(dataset)
tokenized_date = np.array(date)
tokenized_category = np.array(category)

tokenized_dataset = tokenize(tokenized_dataset, tokenized_date, tokenized_category)

similarity = jaccard_score(tokenized_dataset)

title = 'सरकारद्वारा लाल आयोगको प्रतिवेदन सार्वजनिक गर्ने प्रतिबद्धता राजपाको अवरोध खुल्यो'

title_index = dataset.index(title)

similar_titles = list(enumerate(similarity[title_index]))

sorted_similar_titles = sorted(similar_titles, key=lambda x: x[1], reverse=True)

for titles in sorted_similar_titles[:20]:
    print(dataset[titles[0]])
    print(titles[1])

