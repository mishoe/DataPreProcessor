import nltk
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from tqdm import tqdm
from collections import Counter
import re
import heapq
import tensorflow as tf
import pickle as p

ngram_count = 1
num_words_to_keep = 20000
num_ngrams_to_keep = 15000

def string_cleaner(text):
    # Clean the documents
    stop = set(stopwords.words('english') + stopwords.words('spanish') + stopwords.words('french'))
    exclude = string.punctuation
    wordnet_lemmatizer = WordNetLemmatizer()
    start_strip_word = ['abstract', 'background', 'summary', 'objective']
    text = str(text).lower() # downcase
    for word in start_strip_word:
        if text.startswith(word):
            text = text[len(word):]
    tokens = nltk.tokenize.word_tokenize(text) # split string into words (tokens)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stop] # remove stopwords
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    return tokens

def fit_transform_bow_mappings(text_tokens, num_words_to_keep=5000, transform_mode = "count"):
    ## fit() part of the function
    corpuses = [" ".join(tokens) for tokens in text_tokens]
    t = tf.keras.preprocessing.text.Tokenizer(num_words=(num_words_to_keep+1))
    t.fit_on_texts(corpuses)
    word_counts = {key:value for key, value in sorted(t.word_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[:num_words_to_keep]}
    feature_names = list(word_counts.keys())
    ## transform() part of the function
    out_matrix = pd.DataFrame(t.texts_to_matrix(corpuses, mode=transform_mode)[:,1:],columns=feature_names)
    return out_matrix, feature_names, word_counts

def fit_ngram_mappings(text_tokens,ngram_count = 2):

    ngram_mapping_dict = {}
    for corpus in tqdm(text_tokens):
        for i in range(len(corpus) - ngram_count):
            seq = ' '.join(corpus[i:i + ngram_count])
            if seq not in ngram_mapping_dict.keys():
                ngram_mapping_dict[seq] = []
            ngram_mapping_dict[seq].append(corpus[i + ngram_count])

    ngram_counts = {}
    for (key, values) in ngram_mapping_dict.items():
        ngram_counts[key] = dict(Counter(values))

    return ngram_mapping_dict, ngram_counts

def trim_ngram_dict(ngram_mapping_dict, ngram_counts, num_ngrams_to_keep=5000):
    all_counts = []
    for count_dict in ngram_counts.values():
        # print(count_dict)
        all_counts.extend(list(count_dict.values()))
    keep_count = heapq.nlargest(num_ngrams_to_keep, all_counts)[-1]

    current_count_words = 0
    for (key, values) in tqdm(ngram_counts.items()):
        trimmed_values = values.copy()
        for (k, v) in values.items():
            if v < keep_count or current_count_words >= num_ngrams_to_keep:
                trimmed_values.pop(k)
            else:
                current_count_words += 1
        # if there is still a values in the dictionary re-assign the ngram with the trimmed dictionary
        # else, pop the current ngram, as there are no words that meet the current minimum count
        if trimmed_values:
            ngram_mapping_dict[key] = list(trimmed_values.keys())
        else:
            ngram_mapping_dict.pop(key)

    ngram_columns = []
    for (key, values) in tqdm(ngram_mapping_dict.items()):
        for value in values:
            ngram_columns.append(" ".join([key, value]))
    return ngram_mapping_dict, ngram_columns

def transform_ngram_mappings(text_tokens,ngram_columns,ngram_count):
    # create the document-by-ngram matrix
    rows_list = []
    column_index_dict = {col:ind for (col,ind) in zip(ngram_columns,range(len(ngram_columns))) }
    for i in tqdm(range(len(text_tokens))):
        row = np.zeros(len(ngram_columns))
        tokens = text_tokens[i]
        for j in range(len(tokens) - ngram_count):
            seq = ' '.join(tokens[j:j + ngram_count + 1])
            # does the seq (of ngrams) exist in the ngram_columns?
            if seq in column_index_dict:
                row[column_index_dict[seq]] += 1
        rows_list.append(row)
    return pd.DataFrame(np.matrix(rows_list),columns=ngram_columns)


# ngram_mapping_dict,ngram_count_dict = fit_ngram_mappings(text_tokens,ngram_count=ngram_count)
# final_ngram_mapping_dict, ngram_columns = trim_ngram_dict(ngram_mapping_dict, ngram_count_dict, num_ngrams_to_keep=num_ngrams_to_keep)
# ngram_count_matrix = transform_ngram_mappings(text_tokens,ngram_columns,ngram_count)
# bow_count_matrix, bow_columns, bow_word_dict = fit_transform_bow_mappings(text_tokens, num_words_to_keep, transform_mode = "count")
