import pickle

imdb_path = '/Users/yiwenwang/Documents/CMU/20 Spring/ML/imdb/'

import string
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# stopwords_nltk = set(stopwords.words('english'))
stop = set()
for i in string.punctuation:
    stop.add(i)
    
entries = pickle.load(open(imdb_path+"entries.p", "rb"))

import re

def strip(string):
    return re.sub(' +', ' ', string)


def read_txt_str(filename, remove_punc=True):
    with open(filename) as file:
        data = file.read().replace('\n', ' ')
        data = data.replace('<br />', ' ')
        data = strip(data)
    if remove_punc:
        data = data.translate(str.maketrans('', '', string.punctuation))
    return data

def process(string, tokenizer=None, stemmer=None, lemmatizer=None, stop_words=None):
    words = string.lower()
    if tokenizer:
        words = tokenizer(words)
    if stop_words:
        words = [word for word in words if word not in stop]
    if lemmatizer:
        words = [lemmatizer.lemmatize(word) for word in words]
    if stop_words:
        words = [word for word in words if word not in stop]
    if stemmer:
        words = [stemmer.stem(word) for word in words]
    if stop_words:
        words = [word for word in words if word not in stop]
    return words

def get_X(key, tokenizer=None, stemmer=None, lemmatizer=None, stop_words=None):
    X = []
    for label in ['neg', 'pos']:
        for entry in entries[key][label]:
            filename = imdb_path + 'aclImdb/' + key + '/' + label + '/' + entry
            string = read_txt_str(filename)
            string_processed = process(string, tokenizer=tokenizer, stemmer=stemmer, lemmatizer=lemmatizer, stop_words=stop_words)
            X.append(string_processed)
    return X


X_train = get_X('train', tokenizer=word_tokenize, stop_words=stop)
X_test = get_X('test', tokenizer=word_tokenize, stop_words=stop)

pickle.dump(X_train, open(imdb_path+"word2vec/X_train.p", "wb"))
pickle.dump(X_test, open(imdb_path+"word2vec/X_test.p", "wb"))