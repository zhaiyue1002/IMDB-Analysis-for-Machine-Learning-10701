import pickle
import numpy as np
tokenizer = pickle.load(open("../word2vec/tokenizer.p", "rb"))

word_index = tokenizer.word_index
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, 100))

from gensim.models.fasttext import FastText 
model = FastText.load("fasttext_100.model")

for word, i in word_index.items():
    embedding_matrix[i] = model.wv[word]

with open('embedding_fasttext.p', 'wb') as f:
    pickle.dump(embedding_matrix, f)

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))