import pickle
X_train = pickle.load(open('X_train.p', "rb"))
X_test = pickle.load(open('X_test.p', "rb"))
X = X_train + X_test

from gensim.models.fasttext import FastText 

model = FastText(size=100, window=5, min_count=1)
model.build_vocab(sentences=X)
model.train(sentences=X, total_examples=len(X), epochs=10)

model.save("fasttext_100.model")
