import pickle

X_train = pickle.load(open('X_train.p', "rb"))
X_test = pickle.load(open('X_test.p', "rb"))
X = X_train + X_test


texts_train = [' '.join(i) for i in X_train]
texts_test = [' '.join(i) for i in X_test]

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=200000)
tokenizer.fit_on_texts(texts_train + texts_test)

sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

max_len = 0
for i in sequences_train:
    max_len = max(max_len, len(i))
for i in sequences_test:
    max_len = max(max_len, len(i))

padded_train = pad_sequences(sequences_train, maxlen=max_len)
padded_test = pad_sequences(sequences_test, maxlen=max_len)

with open('tokenizer.p', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('padded_train.p', 'wb') as f_train:
    pickle.dump(padded_train, f_train)

with open('padded_test.p', 'wb') as f_test:
    pickle.dump(padded_test, f_test)