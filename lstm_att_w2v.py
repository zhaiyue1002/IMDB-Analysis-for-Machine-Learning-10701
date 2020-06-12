path = '/home/u38430/phase_2/'
import sys
sys.path.insert(1, '/glob/intel-python/python3/lib/python3.6/site-packages')
sys.path.insert(2, '/home/u38430/phase_2')
import pickle
import numpy as np
import tensorflow as tf
padded_test = pickle.load(open(path+'padded_test.p', "rb"))
padded_train = pickle.load(open(path+'padded_train.p', "rb"))
max_len = padded_train.shape[1]

y_train = np.concatenate((np.zeros(12500), np.ones(12500)))
y_test = np.concatenate((np.zeros(12500), np.ones(12500)))

embedding_matrix = pickle.load(open(path+'w2v_300.p', "rb"))
EMBEDDING_DIM = embedding_matrix.shape[1]
num_words = embedding_matrix.shape[0]

from keras.regularizers import l2
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation
from attention import SeqSelfAttention
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras import optimizers

model = Sequential()
model.add(Embedding(num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False, mask_zero=True))

#questionalble
num_lstm = 100
num_dense = 50
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

model.add(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))
model.add(SeqSelfAttention(units=10, attention_type='multiplicative',attention_activation='sigmoid'))
model.add(LSTM(10, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False))
model.add(Dense(num_dense, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(rate_drop_dense))
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy',
        optimizer=adam,
        metrics=['acc'])

STAMP = 'lstm_w2v_att_masked'

from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
best_model_path = path + 'lstm_att/' + STAMP + '.h5'
model.load_weights(best_model_path)

# model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
# class_weight = {0: 1, 1: 1}

# history = model.fit(padded_train, y_train,
#                     validation_data=(padded_test, y_test),
#                     epochs=5, batch_size=128, shuffle=True, 
#                     class_weight=class_weight, callbacks=[early_stopping, model_checkpoint, TerminateOnNaN()])

# with open(path+'lstm_att/history_'+STAMP+'_2.p', 'wb') as f:
#     pickle.dump(history.history, f)
