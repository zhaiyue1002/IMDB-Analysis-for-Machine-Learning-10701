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
from keras.models import Model
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Input
from attention import SeqSelfAttention
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras import optimizers
from keras.layers.merge import concatenate

embedding_layer = (Embedding(num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False, mask_zero=True))

#questionalble
# num_lstm = 100
num_dense = 100
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

sequences_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = embedding_layer(sequences_input)
lstm_layer = LSTM(128, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)
hidden_states = lstm_layer(embedded_sequences)

hidden_states_0 = SeqSelfAttention(attention_type='multiplicative',attention_activation='sigmoid')(hidden_states)
hidden_states_0 = LSTM(16, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False)(hidden_states_0)

hidden_states_1 = SeqSelfAttention(attention_type='multiplicative',attention_activation='sigmoid')(hidden_states)
hidden_states_1 = LSTM(16, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False)(hidden_states_1)

hidden_states_2 = SeqSelfAttention(attention_type='multiplicative',attention_activation='sigmoid')(hidden_states)
hidden_states_2 = LSTM(16, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False)(hidden_states_2)

merged = concatenate([hidden_states_0, hidden_states_1, hidden_states_2])

merged = Dense(num_dense, activation='relu')(merged)
merged = Dropout(rate_drop_dense)(merged)
predicted = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=sequences_input, outputs=predicted)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy',
        optimizer=adam,
        metrics=['acc'])

STAMP = 'multi'

from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
best_model_path = path + 'ulti/' + STAMP + '.h5'
model.load_weights(best_model_path)

model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
class_weight = {0: 1, 1: 1}

history = model.fit(padded_train, y_train,
                    validation_data=(padded_test, y_test),
                    epochs=5, batch_size=128, shuffle=True, 
                    class_weight=class_weight, callbacks=[early_stopping, model_checkpoint, TerminateOnNaN()])

with open(path+'ulti/history_'+STAMP+'_1.p', 'wb') as f:
    pickle.dump(history.history, f)
