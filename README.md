# A Very Brief Description of the Scripts Included

We divided the whole project into smaller tasks. Almost all the scripts include a lot of loading and saving by pickle. Therefore, it is almost impossible to run any of these scripts, but we hope they at least describe how we proceeded.

## Pre-processing

process.py
Raw text -> lists of words.

tokenizer.py
Lists of words -> padded sequences of tokens.

## Training word embeddings

train.py
Training fastText words embeddings on the text of the train and test sets.

matrix.py
Loading and saving word vectors into an embedding matrix that can be later integrated into a keras embedding layer.

## Attention

attention.py
An implementation of self-attention we found on https://github.com/CyberZHG/keras-self-attention.

## Models

Each of the following scripts trains the corresponding model listed in the final report. Here are only some of the best models.

lstm_att_w2v.py, LSTM + Attention + Word2Vec(300)
bilstm_att_w2v, Bi-LSTM + Attention + Word2Vec(300)
stacked.py, LSTM + Attention (stacked) + Word2Vec(300)
multi.py, LSTM + Attention (multi-head) + Word2Vec(300)
