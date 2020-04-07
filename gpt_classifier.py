#please install this:   pip install tensorflow==1.14.0

data_path = "/Users/nouransoliman/Desktop/nlp project/data/train/transcription/"

import os
import re
import operator
import pickle 
import torch
from transformers import *
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Embedding
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold

np.random.seed(42)

def clean_file(lines):
  return re.sub(r'[0-9]+[_][0-9]+', '', lines.replace("*INV:", "").replace("*PAR:", "")).strip().replace("\x15", "").replace("\n", "").replace("\t", " ").replace("[+ ", "[+").replace("[* ", "[*").replace("[: ", "[:").replace(" .", "").replace("'s", "").replace(" ?", "").replace(" !", "").replace(" ]", "]").lower()

def extra_clean(lines):
    lines = clean_file(lines)
    lines = lines.replace("[+exc]", "")
    lines = lines.replace("[+gram]", "")
    lines = lines.replace("[+es]", "")
    lines = re.sub(r'[&][=]*[a-z]+', '', lines) #remove all &=text
    lines = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', lines) #remove all [*char:char(s)]
    lines = re.sub(r'[^A-Za-z0-9\s_]+', '', lines) #remove all remaining symbols except underscore
    lines = re.sub(r'[_]', ' ', lines) #replace underscore with space
    return lines

directcc = data_path+"cc_txt/"
directcd = data_path+"cd_txt/"

datacc = []
datacd = []
dataall = []


for filename in os.listdir(directcc):
    if filename.endswith(".txt") and filename != "cleaned_cc.txt":
      f = open(directcc+filename)
      lines = f.read()
      st = extra_clean(lines)
      datacc.append(st)
      dataall.append(st)


for filename in os.listdir(directcd):
    if filename.endswith(".txt") and filename != "cleaned_cd.txt":
      f = open(directcd+filename)
      lines = f.read()
      st = extra_clean(lines)
      datacd.append(st)
      dataall.append(st)

print(len(datacc), len(datacd))

model_class, tokenizer_class, pretrained_weights = (GPT2Model, GPT2Tokenizer, 'gpt2')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
gpt_model = model_class.from_pretrained(pretrained_weights)
gpt_model_head = GPT2LMHeadModel.from_pretrained(pretrained_weights)
word_embeddings = gpt_model_head.transformer.wte.weight
position_embeddings = gpt_model_head.transformer.wpe.weight

num_classes = 2
n_split = 5
epochs = 30
batch_size = 4
gpt_dim = 768
max_len = 715

def create_model():
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=word_embeddings.shape[0], output_dim=gpt_dim, weights=[word_embeddings.detach()], input_length=max_len, mask_zero=True, trainable=True))
    forward_layer = layers.LSTM(128, recurrent_activation='sigmoid', activation='tanh', return_sequences=False)
    backward_layer = layers.LSTM(128, recurrent_activation='sigmoid', activation='tanh', return_sequences=False, go_backwards=True)
    model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='concat', input_shape=(gpt_dim, )))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

model = create_model()
print(model.summary())

#Data prep
X_cc = datacc
y_cc = np.zeros((len(X_cc), 2))
y_cc[:,0] = 1

X_cd = datacd
y_cd = np.zeros((len(X_cd), 2))
y_cd[:,1] = 1

dataall_conv = []

for entry in dataall:
  dataall_conv.append(tokenizer.encode(entry, add_special_tokens=True))

print(len(dataall_conv))

X = pad_sequences(dataall_conv, maxlen=max_len, padding='pre', value=0)
y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
print(X.shape)
p = np.random.permutation(len(X))
X = X[p]
y = y[p]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    i = 0
    for train_index, val_index in KFold(n_split, shuffle=True).split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print(x_train.shape, y_train.shape)
        model = create_model()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['categorical_accuracy'])
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val))
        score = model.evaluate(x_val, y_val, verbose=0)
        print('Val accuracy:', score)
        i += 1
        model.save_weights(data_path+'model_gpt_w_f'+i+'.h5')