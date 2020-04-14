# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:39:33 2020

@author: mznid
"""

import pandas as pd
import numpy as np
import os
import random
import math
import itertools


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras   
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding,GRU, LSTM,Bidirectional,TimeDistributed,AveragePooling1D,GlobalMaxPool1D,Reshape,Input,Concatenate,concatenate,Attention,Permute, Lambda,RepeatVector,Add
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import seaborn as sb

import nltk
from gensim.models import Word2Vec
from string import punctuation



# UNLIST GPU OPTION FOR TENSORFLOW 2.1. GPU RUNS OUT OF MEMORY WITH THIS NN MODEL AND EVEN RUNS SLOWER THAN CPU 

tf.config.set_visible_devices([], 'GPU')


import tensorboard
import datetime

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# LOAD AND CLEAN

raw = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_train.csv")
rawtest = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_test.csv")
rawdev = pd.read_csv("D:\\stanford-natural-language-inference-corpus\\snli_1.0_dev.csv")




#for ind,each in enumerate(sentences2):
#    if type(each) == float:
#        print(ind)
nullset = {91479,91480,91481,311124,311125,311126}        
nonnullset = set(range(0,len(raw.iloc[:,0]))) - nullset
raw = raw.iloc[sorted(list(nonnullset)),:]

drawlist = []
for ind, each in enumerate(raw.iloc[:,0]):
    if each == '-':
        drawlist.append(ind)
        
raw = raw.iloc[sorted(list((set(range(0,len(raw.iloc[:,0]))) - set(drawlist)))),:]


drawlist = []
for ind, each in enumerate(rawtest.iloc[:,0]):
    if each == '-':
        drawlist.append(ind)
        
rawtest = rawtest.iloc[sorted(list((set(range(0,len(rawtest.iloc[:,0]))) - set(drawlist)))),:]


drawlist = []
for ind, each in enumerate(rawdev.iloc[:,0]):
    if each == '-':
        drawlist.append(ind)
        
rawdev = rawdev.iloc[sorted(list((set(range(0,len(rawdev.iloc[:,0]))) - set(drawlist)))),:]




#########################################################################

# DOWNSAMPLE

#sampleindex = random.sample(range(0,len(raw.iloc[:,0])), round(0.5 * len(raw.iloc[:,0]))) # 0.05
#raw = raw.iloc[sampleindex,:]
#raw.index = range(0,len(raw.iloc[:,0]))

#########################################################################


sentences1 = list(raw.iloc[:,5])
sentences2 = list(raw.iloc[:,6])
labellist = list(raw.iloc[:,0])
labelset = set(labellist)

labellisttest = list(rawtest.iloc[:,0])
sentences1test = list(rawtest.iloc[:,5])
sentences2test = list(rawtest.iloc[:,6])
labellistdev = list(rawdev.iloc[:,0])
sentences1dev = list(rawdev.iloc[:,5])
sentences2dev = list(rawdev.iloc[:,6])





# HANDLE LABELS

labeltrain = np.array(labellist)
labeltest = np.array(labellisttest)
labeldev = np.array(labellistdev)

labelkey = list(labelset)
labelvalue = list(range(0, len(labelkey)))
labeldict = {'contradiction': 0, 'entailment': 1, 'neutral' : 2}



ltrainn = []
ltestn = []
for each in labeltrain:
  ltrainn.append(labeldict[each])
for each in labeltest:
  ltestn.append(labeldict[each])
labeltrainhot = to_categorical(ltrainn)
labeltesthot = to_categorical(ltestn)

ldevn = []
for each in labeldev:
    ldevn.append(labeldict[each])
labeldevhot = to_categorical(ldevn)


# PARSE SENTENCES WITH NLTK pos_tag TO CREATE SET OF WORDS, ENSURING NO HOMONYM CONFUSION

wordset = set()


for each in sentences1:
    for every in nltk.word_tokenize(each.lower()):
        wordset.add(every)

for each in sentences2:
    for every in nltk.word_tokenize(each.lower()):
        wordset.add(every)





wordlist = list(wordset)
worddict = {}
for ind,each in enumerate(wordlist): 
    worddict[each] = ind    


# ADD SPECIAL TAGS TO WORD DICTIONARY
 
worddict = {k:(v+4) for k, v in worddict.items()}
worddict["<PAD>"] = 0
worddict["<START>"] = 1
worddict["<END>"] = 2
worddict["<UNK>"] = 3
worddict["<DELIM>"] = 4


  
# CREATE LISTS OF NLTK POS TUPLE IDs TO REPRESENT SENTENCES   (TRY NO POS TAG TO REDUCE VOCAB SIZE)

#w2vlistedsentence1 = []
#w2vlistedsentence2 = []

listedsentence1 = []
for each in sentences1:
    temp = [1]
    #tempw2v = []
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        #x,y = every
        temp.append(worddict[every])
        #tempw2v.append(x+y)
    #w2vlistedsentence1.append(tempw2v)
    listedsentence1.append(temp)
        
listedsentence2 = []    
for each in sentences2:
    temp = [1]
    #tempw2v = []
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        #x,y = every
        temp.append(worddict[every])
        #tempw2v.append(x+y)
    #w2vlistedsentence2.append(tempw2v)
    listedsentence2.append(temp)
    



listedsentences1test = []
for each in sentences1test:
    temp = [1]
    #tempw2v = []
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        #x,y = every
        if every not in worddict:
            temp.append(3)                         # 3 is the index of "<UNK>"
        else:
            temp.append(worddict[every])
        #tempw2v.append(x+y)
    #w2vlistedsentence1.append(tempw2v)
    listedsentences1test.append(temp)
        
listedsentences2test = []    
for each in sentences2test:
    temp = [1]
    #tempw2v = []
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        #x,y = every
        if every not in worddict:
            temp.append(3)
        else:
            temp.append(worddict[every])
        #tempw2v.append(x+y)
    #w2vlistedsentence2.append(tempw2v)
    listedsentences2test.append(temp)
    

listedsentences1dev = []
for each in sentences1dev:
    temp = [1]
    #tempw2v = []
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        #x,y = every
        if every not in worddict:
            temp.append(3)
        else:
            temp.append(worddict[every])
        #tempw2v.append(x+y)
    #w2vlistedsentence1.append(tempw2v)
    listedsentences1dev.append(temp)
        
listedsentences2dev = []    
for each in sentences2dev:
    temp = [1]
    #tempw2v = []
    for every in nltk.word_tokenize(each.lower()): #nltk.pos_tag(nltk.word_tokenize(each.lower()))
        #x,y = every
        if every not in worddict:
            temp.append(3)
        else:
            temp.append(worddict[every])
        #tempw2v.append(x+y)
    #w2vlistedsentence2.append(tempw2v)
    listedsentences2dev.append(temp)
    




# CREATE INVERSE WORD DICTIONARY

reverseworddict = dict([(value,key) for (key, value) in worddict.items()])



# PADDING


# treat maxlens differently to reduce padding noise? 

maxlen1 = 24
maxlen2 = 24

listedsentence1 = pad_sequences(listedsentence1, maxlen=maxlen1, value=worddict["<PAD>"], padding='pre')  # maxlen
listedsentence2 = pad_sequences(listedsentence2, maxlen=maxlen2, value=worddict["<PAD>"], padding='post')   # maxlen


listedsentences1test = pad_sequences(listedsentences1test, maxlen=maxlen1, value=worddict["<PAD>"], padding='pre')  # maxlen
listedsentences2test = pad_sequences(listedsentences2test, maxlen=maxlen2, value=worddict["<PAD>"], padding='post')   # maxlen


listedsentences1dev = pad_sequences(listedsentences1dev, maxlen=maxlen1, value=worddict["<PAD>"], padding='pre')  # maxlen
listedsentences2dev = pad_sequences(listedsentences2dev, maxlen=maxlen2, value=worddict["<PAD>"], padding='post')   # maxlen


#################

# TURN DATA TO NUMPY AND SPLIT TEST/TRAIN


new1 = np.array(listedsentence1)
new2 = np.array(listedsentence2)

new1test = np.array(listedsentences1test)
new2test = np.array(listedsentences2test)

new1dev = np.array(listedsentences1dev)
new2dev = np.array(listedsentences2dev)


train1 = new1
train2 = new2
test1 = new1test
test2 = new2test
dev1 = new1dev
dev2 = new2dev


train1.shape

####################################################

# ROUND DOWN TRAIN AND TEST SETS TO BE DIVISIBLE BY BATCH SIZE (RNNs seem to require exactness, at least in stateful mode)

BATCH_SIZE = 64

trainslice = math.floor(len(train1)/BATCH_SIZE) * BATCH_SIZE
testslice = math.floor(len(test1)/BATCH_SIZE) * BATCH_SIZE
devslice = math.floor(len(dev1)/BATCH_SIZE) * BATCH_SIZE


train1 = train1[:trainslice]
train2 = train2[:trainslice]
test1 = test1[:testslice]
test2 = test2[:testslice]
dev1 = dev1[:devslice]
dev2 = dev2[:devslice]

labeltrainhot = labeltrainhot[:trainslice]
labeltesthot = labeltesthot[:testslice]
labeldevhot = labeldevhot[:devslice]

# CREATE MODEL

def concat_in_out(X, Y):
    numex = X.shape[0]  # num examples
    glue = worddict["<DELIM>"] * np.ones(numex).reshape(numex, 1)
    inp_train = np.concatenate((X, glue, Y), axis=1)
    return inp_train

net_train = concat_in_out(train1, train2)
net_test = concat_in_out(test1, test2)
net_dev = concat_in_out(dev1,dev2)

N = maxlen1 + maxlen2 + 1 # 1 for delim   #2 * maxlen + 1
L = maxlen1    # maxlen


def get_Y(X, xmaxlen):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim
def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans
def get_R(X):
    Y = X[0]
    alpha = X[1]
    ans = tf.keras.backend.batch_dot(Y, alpha)   #K.T.batched_dot(Y, alpha) #tf.keras.backend.batch_dot(Y, alpha)   # Theano keras  batched_dot ?
    return ans


def Create_Model(length, vocab_size):
    
	# INPUT, EMBEDDING
    inputs0 = Input(batch_shape=(BATCH_SIZE, len(net_train[0])))
    embedding1 = Embedding(vocab_size, 150)(inputs0)
    drop_out1 = Dropout(0.1, name='dropout1')(embedding1)
    
    # 2 PARTs FOR BIDIRECTIONAL LSTM
    lstm_fwd = LSTM(150, return_sequences=True, name='lstm_fwd')(drop_out1)
    lstm_bwd = LSTM(150, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out1)
    
    # CONCAT 2 PARTS FOR BILSTM
    bilstm = concatenate([lstm_fwd, lstm_bwd], name='bilstm') # , mode='concat'
    drop_out2 = Dropout(0.1)(bilstm)
    
    # "LAST ELEMENT FROM TIME DIM"
    h_n = Lambda(get_H_n, output_shape=(300,), name="h_n")(drop_out2)
    Whn = Dense(300, kernel_regularizer = l2(0.01), name="Wh_n")(h_n)
    Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
    
    # "FIRST first-sentence-MAXLEN ELEMENTS FROM TIME DIM"
    Y = Lambda(get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, 300))(drop_out2)
    WY = TimeDistributed(Dense(300, kernel_regularizer = l2(0.01)), name="WY")(Y)
    
    # CONCAT 2 LAMBA "ATTENTIONS"?
    merged1 = Add()([Whn_x_e, WY])    #  concatenate([Whn_x_e, WY], name="merged1", mode='sum')
    M = Activation('tanh', name="M")(merged1)
    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    
    # FLATTEN AND GATE
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

    # CHANGE SHAPE OF Y
    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)
    
    # CONCAT CHANGED Y WITH MAIN PATHWAY OF NN
    r_ = Lambda(get_R,  output_shape=(300, 1))([Y_trans,alpha]) #, output_shape=(300, 1)
    #r_ = concatenate([Y_trans, alpha], output_shape=(300, 1), name="r_", mode=get_R)
    r = Reshape((300,), name="r")(r_)
    Wr = Dense(300, kernel_regularizer=l2(0.01))(r)
    
    # CALL ALL THE WAY BACK TO h_n... god knows why
    Wh = Dense(300, kernel_regularizer=l2(0.01))(h_n)
    
    # MERGE MAIN PATHWAY OF NN WITH CALLBACK LAYER TO h_n
    merged2 = Add()([Wr, Wh])   #  concatenate([Wr, Wh], mode='sum')
    h_star = Activation('tanh')(merged2)
    outputs = Dense(3, activation='softmax')(h_star)
    
    model = Model(inputs=[inputs0], outputs=outputs)
    

    
    model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel2.png')
    return model


#################################################### 

# TRAIN MODEL

length = len(listedsentence1[0])
vocab_size = len(worddict)

model = Create_Model(length, len(reverseworddict))   #vocab_size

model.fit([net_train], labeltrainhot, epochs=6, batch_size=BATCH_SIZE, validation_data=([net_test], labeltesthot), callbacks=[tensorboard_callback])


####################################################

# CREATE MODEL PREDICTIONS FOR CONFUSION MATRIX

nnpredictions = model.predict([net_test])

testlabellist = list(labeltest[:testslice])

labeldictreverse = { 0 : 'contradiction', 1 : 'entailment', 2 : 'neutral'}



preds = []
for each in nnpredictions:
    index = np.argmax(each)
    preds.append(labeldictreverse[index])
correct = []    
for each in range(0,len(preds)):
    if preds[each] == testlabellist[each]:
        correct.append(1)
    else:
        correct.append(0)    
accuracy = sum(correct)/len(correct)
print(accuracy)        


len(preds)
cf = pd.crosstab(np.array(preds), np.array(testlabellist))



sb.heatmap(cf, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Actual')
plt.ylabel('Prediction')




nnpredictions = model.predict([net_dev])

testlabellist = list(labeldev[:devslice])

labeldictreverse = { 0 : 'contradiction', 1 : 'entailment', 2 : 'neutral'}



preds = []
for each in nnpredictions:
    index = np.argmax(each)
    preds.append(labeldictreverse[index])
correct = []    
for each in range(0,len(preds)):
    if preds[each] == testlabellist[each]:
        correct.append(1)
    else:
        correct.append(0)    
accuracy = sum(correct)/len(correct)
print(accuracy)        


len(preds)
cf = pd.crosstab(np.array(preds), np.array(testlabellist))



sb.heatmap(cf, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Actual')
plt.ylabel('Prediction')



