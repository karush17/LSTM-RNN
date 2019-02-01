# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:25:14 2019

@author: Karush
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,LSTM
from keras.utils import np_utils

# DATA IMPORTING

data = np.zeros((10,126))

data = data.transpose()  
data[np.isnan(data)] = np.mean(data[~np.isnan(data)])      

# TRAINING LSTM CELLS

seq_length = 40 ## length of the sequence
n_states=8 ## quantization states

n_length = len(data);ref=np.zeros((1,1));gen_data = np.zeros((40,0));acc = np.zeros((0,1))
for j in range(0,126):  ##len(data[0,:])
    loss = 'categorical_crossentropy'
    data_seq = data[:,j];n_patterns =  n_length - seq_length
    interval = ((max(data_seq) - min(data_seq))/n_states).astype(np.float)
    ref=np.zeros((0,1))
    for q in range(1,n_states+1):
        ref = np.r_['0,2',ref,np.reshape(np.array([min(data_seq)+ interval*q]),(1,1))]
    ref = ref[:,0]
    data_seq = np.digitize(data_seq,ref)
    dataX = np.zeros((0,seq_length));dataY = np.zeros((0,1))  
    for i in range(0, n_patterns, 1):
        seq_in = data_seq[i:i + seq_length].astype(np.float)
        seq_out = data_seq[i + seq_length].astype(np.float)
        dataX = np.r_['0,2',dataX,seq_in]
        dataY = np.r_['0,2',dataY,seq_out]
    print ("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    y = np_utils.to_categorical(dataY)
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64,return_sequences=True)) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(y.shape[1], activation='softmax'))
    # load the network weights
    model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])
    model.fit(X,y,epochs=50,batch_size=1, verbose=1)
    
    # DATA GENERATION
    
    start = np.random.randint(0, len(dataX))
    pattern = dataX[start]
    pattern = np.reshape(pattern,(len(pattern),1))
    for i in range(1,seq_length,1):
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x, verbose=1)
        result = np.argmax(prediction)
        pattern = np.r_['0,2',pattern,np.reshape(np.array([result]),(1,1))].astype(np.float)
        pattern = pattern[1:len(pattern)]
    
    gen_feat = np.zeros((0,1))
    for i in range(0,len(pattern)):
        a = pattern[i].astype(np.int)
        feat = np.random.uniform(ref[a-2],ref[a-1])
        gen_feat = np.r_['0,2',gen_feat,np.reshape(np.array([feat]),(1,1))]
    
    scores = model.evaluate(X,y)
    acc = np.r_['0,2',acc,np.reshape([scores[1]],(1,1))]
    gen_data = np.c_[gen_data,gen_feat]
    
#gen_data = np.c_[gen_data,gen_feat]
np.savetxt("new_data.csv",gen_data,delimiter=",")





