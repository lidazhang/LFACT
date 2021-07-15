#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 20:04:52 2018

@author: zhanglida
"""

"""Utilities for parsing PTB text files."""


import collections
import os

import numpy as np
import csv
import pickle as pkl
from random import randint
import gc
from sklearn.preprocessing import OneHotEncoder


def ptb_raw_data(data_path,config):
    
    pred_len=config.predict_num
    encoder_len=config.num_steps
    batch_size=config.batch_size
    step = 50
    dim=config.classes
    
    print('date preparing')
    with open(data_path) as f:
        raw_data = f.read()
    alldata=np.fromstring(raw_data, dtype=np.uint8)
    unique, data = np.unique(alldata, return_inverse=True)
    print(len(data),111)
    raw_data=data
    
    
    onehot = np.zeros((len(data), dim))
    print(onehot.shape,222)
    for i in range(len(data)):
        onehot[i,data[i]]=1
    #onehot[np.arange(len(data)), data] = 1
    print(onehot.shape,333)
    return onehot
    '''
    input_data=[]
    target=[]
    encoder_len=config.num_steps
    predict_len=config.predict_num    
    step=2
    step_num=(len(data)-encoder_len-predict_len)//step
    
    for i in range(step_num):
        input_data.append(onehot[i*step:i*step+encoder_len])
        target.append(onehot[i*step+encoder_len:i*step+encoder_len+predict_len])
    
    input_data=np.array(input_data)
    target=np.array(target)
    print(input_data.shape)
    print(target.shape)
    return input_data,target
    '''
    
    '''
    
    raw_data = np.array(raw_data)
    raw_batch_len = int(len(raw_data)//batch_size)
    batch_data = np.reshape(raw_data,[batch_size,raw_batch_len])
    
    epoch_size=int((raw_batch_len-pred_len-encoder_len)//step)-1
    for i in range(epoch_size):
        x=np.zeros([batch_size, encoder_len, dim])
        y=np.zeros([batch_size, pred_len, dim])
        for ii in range(batch_size):
            onehot_x = np.zeros((encoder_len, dim))
            onehot_y = np.zeros((pred_len, dim))
            onehot_x[np.arange(encoder_len), batch_data[ii,i*step:i*step+encoder_len]] = 1
            onehot_y[np.arange(pred_len), batch_data[ii,i*step+encoder_len:i*step+encoder_len+pred_len]] = 1
            x[ii]=onehot_x
            y[ii]=onehot_y
    print('finished')
    return epoch_size
    '''
        


def ptb_iterator(raw_data, index, batch_size, num_steps, config):
    
    encoder_len=config.num_steps
    dim=config.classes #205 #34
    
    batch_num=len(index)//batch_size
    
    #for i in range(1):
    for i in range(batch_num):
        x=np.zeros([batch_size, encoder_len+1, dim])
        for ii in range(batch_size):
            pos = index[i*batch_size+ii]
            x[ii]=raw_data[pos*encoder_len:(pos+1)*encoder_len+1].toarray()
            #y[ii]=raw_data[a*step+encoder_len+start:a*step+encoder_len+pred_len+start+1].toarray()

        yield (x)
        del x
        #del y
        gc.collect() 
  


