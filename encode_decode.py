#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:24:06 2018

@author: zhanglida
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from act_cell6 import ACTCell
import tensorflow as tf
#from GRU import GRUcell
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, static_rnn
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
from tensorflow.python.ops import array_ops
from general_tf_utilities import initializerGRU,grucell,initializerAttnLSTM,attnlstmcell


class seq2seqModel(object):
    

    def __init__(self, config, is_training=False,batchnorm=False):
        keep_prob=0.8
        self.mu=0.1
        self.config = config
        self.stocks = stocks = config.stocks
        self.classes =classes= config.classes
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.encode_len = encode_len = config.num_steps
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        #vocab_size = config.vocab_size
        self.max_grad_norm = config.max_grad_norm
        self.use_lstm = config.use_lstm
        self.predict_num = predict_num = config.predict_num 
        self.min_act_step = config.min_act_step

        # Placeholders for inputs.
        self.input_data = tf.placeholder(tf.float32, [batch_size, encode_len+1,classes])
        #self.targets = tf.placeholder(tf.float32, [batch_size, predict_num+1,classes])
        self.initial_state = array_ops.zeros(tf.stack([self.batch_size, self.encode_len]),
                 dtype=tf.float32).set_shape([None, self.encode_len])

        #embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.hidden_size])

        # Set up ACT cell and inner rnn-type cell for use inside the ACT cell.
        

                
        inputs = self.input_data
        #with tf.variable_scope('norm',reuse=tf.AUTO_REUSE):
         #   inputs = tf.layers.batch_normalization(inputs,training=batchnorm)
        inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(inputs, encode_len+1, 1)]        
        
        target = self.input_data[:,1:,:]
        tar = tf.transpose(target, perm=[1, 0, 2]) 
        
        print('encode model building..')
        #with tf.variable_scope("encode", reuse=True):

        with tf.variable_scope('encoder',reuse=False):
            #encoder_cell = initializerGRU(config.hidden_size,self.classes,name='encoder')
            cell_gru = initializerGRU(self.hidden_size,classes,name='state_pre',if_output=True,noutput=classes)
            #encoder_cell[0]=tf.expand_dims(encoder_cell[0],0)
            s0=tf.tile(cell_gru[0],[self.batch_size])
            s0=tf.reshape(s0,[self.batch_size,-1])
            s00=[]
            for i in range(config.max_computation):
                s00.append(s0)

            states=[]
            output_all=[]
            losses_his=[]
            
            with tf.variable_scope("ACT_encoder"):
                act_encoder = ACTCell(keep_prob,batchnorm,self.config.hidden_size, cell_gru, config.epsilon,max_computation=config.max_computation, 
                                      batch_size=self.batch_size,noutput=classes, min_step=self.min_act_step)

            for i in range(encode_len):
                if i ==0:
                    #_,encode_state = grucell(inputs[i],*encoder_cell)
                    outputs,encode_state,loss = act_encoder(inputs[i],s00,tar[i])
                else:
                    tf.get_variable_scope().reuse_variables()    
                    #_,encode_state = grucell(inputs[i],states[-1],*encoder_cell[1:]) 
                    outputs,encode_state,loss = act_encoder(inputs[i],states[-1],tar[i])
                states.append(encode_state)
                output_all.append(outputs)
                losses_his.append(loss)
            
            #outputs, encode_final_state = static_rnn(encoder_cell, inputs, dtype = tf.float32)
            #output = outputs[-1]
        
        
        
                        
        predict = output_all
        predict=tf.transpose(predict, perm=[1, 0, 2])
        
        losses=[]
        for i in range(batch_size):
            ce=[]
            for j in range(self.encode_len):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target[i,j], logits=predict[i,j])
                ce.append(tf.reduce_sum(cross_entropy))
            losses.append(tf.reduce_sum(ce))

        
        ponder_cost0,Nt0 = act_encoder.calculate_ponder_cost(time_penalty=self.config.ponder_time_penalty)     
        #ponder_cost1,Nt1 = act_decoder.calculate_ponder_cost(time_penalty=self.config.ponder_time_penalty)
        
        Nt=Nt0 #50*32
        Nt=tf.transpose(Nt) #32*50
         
        
        self.cost = (tf.reduce_sum(losses) / batch_size)+ponder_cost0+self.mu*(tf.reduce_sum(losses_his) / batch_size)
        #self.loss = tf.reduce_sum(losses) / batch_size
        self.ponder = (ponder_cost0)/self.config.ponder_time_penalty
        self.final_state = tf.nn.softmax(predict,-1)
        self.Nt=Nt
        
        #self.gradient=tf.gradients(losses[0], inputs[0])
        #self.gradient=tf.gradients(losses, inputs)
        #self.gradient=tf.transpose(tf.gradients(self.cost, inputs), perm=[1, 0, 2])
        
        if is_training:              
            #self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

