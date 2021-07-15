#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:30:32 2018

@author: zhanglida
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import static_rnn
from tensorflow.python.ops import variable_scope as vs
from general_tf_utilities import initializerGRU,grucell


class ACTCell(RNNCell):
    """
    A RNN cell implementing Graves' Adaptive Computation Time algorithm
    """
    def __init__(self, keep_prob,batchnorm, num_units, cell, epsilon,
                 max_computation, batch_size,noutput, sigmoid_output=False,min_step=1.0):

        self.batch_size = batch_size
        self.one_minus_eps = tf.fill([self.batch_size], tf.constant(min_step - epsilon, dtype=tf.float32))
        self._num_units = num_units #hidden_size  #input_size
        self.classes = classes = 5
        #self.cell_state = cell_state
        self.cell = cell
        self.max_computation = max_computation
        self.ACT_remainder = []
        self.ACT_iterations = []
        self.ACT_ht=[]
        #self.ACT_state=[]
        self.sigmoid_output = sigmoid_output
        self._output_size = noutput
        self.matrix_size=num_units*2

        self._state_is_tuple = False
        
        self.zeta=[]
        self.u=[]
        self.drop=keep_prob
        self.batchnorm=batchnorm
        with tf.variable_scope("state"):
            wh=tf.get_variable("wh", [self._num_units,1])
            vh=tf.get_variable("vh", [self._num_units,1])
            bh=tf.get_variable("bh", [1])
            self.wh=wh
            self.vh=vh
            self.bh=bh
            
            ws=tf.get_variable("ws", [self._num_units,self._num_units])
            vs=tf.get_variable("vs", [self._num_units,self._num_units])
            bs=tf.get_variable("bs", [self._num_units])
            self.ws=ws
            self.vs=vs
            self.bs=bs
            
            wq=tf.get_variable("wq", [self._num_units,1])
            vq=tf.get_variable("vq", [self._num_units,1])
            bq=tf.get_variable("bq", [1])
            self.wq=wq
            self.vq=vq
            self.bq=bq
            vn=tf.get_variable("vn", [self.max_computation,self.batch_size])
            self.vn=vn
            

    @property
    def input_size(self):
        return self._num_units#tf.divide(self._num_units,self.classes)
    @property
    def output_size(self):
        return self._output_size
    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state,target, timestep=0, scope=None):
        

        with vs.variable_scope(scope or type(self).__name__):
            # define within cell constants/ counters used to control while loop for ACTStep
            prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob") #[0,0,0...0] batch_size 0
            prob_compare = tf.zeros_like(prob, tf.float32, name="prob_compare") #[0,0,0...0] prob size (batch_size) 0
            counter = tf.zeros_like(prob, tf.float32, name="counter") #[0,0,0...0] prob size (batch_size) 0
            #acc_outputs = tf.fill([self.max_computation,self.batch_size, self.output_size], 0.0, name='output_accumulator')
            #acc_states = tf.zeros_like(state[0], tf.float32, name="state_accumulator")
            #acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            batch_mask = tf.fill([self.batch_size], True, name="batch_mask") #[true, true...true] batch_size true
            #p=tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "p")
            
            # While loop stops when this predicate is FALSE.
            # Ie all (probability < 1-eps AND counter < N) are false. 
            state_pre = state[-1]
            u_list=tf.zeros_like(state)
            output = tf.zeros_like(inputs)
            loss=0.0
            #zeta=tf.fill([15,self.batch_size,self.matrix_size], tf.constant(0.0, dtype=tf.float32), "u")
    
            def halting_predicate(batch_mask, prob_compare, prob,
                          counter, state, inputs,output, acc_output,u,target,loss):
                return tf.reduce_any(tf.logical_and(
                        tf.less(prob_compare,self.one_minus_eps),
                        tf.less(counter, self.max_computation)))

            # Do while loop iterations until predicate above is false.

            _,_,remainders,iterations,_,inputs,output,state_pre,u_all,target,loss = \
                tf.while_loop(halting_predicate, self.act_step,
                              loop_vars=[batch_mask, prob_compare, prob,
                                         counter, state, inputs,output, state_pre,u_list,target,loss])
        
        

        
        #accumulate remainder  and N values
        self.ACT_remainder.append(tf.reduce_mean(1 - remainders))
        #self.ACT_iterations.append(tf.reduce_mean(iterations))
        self.ACT_iterations.append(iterations)
        #self.ACT_ht.append(remainders)

        if self.sigmoid_output:
            output = tf.sigmoid(tf.contrib.rnn.BasicRNNCell._linear(output,self.batch_size,0.0))

        if self._state_is_tuple:
            next_c, next_h = tf.split(next_state, 2, 1)
            next_state = tf.contrib.rnn.LSTMStateTuple(next_c, next_h)
        #print(self.ACT_state,333333333)
        return output, u_all,loss #acc_states acc_outputs  self.ACT_state

    def calculate_ponder_cost(self, time_penalty):
        '''returns tensor of shape [1] which is the total ponder cost'''
        return time_penalty * tf.reduce_sum(
            tf.add_n(self.ACT_remainder)/len(self.ACT_remainder) +
            tf.to_float(tf.add_n(self.ACT_iterations)/len(self.ACT_iterations))),self.ACT_iterations#self.ACT_ht

    def act_step(self,batch_mask,prob_compare,prob,counter,state,inputs,output_pre,state_pre,u_list,target,loss):
        
        depth = tf.reduce_max(counter)
        
        vv = tf.reshape(depth,[1])
        idxsparse = tf.SparseTensor(indices=[[0,0]], values=vv, dense_shape=[1,3])
        idx = tf.sparse_tensor_to_dense(idxsparse)
        idx = tf.cast(idx,tf.int64)
        masksparse=tf.SparseTensor(indices=idx, values=[1.], dense_shape=[self.max_computation, 1,1])
        layer_mask = tf.sparse_tensor_to_dense(masksparse)
        
        #binary_flag = depth*tf.ones([self.batch_size, 1], dtype=tf.float32)
        

        
        depth = tf.cast(depth,tf.int32)
        #state_cur = tf.gather(state,depth)
        
        beta=[]
        for i in range(self.max_computation):
            beta_tin = tf.sigmoid(tf.matmul(state[i],self.wq)+tf.matmul(state_pre,self.vq)+self.bq)
            beta.append(beta_tin*tf.transpose([self.vn[depth]]))
        alpha = tf.exp(beta)/tf.reduce_sum(tf.exp(beta),0)
        state_tm1 = tf.reduce_sum(state*alpha,0)
        
        state_mix = tf.matmul(state_tm1,self.ws)+tf.matmul(state_pre,self.vs)+self.bs
        state_mix = tf.sigmoid(state_mix)
        
        output, new_state = grucell(inputs,state_mix,*self.cell[1:],if_output=True)
        #_, state_mid = grucell(output_pre,state_cur,*self.cell_state[1:],if_output=True)
        #output, state_x = grucell(inputs,state_mid,*self.cell_x[1:],if_output=True)
        #new_state = tf.matmul(state_mid,self.wa)+tf.matmul(state_x,self.va)+self.ba
        state_pre = new_state
        #output = tf.layers.batch_normalization(output,training=self.batchnorm)
        #output = tf.nn.dropout(output,self.drop)
        
        
        #p=tf.sigmoid(tf.matmul(new_state,self.wh)+tf.matmul(state_cur,self.vh)+self.bh)
        p=tf.sigmoid(tf.matmul(new_state,self.wh)+self.bh)
        p=tf.reshape(p,[self.batch_size])
        
        new_batch_mask = tf.logical_and(tf.less(prob + p, self.one_minus_eps), batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32) #true-->1.  false-->0.

        
        prob_compare += p * tf.cast(batch_mask, tf.float32)

        
        counter += tf.cast(batch_mask, tf.float32) #if previous round doesn't stop, then this round dept add 1

        prob += p * new_float_mask #pt

        
        #float_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)
        new_u=u_list+new_state*layer_mask
        
        ce=[]
        batch_mask_=tf.cast(batch_mask,tf.float32)
        for j in range(self.batch_size):
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=target[j], logits=output[j])
            ce.append(cross_entropy)
        ce_sum=tf.reduce_sum(ce*batch_mask_)
        loss += ce_sum
        
        
        return [new_batch_mask, prob_compare, prob, counter, state,inputs, output, state_pre, new_u,target,loss]