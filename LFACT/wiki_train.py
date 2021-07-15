 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:44:45 2018

@author: zhanglida
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import config as cf
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import datetime
import random
#from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import math

from encode_decode import seq2seqModel
from epoch import run_epoch
import saveload
from reader4 import ptb_raw_data

random.seed(1)
tf.set_random_seed(1)
np.random.seed(1)

#conf = tf.ConfigProto()
#conf.gpu_options.per_process_gpu_memory_fraction = 0.95



def get_config(conf):
    if conf == "small":
        return cf.SmallConfig
    elif conf == "medium":
        return cf.MediumConfig
    elif conf == "large":
        return cf.LargeConfig
    elif conf == "titanx":
        return cf.TitanXConfig
    else:
        raise ValueError('did not enter acceptable model config:', conf)

def main(unused_args):

    config = get_config(FLAGS.model_size)
    eval_config = get_config(FLAGS.model_size)
    saved_model_path = FLAGS.model_path
    weights_dir = FLAGS.weights_dir
    verbose = FLAGS.verbose
    debug = FLAGS.debug
    print("config.ponder_time_penalty:",config.ponder_time_penalty)
    print("config.min_act_step:",config.min_act_step)

    if weights_dir is not None:
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
    
    train_loss_ite=[]
    val_loss_ite=[]
    test_loss_ite=[]
    fmeasure=[]
    
    modelname = 'wiki256_10M_LFACTx_sigmoid_loss_right_rnn'
    print(modelname)
    
    testset = 9
    modelpath=None#os.path.join(FLAGS.data_path, 'ACTmodel/seqACT20to1feedmodel8.pkl')
    iteration = 10
    for ite in [0]:
        train_loss_oneite=[]
        val_loss_oneite=[]
        bpc_min = 999999
        print('ite:',ite)
        
        
        #datapath = '/Users/zhanglida/Desktop/deeplearning/week30/enwik8'
        datapath = os.path.join(FLAGS.data_path, 'enwik8')
        with open(datapath) as f:
            raw_data = f.read()
        alldata=np.fromstring(raw_data, dtype=np.uint8)
        #unique, data = np.unique(alldata, return_inverse=True)
        data= alldata
        
        length=len(data)
        sample_len=12800
        index=random.sample(range(length//50-1), sample_len)
        
        col=alldata
        row=range(len(alldata))
        d=np.ones(len(alldata))
        onehot = sparse.csr_matrix((d,(row,col)),shape=(len(alldata),256))
        
        train_index=index[:int(sample_len*.8)]        
        val_index=index[int(sample_len*.8):int(sample_len*.9)]        
        test_index=index[int(sample_len*.9):]
        
        del data
        del alldata
        
    
        with tf.Graph().as_default(), tf.Session() as session:
            tf.set_random_seed(-1)
            saved_model_path=modelpath
            Nt1=[]
            Nt2=[]
            Nt3=[]
            #saved_model_path=os.path.join(FLAGS.data_path, 'ACTmodel/model'+str(1)+'.pkl')

            initialiser = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            
            with tf.variable_scope('model', reuse=None, initializer=initialiser):
                
                m = seq2seqModel(config, is_training=True,batchnorm=True)

                    
            with tf.variable_scope("model", reuse=True):
                m_val = seq2seqModel(eval_config, is_training=False)
                m_test = seq2seqModel(eval_config, is_training=False)
                
            tf.global_variables_initializer().run()
            
            variables = tf.trainable_variables()
            values = session.run(variables)
            for var, val in zip(variables, values):
                print(var)
            print('-----------')
            
            # If we have a saved/pre-trained model, load it.
            model_trained=None#os.path.join(FLAGS.data_path, 'ACTmodel/wiki256_random_LFACTx_nonlinear_rnn.pkl')
            #model_trained=None#os.path.join(FLAGS.data_path, 'ACTmodel/'+modelname+str(config.num_steps)+'to'+str(config.predict_num)+'feedmodel'+str(ite)+'.pkl')
            if model_trained is not None:
                saveload.main(model_trained, session)
            
            print("starting training")
            max_steps = 100 if debug else None
            
            train_loss=0
            modelpath=os.path.join(FLAGS.data_path, 'ACTmodel/'+modelname+'.pkl')
            saved_model_path = modelpath
            #model_100_path = os.path.join(FLAGS.data_path, 'ACTmodel/'+modelname+'_100.pkl')
            model_last_path = os.path.join(FLAGS.data_path, 'ACTmodel/'+modelname+'_last.pkl')
            ite_save = [5,20,50,100,200]
            for i in range(config.max_max_epoch):
                #print(datetime.datetime.now())
                #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                #session.run(tf.assign(m.lr, config.learning_rate * lr_decay))
                train_loss,_,Nt_train = run_epoch(config,session, m, onehot, train_index, m.train_op, max_steps=max_steps,verbose=True)
                #Nt1.append(Nt_train)
                train_loss_oneite.append(train_loss)
                if verbose:
                    #print("Epoch: {} Learning rate: {}".format(i + 1, float(session.run(m.lr))))
                    print("ite: {} Epoch: {} Train Loss: {}".format(ite, i, train_loss))
                    
                val_loss,bpc_val,Nt_val = run_epoch(config,session, m_val, onehot, val_index, tf.no_op(), fmea_cal=True)
                #Nt2.append(Nt_val)
                val_loss_oneite.append(val_loss)
                print("ite: {} Epoch: {} Val Loss: {}".format(ite, i, val_loss))
                
                
                sample_size = len(bpc_val)
                bpc_val = np.reshape(bpc_val,[sample_size*config.batch_size,config.num_steps])
                bpc_val_list = []
                for pred in range(config.num_steps):
                    res = bpc_val[:,pred]
                    bpc = -math.log(np.sum(res)/(sample_size*config.batch_size),2)
                    bpc_val_list.append(bpc)
                bpc_ave=np.mean(bpc_val_list)
                
                if bpc_ave < bpc_min:
                    bpc_min = bpc_ave
                    print("bpc_min2:",bpc_min)
                    try:
                        os.remove(saved_model_path)
                        saveload.main(saved_model_path, session)
                    except:
                        saveload.main(saved_model_path, session)
                
                Ntmax_train=np.max(Nt_train,0)
                Nt90_train = np.percentile(Nt_train,90,axis=0)
                Ntmean_train = np.mean(Nt_train, axis=0)
                print('train max',list(Ntmax_train))
                print('train 90',list(Nt90_train))
                print('train mean',list(Ntmean_train))
                
                Ntmax_val=np.max(Nt_val,0)
                Nt90_val = np.percentile(Nt_val,90,axis=0)
                Ntmean_val = np.mean(Nt_val, axis=0)
                print('val max',list(Ntmax_val))
                print('val 90',list(Nt90_val))
                print('val mean',list(Ntmean_val))
                
                if i in ite_save:
                    saved_path = os.path.join(FLAGS.data_path, 'ACTmodel/'+modelname+'_'+str(i)+'.pkl')
                    saveload.main(saved_path, session)
                
            saveload.main(model_last_path, session)
            
            
            train_loss_ite.append(train_loss_oneite)
            val_loss_ite.append(val_loss_oneite)
            print('start test best val')          
            
            saveload.main(saved_model_path, session)            
            test_loss,bpc_test,Nt_test = run_epoch(config,session, m_test, onehot, test_index, tf.no_op(), fmea_cal=True)
            sample_size = len(bpc_test)
            bpc_test = np.reshape(bpc_test,[sample_size*config.batch_size,config.num_steps])
            bpc_test_list = []
            for pred in range(config.num_steps):
                res = bpc_test[:,pred]
                bpc = -math.log(np.sum(res)/(sample_size*config.batch_size),2)
                bpc_test_list.append(bpc)
            bpc_ave=np.mean(bpc_test_list)
            print('test1 bpc ave: ', bpc_ave)
            print('test1 bpc: ',bpc_test_list)
            
            Ntmax_test=np.max(Nt_test,0)
            Nt90_test = np.percentile(Nt_test,90,axis=0)
            Ntmean_test = np.mean(Nt_test, axis=0)
            print('test1 max',list(Ntmax_test))
            print('test1 90',list(Nt90_test))
            print('test1 mean',list(Ntmean_test))
            
            print('start test last') 
            saveload.main(model_no_val, session)            
            test_loss,bpc_test,Nt_test = run_epoch(config,session, m_test, onehot, test_index, tf.no_op(), fmea_cal=True)            
            sample_size = len(bpc_test)
            bpc_test = np.reshape(bpc_test,[sample_size*config.batch_size,config.num_steps])
            bpc_test_list = []
            for pred in range(config.num_steps):
                res = bpc_test[:,pred]
                bpc = -math.log(np.sum(res)/(sample_size*config.batch_size),2)
                bpc_test_list.append(bpc)
            bpc_ave=np.mean(bpc_test_list)            
            print('test2 bpc ave: ', bpc_ave)
            print('test2 bpc: ',bpc_test_list)
            
            Ntmax_test=np.max(Nt_test,0)
            Nt90_test = np.percentile(Nt_test,90,axis=0)
            Ntmean_test = np.mean(Nt_test, axis=0)
            print('test2 max',list(Ntmax_test))
            print('test2 90',list(Nt90_test))
            print('test2 mean',list(Ntmean_test))
            
            
            if verbose:
                print("Ite: {} Test Loss: {}".format(ite, test_loss))
                #print("Test Perplexity: %.3f" % test_loss)
                
            print('train loss:')
            print(train_loss_oneite)
            print('----------')
            print('val loss:')
            print(val_loss_oneite)
            print('----------')
            
            
            
            
    print("Ave Train Loss: {} Ave Test Loss: {}".format(np.mean(train_loss_ite),np.mean(test_loss_ite)))
    print('fmeasure:',fmeasure)
    print("ave F-1:", np.mean(fmeasure, axis=-1))
    print("config.ponder_time_penalty:",config.ponder_time_penalty)
    print("config.min_act_step:",config.min_act_step)
    #print("f1_val[0]")

    

if __name__ == '__main__':
    flags = tf.flags
    logging = tf.logging
    flags.DEFINE_string("model_size", "small", "Size of model to train, either small, medium or large")
    #flags.DEFINE_string("data_path", './', "data_path")
    flags.DEFINE_string("data_path", '/home/lzhang/ACTwiki/', "data_path")

    flags.DEFINE_string("model_path", None, "full path of a saved model to load")
    flags.DEFINE_string("weights_dir", None, "full directory path to save weights into per epoch")
    flags.DEFINE_boolean("verbose", True, "Verbosity of the training")
    flags.DEFINE_boolean("debug", True, "Uses small corpuses for debugging purposes")
    FLAGS = flags.FLAGS

    from tensorflow.python.platform import flags
    from sys import argv

    flags.FLAGS._parse_flags()
    main(argv)
