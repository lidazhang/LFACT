import tensorflow as tf
import numpy as np
from keras.layers.convolutional import Conv2DTranspose

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops


#set weight
def weight_variable(shape,name=None,w_init=0.1):
    if name is None:
        #return tf.Variable(tf.truncated_normal(shape)*w_init)
        return tf.get_variable(shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    else:
        #return tf.Variable(tf.truncated_normal(shape)*w_init,name=name)
        return tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape,name=None,if_random=False):
    if name is None:
        return tf.Variable(tf.constant(0.0,shape=shape)) if not if_random else tf.Variable(tf.random_normal(shape))
    else:
        return tf.Variable(tf.constant(0.0,shape=shape),name=name) if not if_random else tf.Variable(tf.random_normal(shape),name=name)
def weigth_variable_list(shape_list,w_init=0.1):
    #return [tf.Variable(tf.truncated_normal(each['shape'])*w_init,name=each['name']) for each in shape_list]
    return [ tf.get_variable(name=each['name'],shape=each['shape'],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for each in shape_list]
def bias_variable_list(shape_list,if_random=False):
    if if_random:
        return [tf.Variable(tf.random_normal(each['shape']),name=each['name']) for each in shape_list]
    else:
        return [tf.Variable(tf.constant(0.0,shape=each['shape']),name=each['name']) for each in shape_list]

def queryVariablesList(nameQuery=None):
    t_vars = tf.trainable_variables()
    if nameQuery is None:
        return t_vars
    elif type(nameQuery) is type([1,2]):
        for each in nameQuery:
            t_vars = [var for var in t_vars if each in var.name]
        return t_vars
    else:
        return [var for var in t_vars if nameQuery in var.name]


def diffCoordinates(x1,x2,nframe,nP = 11,nBall=2,scale_x=2,scale_y=1):
    s = tf.reshape(x1,shape=[-1,nframe,nP,nBall]) - tf.reshape(x2,shape=[-1,nframe,nP,nBall])
    sx = s[:,:,:,0] * scale_x
    sy = s[:,:,:,1] * scale_y
    return 0.5*(tf.reduce_mean(tf.square(sx)) + tf.reduce_mean(tf.square(sy)))

def diffSpeed(x1,x2,order=2):
    s1 = x1[:,1:] - x1[:,:-1]
    s2 = x2[:,1:] - x2[:,:-1]
    if order == 2:
        return tf.reduce_mean(tf.square(s1-s2))
    elif order == 1:
        return tf.reduce_mean(tf.abs(s1-s2))


def scaleXY(x,nframe,scale_x=2,scale_y=1,nP = 11, nBall = 2,outputDim = 4):
    s = tf.reshape(x,shape=[-1,nframe,nP,nBall])
    sx = s[:,:,:,0] * scale_x
    sy = s[:,:,:,1] * scale_y
    if outputDim == 4:
        return tf.stack([sx,sy],axis=3)
    elif outputDim == 3:
        return tf.reshape(tf.stack([sx,sy],axis=3),shape=[-1,nframe,nP*nBall])


# sort rows of tensors

def sortTensor(xxx,vvv,nDim):
    # sort xxx given vvv, xxx being a 2-d tensor, vvv being a vector
    # nDim: either a real int, or an int placeholder
    _,idxVVV = tf.nn.top_k(vvv,k = nDim)
    xxx_sorted = tf.gather(xxx,idxVVV)
    return xxx_sorted

def sortTensor3D(xxx_3d,vvv_2d,nDim,nRow):

    _,idxVVV = tf.nn.top_k(vvv_2d,k = nDim)

    iii = tf.cast( tf.range(nRow),idxVVV.dtype)
    iii = tf.stack( [iii]*nDim,axis = 1 )
    iii = tf.stack([iii,idxVVV],axis = 2)

    xxx_sorted = tf.gather_nd(xxx_3d,iii)

    return xxx_sorted

# discount negative elements in tensors
def discountNegativeElements(tf_input,p_threshold,mode = 'RANDOM'):
    assert mode in ['RANDOM','FIXED']
    flag_positive = tf.cast(tf_input>0.,tf.float32)
    weights = tf.minimum(flag_positive+p_threshold,1.0)

    if mode is 'RANDOM':
        flag_random = tf.random_uniform(shape=tf_input.shape)
        weights_final = tf.cast( flag_random < weights ,tf.float32)
        return weights_final * tf_input

    elif mode is 'FIXED':
        return weights * tf_input

# RBF integer c
def sampleC_RBF(nC,nDim,M):
    c_list = []
    while len(c_list)<nC:
        this_c = np.random.choice(range(M),size=[nDim])
        flag = True
        for each in c_list:
            if abs(this_c - each).sum()==0:
                flag = False
                break
        if flag:
            c_list.append(this_c)
    return np.stack(c_list)

#clamp
def weight_clip(weight_list,clip):
    clip_critic_vars_op = [var.assign(tf.clip_by_value(var, -clip, clip)) for var in weight_list]
    return clip_critic_vars_op


def setOptimizer(lr,loss,opt='adam',momentum=0.9,use_nesterov=True,var_list=None,gradient_clip=None,beta1_adam=None,beta2_adam=None,name=None):
    assert opt in ['adam','adadelta','rmsprop','momentum','sgd']
    if gradient_clip is None:
        if opt == 'adam':
            beta1 = beta1_adam if beta1_adam is not None else 0.9
            beta2 = beta2_adam if beta2_adam is not None else 0.999
            return tf.train.AdamOptimizer(lr,beta1=beta1,beta2=beta2).minimize(loss,var_list=var_list) if name is None else tf.train.AdamOptimizer(lr,beta1=beta1,beta2=beta2,name=name).minimize(loss,var_list=var_list)
        if opt == 'adadelta':
            return tf.train.AdadeltaOptimizer(lr).minimize(loss,var_list=var_list) if name is None else tf.train.AdadeltaOptimizer(lr,name=name).minimize(loss,var_list=var_list)
        if opt == 'rmsprop':
            return tf.train.RMSPropOptimizer(lr).minimize(loss,var_list=var_list) if name is None else tf.train.RMSPropOptimizer(lr,name=name).minimize(loss,var_list=var_list)
        if opt == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum,use_nesterov=use_nesterov).minimize(loss,var_list=var_list) if name is None else tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum,use_nesterov=use_nesterov,name=name).minimize(loss,var_list=var_list)
        if opt == 'sgd':
            return tf.train.GradientDescentOptimizer(lr).minimize(loss,var_list=var_list) if name is None else tf.train.GradientDescentOptimizer(lr,name=name).minimize(loss,var_list=var_list)
    else:
        assert gradient_clip>0
        if opt=='adam':
            beta1 = beta1_adam if beta1_adam is not None else 0.9
            beta2 = beta2_adam if beta2_adam is not None else 0.999
            train_step = tf.train.AdamOptimizer(lr,beta1=beta1,beta2=beta2)
            gradients, variables = zip(*train_step.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
            train_step = train_step.apply_gradients(zip(gradients, variables))
            return train_step



# sparse_tensor_to_dense

def sparseTo4DTensor(xxx,batchsize,nPlayer,M,int_type = tf.int64):
    xxx += 0.1 # add a small offset
    yyy = tf.cast(xxx,int_type)

    # appendix
    iii_append = tf.reshape( tf.cast( tf.range(nPlayer),yyy.dtype ),shape=[-1,1])
    iii_append = tf.stack([iii_append]*batchsize,axis = 0)
    yyy = tf.concat([iii_append,yyy],axis = 2)

    # prefix
    iii_prefix = tf.cast(tf.range(batchsize),yyy.dtype)
    iii_prefix = tf.stack([iii_prefix]*nPlayer,axis = 0)
    iii_prefix = tf.transpose(iii_prefix,[1,0])
    iii_prefix = tf.reshape(iii_prefix,shape=[batchsize,nPlayer,1])
    yyy = tf.concat([iii_prefix,yyy],axis = 2) #batchsize, nPlayer

    # reshape indices
    yyy = tf.reshape(yyy,shape=[-1,4])

    # sparse tensor to dense tensor
    zzz = tf.SparseTensor(indices=yyy,values=[1.0]*batchsize*nPlayer,dense_shape=(batchsize,nPlayer,M,M))
    qqq = tf.sparse_tensor_to_dense(zzz)

    qqq = tf.transpose(qqq,[0,2,3,1])

    return qqq #batchsize, M, M, channel(nPlayer)


#===================
#Layer Normalization
#===================
def Layernorm(name, norm_axes, inputs):
    mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]

    #offset = lib.param(name+'.offset', np.zeros(n_neurons, dtype='float32'))
    #scale = lib.param(name+'.scale', np.ones(n_neurons, dtype='float32'))
    offset = bias_variable([n_neurons],name=name+'.offset')
    scale = tf.Variable(np.ones(n_neurons,dtype='float32'),name=name+'.scale')

    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    offset = tf.reshape(offset, [-1] + [1 for i in range(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in range(len(norm_axes)-1)])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result

###############################
#ANOTHER VERSION OF LAYER NORM
###############################
def layer_norm(input_tensor, num_variables_in_tensor = 1, initial_bias_value = 0.0, scope = "layer_norm"):
  with tf.variable_scope(scope):
    '''for clarification of shapes:
    input_tensor = [batch_size, num_neurons]
    mean = [batch_size]
    variance = [batch_size]
    alpha = [num_neurons]
    bias = [num_neurons]
    output = [batch_size, num_neurons]
    '''
    input_tensor_shape_list = input_tensor.get_shape().as_list()

    num_neurons = input_tensor_shape_list[1]/num_variables_in_tensor



    alpha = tf.get_variable('layer_norm_alpha', [num_neurons * num_variables_in_tensor],
            initializer = tf.constant_initializer(1.0))

    bias = tf.get_variable('layer_norm_bias', [num_neurons * num_variables_in_tensor],
            initializer = tf.constant_initializer(initial_bias_value))

    if num_variables_in_tensor == 1:
      input_tensor_list = [input_tensor]
      alpha_list = [alpha]
      bias_list = [bias]

    else:
      input_tensor_list = tf.split(1, num_variables_in_tensor, input_tensor)
      alpha_list = tf.split(0, num_variables_in_tensor, alpha)
      bias_list = tf.split(0, num_variables_in_tensor, bias)

    list_of_layer_normed_results = []
    for counter in range(num_variables_in_tensor):
      mean, variance = moments_for_layer_norm(input_tensor_list[counter], axes = [1], name = "moments_loopnum_"+str(counter)+scope) #average across layer

      output =  (alpha_list[counter] * (input_tensor_list[counter] - mean)) / variance + bias[counter]

      list_of_layer_normed_results.append(output)

    if num_variables_in_tensor == 1:
      return list_of_layer_normed_results[0]
    else:
      return tf.concat(1, list_of_layer_normed_results)


def moments_for_layer_norm(x, axes = 1, name = None, epsilon = 0.001):
  '''output for mean and variance should be [batch_size]'''

  if not isinstance(axes, list): axes = list(axes)

  with tf.op_scope([x, axes], name, "moments"):
    mean = tf.reduce_mean(x, axes, keep_dims = True)

    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims = True) + epsilon)

    return mean, variance



########################
# Dense / Complex Layers
########################

def DenseLayerStack(x_input,Wlist,blist,Activation='relu',if_batchnorm=True,if_dropout=False,keep_prob=None,if_layernorm=False,name_layer=None,reuse=None):
    assert Activation in ['relu','linear','sigmoid','tanh']
    assert if_batchnorm==False or if_layernorm==False

    if if_layernorm:
        assert name_layer is not None
        layer_count = 0

    x_output = tf.identity(x_input)

    for W,b in zip(Wlist,blist):

        x_output = tf.matmul(x_output,W)+b

        if Activation == 'relu':
            x_output = tf.nn.relu(x_output)
        elif Activation == 'sigmoid':
            x_output = tf.nn.sigmoid(x_output)
        elif Activation == 'tanh':
            x_output = tf.nn.tanh(x_output)

        if if_batchnorm:

            if reuse is not None:
                with tf.variable_scope(name_layer) as scope:
                    if reuse:
                        scope.reuse_variables()
                    x_output = tf.layers.batch_normalization(x_output)
            else:
                x_output = tf.layers.batch_normalization(x_output)
        if if_layernorm:
            #x_output = Layernorm(name_layer+'_'+str(layer_count),[1],x_output)
            x_output = layer_norm(x_output, num_variables_in_tensor = 1, initial_bias_value = 0.0, scope = name_layer+'_layer_norm_'+str(layer_count))
            layer_count += 1

        if if_dropout:
            x_output = tf.nn.dropout(x_output,keep_prob)


    return x_output


class DenseStackNormal:

    def __init__(self,session,nameLayer,reuse=False,*args,**kwargs):
        self.session = session
        self.nameLayer = nameLayer
        self.reuse = reuse

        with tf.variable_scope(nameLayer,reuse=reuse):
            self.__build_model__(*args,**kwargs)


    def __build_model__(self,dim_list,keep_prob = None,dropout_mask_list=None):

        assert type(dim_list) is type([1,2]) and len(dim_list)>1

        self.W_list = []
        self.g_list = []
        self.b_list = []
        layer_count = 0
        for dim_in,dim_out in zip(dim_list[:-1],dim_list[1:]):
            self.W_list.append( weight_variable(name='Normal_W_'+str(layer_count),shape=[dim_in,dim_out]) )
            self.b_list.append( bias_variable(name='Normal_b_'+str(layer_count),shape=[dim_out]) )
            layer_count+=1

        self.keep_prob = keep_prob
        self.dropout_mask_list = dropout_mask_list

        if dropout_mask_list is not None:
            assert type(dropout_mask_list) is type([1,2])
            assert len(dropout_mask_list) == len(self.W_list)

    def call(self,x_input,Activation = 'relu',if_dropout=False,use_batchnorm=False,name_layer=None,reuse=None,alpha_leaky_relu = 0.01,alpha_elu = 1.0):


        assert Activation in ['relu','linear','sigmoid','tanh','leakyrelu','elu']


        #call model
        x_output = tf.identity(x_input)
        if self.dropout_mask_list is not None and if_dropout:

            lcount = 0

            for W,b,mask in zip(self.W_list,self.b_list,self.dropout_mask_list):

                #x_output_s = tf.shape(x_output)

                W_ = W*mask

                x_output = (1/self.keep_prob)*tf.matmul(x_output,W_) + b

                #x_output = tf.reshape(x_output,shape=(x_output_s[0],tf.shape(W)[1]))


                if Activation == 'relu':
                    x_output = tf.nn.relu(x_output)
                elif Activation == 'sigmoid':
                    x_output = tf.nn.sigmoid(x_output)
                elif Activation == 'tanh':
                    x_output = tf.nn.tanh(x_output)
                elif Activation == 'leakyrelu':
                    x_output = tf.maximum(x_output, alpha_leaky_relu*x_output)
                elif Activation == 'elu':
                    x_output = tf.maximum(x_output,tf.minimum(0.0, alpha_elu * tf.exp(x_output) - 1 ) )


                if use_batchnorm:
                    #with tf.variable_scope(self.nameLayer,reuse=self.reuse):
                    if reuse is not None:
                        with tf.variable_scope(name_layer+str(lcount)) as scope:
                            if reuse:
                                scope.reuse_variables()
                            x_output = tf.layers.batch_normalization(x_output)
                    else:
                        x_output = tf.layers.batch_normalization(x_output)

                lcount+=1

        else:

            lcount = 0

            for W,b in zip(self.W_list,self.b_list):

                x_output = tf.matmul(x_output,W) + b

                if if_dropout:
                    x_output = tf.nn.dropout(x_output,self.keep_prob)


                if Activation == 'relu':
                    x_output = tf.nn.relu(x_output)
                elif Activation == 'sigmoid':
                    x_output = tf.nn.sigmoid(x_output)
                elif Activation == 'tanh':
                    x_output = tf.nn.tanh(x_output)

                if use_batchnorm:
                    if reuse is not None:
                        with tf.variable_scope(name_layer+str(lcount)) as scope:
                            if reuse:
                                scope.reuse_variables()
                            x_output = tf.layers.batch_normalization(x_output)
                    else:
                        x_output = tf.layers.batch_normalization(x_output)

                lcount+=1


        return x_output




def OneDenseLayerComplex(r_in,i_in,r_w,i_w,r_b,i_b,Activation='relu',if_batchnorm=True,if_dropout=False,keep_prob=None):
    assert Activation in ['relu','linear','sigmoid','tanh']

    RR = tf.matmul(r_in,r_w) + r_b
    RI = tf.matmul(r_in,i_w) + i_b
    IR = tf.matmul(i_in,r_w) + r_b
    II = tf.matmul(i_in,i_w) + i_b

    Rpart = RR - II
    Ipart = IR + RI

    if Activation == 'relu':
        Rpart = tf.nn.relu(Rpart)
        Ipart = tf.nn.relu(Ipart)
    elif Activation == 'sigmoid':
        Rpart = tf.nn.sigmoid(Rpart)
        Ipart = tf.nn.sigmoid(Ipart)
    elif Activation == 'tanh':
        Rpart = tf.nn.tanh(Rpart)
        Ipart = tf.nn.tanh(Ipart)

    if if_batchnorm:
        Rpart = tf.layers.batch_normalization(Rpart)
        Ipart = tf.layers.batch_normalization(Ipart)

    if if_dropout:
        Rpart = tf.nn.dropout(Rpart,keep_prob)
        Ipart = tf.nn.dropout(Ipart,keep_prob)

    return Rpart,Ipart

class DenseStackWithWeightNormalization:

    def __init__(self,session,nameLayer,reuse=False,*args,**kwargs):
        self.session = session
        self.nameLayer = nameLayer

        with tf.variable_scope(nameLayer,reuse=reuse):
            self.__build_model__(*args,**kwargs)


    def __build_model__(self,dim_list,keep_prob = None,dropout_mask_list=None):

        assert type(dim_list) is type([1,2]) and len(dim_list)>1

        self.W_list = []
        self.g_list = []
        self.b_list = []
        layer_count = 0
        for dim_in,dim_out in zip(dim_list[:-1],dim_list[1:]):
            self.W_list.append( weight_variable(name='WN_W_'+str(layer_count),shape=[dim_in,dim_out]) )
            self.g_list.append( bias_variable(name='WN_g_'+str(layer_count),shape=[dim_out],if_random=True ) )
            self.b_list.append( bias_variable(name='WN_b_'+str(layer_count),shape=[dim_out]) )
            layer_count+=1

        self.keep_prob = keep_prob
        self.dropout_mask_list = dropout_mask_list

        if dropout_mask_list is not None:
            assert type(dropout_mask_list) is type([1,2])
            assert len(dropout_mask_list) == len(self.W_list)

    def call(self,x_input,Activation = 'relu',if_dropout=False,mode='use'):

        assert mode in ['use','initialize']
        if mode is 'initialize':
            pre_activation_list = []

        assert Activation in ['relu','linear','sigmoid','tanh']


        #call model
        x_output = tf.identity(x_input)
        if self.dropout_mask_list is not None:

            for W,g,b,mask in zip(self.W_list,self.g_list,self.b_list,self.dropout_mask_list):

                if mode is 'initialize':
                    pre_activation_list.append( tf.matmul(x_output,W*mask)/tf.norm(W,axis=0) )

                x_output = (1/self.keep_prob)*g*tf.matmul(x_output,W*mask)/tf.norm(W,axis=0) + b



                if Activation == 'relu':
                    x_output = tf.nn.relu(x_output)
                elif Activation == 'sigmoid':
                    x_output = tf.nn.sigmoid(x_output)
                elif Activation == 'tanh':
                    x_output = tf.nn.tanh(x_output)

        else:

            for W,g,b in zip(self.W_list,self.g_list,self.b_list):

                if mode is 'initialize':
                    pre_activation_list.append( tf.matmul(x_output,W)/tf.norm(W,axis=0) )

                x_output = g*tf.matmul(x_output,W)/tf.norm(W,axis=0) + b

                if if_dropout:
                    x_output = tf.nn.dropout(x_output,self.keep_prob)


                if Activation == 'relu':
                    x_output = tf.nn.relu(x_output)
                elif Activation == 'sigmoid':
                    x_output = tf.nn.sigmoid(x_output)
                elif Activation == 'tanh':
                    x_output = tf.nn.tanh(x_output)

        if mode is 'initialize':
            return pre_activation_list
        else:
            return x_output

def OneConvLayerComplex(r_t_real,r_t_imaginary,conv_real_weights,conv_imaginary_weights,
                        conv_real_bias,conv_imaginary_bias,stridePool1,
                        Activation='relu',if_batchnorm=True,):

    assert Activation in ['relu','linear','sigmoid','tanh']

    RR = tf.nn.conv2d(r_t_real,conv_real_weights,[1,1,1,1],padding='VALID') + conv_real_bias
    RI = tf.nn.conv2d(r_t_real,conv_imaginary_weights,[1,1,1,1],padding='VALID') + conv_imaginary_bias
    IR = tf.nn.conv2d(r_t_imaginary,conv_real_weights,[1,1,1,1],padding='VALID') + conv_real_bias
    II = tf.nn.conv2d(r_t_imaginary,conv_imaginary_weights,[1,1,1,1],padding='VALID') + conv_imaginary_bias

    RR = tf.nn.max_pool(RR,[1,stridePool1,1,1],[1,stridePool1,1,1],padding='SAME')
    RI = tf.nn.max_pool(RI,[1,stridePool1,1,1],[1,stridePool1,1,1],padding='SAME')
    IR = tf.nn.max_pool(IR,[1,stridePool1,1,1],[1,stridePool1,1,1],padding='SAME')
    II = tf.nn.max_pool(II,[1,stridePool1,1,1],[1,stridePool1,1,1],padding='SAME')

    Rpart = RR-II
    Ipart = RI+IR

    if Activation == 'relu':
        Rpart = tf.nn.relu(Rpart)
        Ipart = tf.nn.relu(Ipart)
    elif Activation == 'sigmoid':
        Rpart = tf.nn.sigmoid(Rpart)
        Ipart = tf.nn.sigmoid(Ipart)
    elif Activation == 'tanh':
        Rpart = tf.nn.tanh(Rpart)
        Ipart = tf.nn.tanh(Ipart)

    if if_batchnorm:
        Rpart = tf.layers.batch_normalization(Rpart)
        Ipart = tf.layers.batch_normalization(Ipart)

    return Rpart,Ipart

def OneDeconvLayer(deconv_in,channel_out,deconv_kernel,deconv_stride = (1,1)):
    deconvLayer = Conv2DTranspose(channel_out,deconv_kernel,strides=deconv_stride,padding='same')
    deconv_out = deconvLayer(deconv_in)
    return deconv_out

##################
#LSTM UTILITIES
##################

def initializerLSTM(nhidden,v_each_frame,w_init=None,name=None,if_output=False,noutput=None):
    #h0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_h0')
    #c0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_c0')



    if name is not None:
        kernelLSTM = weight_variable([v_each_frame,4*nhidden],w_init=w_init,name=name+'_kernel')
        recurrent_kernelLSTM = weight_variable([nhidden,4*nhidden],w_init=w_init,name=name+'_recurrent_kernel')
        biasLSTM = bias_variable([4*nhidden],name=name+'_bias')
        h0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_h0')
        c0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_c0')
        #h0LSTM = tf.identity(h0LSTM,name=name+'_h0')
        #c0LSTM = tf.identity(c0LSTM,name=name+'_c0')
        #kernelLSTM = tf.identity(kernelLSTM,name=name+'_kernel')
        #recurrent_kernelLSTM = tf.identity(recurrent_kernelLSTM,name=name+'_recurrent_kernel')
        #biasLSTM = tf.identity(biasLSTM,name=name+'_bias')

    paramsLSTM = [h0LSTM,c0LSTM,kernelLSTM,recurrent_kernelLSTM,biasLSTM]

    if if_output:
        #Woutput = weight_variable([v_each_frame,noutput],w_init=w_init)
        #boutput = bias_variable([noutput])
        if name is not None:
            Woutput = weight_variable([nhidden,noutput],w_init=w_init,name=name+'_Woutput')
            boutput = bias_variable([noutput],name=name+'_boutput')
            #Woutput = tf.identity(Woutput,name=name+'_Woutput')
            #boutput = tf.identity(boutput,name=name+'_Woutput')

        paramsLSTM = paramsLSTM + [Woutput,boutput]

    return paramsLSTM

def lstmcell(x_t,h_tm1,c_tm1,kernel,recurrent_kernel,bias,Woutput=None,boutput=None,if_output=False,keep_prob=None,keep_prob_recurrent=None):
    kernel_out = tf.matmul(x_t,kernel)
    
    recurrent_kernel_out = tf.matmul(h_tm1,recurrent_kernel) + bias

    if keep_prob_recurrent is not None:
        kernel_out = tf.nn.dropout(kernel_out,keep_prob_recurrent)
        recurrent_kernel_out = tf.nn.dropout(recurrent_kernel_out,keep_prob_recurrent)

    x0,x1,x2,x3 = tf.split(kernel_out,num_or_size_splits=4,axis=1)
    r0,r1,r2,r3 = tf.split(recurrent_kernel_out,num_or_size_splits=4,axis=1)

    f = tf.sigmoid(tf.add(x0,r0)) #use sigmoid activation
    i = tf.sigmoid(tf.add(x1,r1))
    c_prime = tf.tanh(tf.add(x2,r2))
    c_t = f*c_tm1 + i * c_prime
    o = tf.sigmoid(tf.add(x3,r3))
    h_t = o*tf.tanh(c_t)

    if if_output:
        output_t = tf.matmul(h_t,Woutput) + boutput
        if keep_prob is not None:
            output_t = tf.nn.dropout(output_t,keep_prob)
        return output_t,h_t,c_t
    else:
        return h_t,c_t


def lstmlayer(x,h0,c0,kernel,recurrent_kernel,bias,nframe,batchsize,if_output=False,Woutput=None,boutput=None):
    #x: batchsize, nframe, each_frame
    #
    assert kernel.get_shape()[1] == recurrent_kernel.get_shape()[1],'weight dim do not match!'
    assert kernel.get_shape()[1]%4==0,'weight dim should be a multiple of 4 (f, i, c_prime, o) '


    if len(h0.get_shape())==1:
        s_h_tm1 = int(h0.get_shape()[0])
        h_tm1 = tf.tile(h0,[tf.shape(x)[0]])
        h_tm1 = tf.reshape(h_tm1,shape=[-1,s_h_tm1],name='h_reshape_lstm')
        #h_tm1 = tf.stack([h0]*batchsize,axis=0)
    else:
        h_tm1 = tf.identity(h0)

    if len(c0.get_shape())==1:
        s_c_tm1 = int(c0.get_shape()[0])
        c_tm1 = tf.tile(c0,[tf.shape(x)[0]])
        c_tm1 = tf.reshape(c_tm1,shape=[-1,s_c_tm1],name='c_reshape_lstm')
        #c_tm1 = tf.stack([c0]*batchsize,axis=0)
    else:
        c_tm1 = tf.identity(c0)
    h_list = []
    c_list = []
    if if_output:
        output_list = []

    for t in range(nframe):
        x_t = x[:,t]
        kernel_out = tf.matmul(x_t,kernel)
        recurrent_kernel_out = tf.matmul(h_tm1,recurrent_kernel) + bias
        x0,x1,x2,x3 = tf.split(kernel_out,num_or_size_splits=4,axis=1)
        r0,r1,r2,r3 = tf.split(recurrent_kernel_out,num_or_size_splits=4,axis=1)
        f = tf.sigmoid(tf.add(x0,r0)) #use sigmoid activation
        i = tf.sigmoid(tf.add(x1,r1))
        c_prime = tf.tanh(tf.add(x2,r2))
        c_t = f*c_tm1 + i * c_prime
        o = tf.sigmoid(tf.add(x3,r3))
        h_t = o*tf.tanh(c_t)

        if if_output:
            output_t = tf.matmul(h_t,Woutput) + boutput
            output_list.append(output_t)

        h_list.append(h_t)
        h_tm1 = h_t
        c_list.append(c_t)
        c_tm1 = c_t

    h_list = tf.stack(h_list,axis=1)
    c_list = tf.stack(c_list,axis=1)
    if if_output:
        output_list = tf.stack(output_list,axis=1)
        return output_list,h_list,c_list
    else:
        return h_list,c_list



def initializerAttnLSTM(nhidden,v_each_frame,w_init=None,name=None,if_output=False,noutput=None):
    #h0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_h0')
    #c0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_c0')



    if name is not None:
        kernelLSTM = weight_variable([v_each_frame,4*nhidden],w_init=w_init,name=name+'_kernel')
        recurrent_kernelLSTM = weight_variable([nhidden,4*nhidden],w_init=w_init,name=name+'_recurrent_kernel')
        attn_kernelLSTM = weight_variable([nhidden,4*nhidden],w_init=w_init,name=name+'attn_kernelLSTM')
        biasLSTM = bias_variable([4*nhidden],name=name+'_bias')
        h0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_h0')
        c0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_c0')
        #h0LSTM = tf.identity(h0LSTM,name=name+'_h0')
        #c0LSTM = tf.identity(c0LSTM,name=name+'_c0')
        #kernelLSTM = tf.identity(kernelLSTM,name=name+'_kernel')
        #recurrent_kernelLSTM = tf.identity(recurrent_kernelLSTM,name=name+'_recurrent_kernel')
        #biasLSTM = tf.identity(biasLSTM,name=name+'_bias')

    paramsLSTM = [h0LSTM,c0LSTM,kernelLSTM,recurrent_kernelLSTM,attn_kernelLSTM,biasLSTM]

    if if_output:
        #Woutput = weight_variable([v_each_frame,noutput],w_init=w_init)
        #boutput = bias_variable([noutput])
        if name is not None:
            Woutput = weight_variable([nhidden,noutput],w_init=w_init,name=name+'_Woutput')
            boutput = bias_variable([noutput],name=name+'_boutput')
            #Woutput = tf.identity(Woutput,name=name+'_Woutput')
            #boutput = tf.identity(boutput,name=name+'_Woutput')

        paramsLSTM = paramsLSTM + [Woutput,boutput]

    return paramsLSTM

def attnlstmcell(x_t,c_attn,h_tm1,c_tm1,kernel,recurrent_kernel,attn_kernel,bias,Woutput=None,boutput=None,if_output=False,keep_prob=None,keep_prob_recurrent=None):
    if c_attn == 0:
        c_attn = tf.zeros_like(h_tm1,dtype=tf.float32)
    kernel_out = tf.matmul(x_t,kernel)
    recurrent_kernel_out = tf.matmul(h_tm1,recurrent_kernel) + bias
    attn_kernel_out = tf.matmul(c_attn,attn_kernel)

    if keep_prob_recurrent is not None:
        kernel_out = tf.nn.dropout(kernel_out,keep_prob_recurrent)
        recurrent_kernel_out = tf.nn.dropout(recurrent_kernel_out,keep_prob_recurrent)
        attn_kernel_out = tf.nn.dropout(attn_kernel_out,keep_prob_recurrent)

    x0,x1,x2,x3 = tf.split(kernel_out,num_or_size_splits=4,axis=1)
    r0,r1,r2,r3 = tf.split(recurrent_kernel_out,num_or_size_splits=4,axis=1)
    c0,c1,c2,c3 = tf.split(attn_kernel_out,num_or_size_splits=4,axis=1)

    f = tf.sigmoid(tf.add(tf.add(x0,r0),c0)) #use sigmoid activation
    i = tf.sigmoid(tf.add(tf.add(x1,r1),c1)) 
    c_prime = tf.tanh(tf.add(tf.add(x2,r2),c2)) 
    c_t = f*c_tm1 + i * c_prime
    o = tf.sigmoid(tf.add(tf.add(x3,r3),c3)) 
    h_t = o*tf.tanh(c_t)

    if if_output:
        output_t = tf.matmul(h_t,Woutput) + boutput
        if keep_prob is not None:
            output_t = tf.nn.dropout(output_t,keep_prob)
        return output_t,h_t,c_t
    else:
        return h_t,c_t


def initializerGRU(nhidden,v_each_frame,w_init=None,name=None,if_output=False,noutput=None):
    #h0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_h0')
    #c0LSTM = tf.Variable(tf.zeros([nhidden]),name=name+'_c0')

    if name is not None:
        gate_kernel = weight_variable([v_each_frame,2*nhidden],w_init=w_init,name=name+'_gate_kernel')
        gate_rekernel = weight_variable([nhidden,2*nhidden],w_init=w_init,name=name+'_gate_rekernel')
        gate_bias = bias_variable([2*nhidden],name=name+'_gate_bias')
        candidate_kernel = weight_variable([v_each_frame,nhidden],w_init=w_init,name=name+'_candidate_kernel')
        candidate_rekernel = weight_variable([nhidden,nhidden],w_init=w_init,name=name+'_candidate_rekernel')
        candidate_bias = bias_variable([nhidden],name=name+'_candidate_bias')
        s0GRU = tf.Variable(tf.zeros([nhidden]),name=name+'_s0')


    paramsGRU = [s0GRU,gate_kernel,gate_rekernel,gate_bias,candidate_kernel,candidate_rekernel,candidate_bias]

    if if_output:
        #Woutput = weight_variable([v_each_frame,noutput],w_init=w_init)
        #boutput = bias_variable([noutput])
        if name is not None:
            Woutput = weight_variable([nhidden,noutput],w_init=w_init,name=name+'_Woutput')
            boutput = bias_variable([noutput],name=name+'_boutput')
            #Woutput = tf.identity(Woutput,name=name+'_Woutput')
            #boutput = tf.identity(boutput,name=name+'_Woutput')

        paramsGRU = paramsGRU + [Woutput,boutput]

    return paramsGRU

def grucell(x_t,
            s_tm1,
            gate_kernel,
            gate_rekernel,
            gate_bias,
            candidate_kernel,
            candidate_rekernel,
            candidate_bias,
            Woutput=None,
            boutput=None,
            if_output=False,
            keep_prob=None):
    
    gate_inputs = tf.add(math_ops.matmul(x_t, gate_kernel), math_ops.matmul(s_tm1, gate_rekernel))
    gate_inputs = nn_ops.bias_add(gate_inputs, gate_bias)
    
    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    
    r_state = r * s_tm1
    
    candidate = tf.add(math_ops.matmul(x_t, candidate_kernel), math_ops.matmul(r_state, candidate_rekernel))
    candidate = nn_ops.bias_add(candidate, candidate_bias)
    
    c = tf.tanh(candidate)
    new_h = u * s_tm1 + (1 - u) * c

    if if_output:
        output_t = tf.matmul(new_h,Woutput) + boutput
        if keep_prob is not None:
            output_t = tf.nn.dropout(output_t,keep_prob)
        return output_t,new_h
    else:
        return 0,new_h


###################
#LABEL-BALL PARSING
###################
def keepTopValue(beta_copy,topk,nDim,idxDim,negInf=-9999,if_return_index=False):
    thisBeta = tf.identity(beta_copy)
    thisBeta_topk = tf.nn.top_k(thisBeta,topk) #topk was 2 at the beginning.
    thisBeta_topk = tf.reduce_min(thisBeta_topk.values,axis=idxDim)
    beta_topk_index = tf.cast( tf.greater_equal(thisBeta,tf.stack([thisBeta_topk]*nDim,axis=idxDim)) , tf.float32)
    if if_return_index:
        return beta_topk_index
    thisBeta = thisBeta*beta_topk_index + negInf*(1-beta_topk_index)
    return thisBeta

def labelToBall(x_offensive_copy,x_label_copy,nOffensivePlayer=5,nBall=2):
    x_offensive = tf.reshape(tf.identity(x_offensive_copy),shape=[-1,nOffensivePlayer,nBall])
    x_label = tf.stack([tf.identity(x_label_copy)]*nBall,axis=2)
    x_ball = x_offensive*x_label
    x_ball = tf.reduce_sum(x_ball,axis=1)
    return x_ball
