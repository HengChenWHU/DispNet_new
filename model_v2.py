import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tflearn.layers.conv import global_avg_pool

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

from load_data import *

MAX_DISPLACEMENT=50
IMG_W=1024
IMG_H=256
CONV_WEIGHT_DECAY=0.01


################activate function#################
def Relu(x):
    return tf.nn.relu(x)

def dropout(x):
    tf.nn.dropout(x, KEPP_PROB, noise_shape=None, seed=None,name=None) 

def Relu6(x):
    return tf.nn.relu6(x)
    
def Swish(x,beta=1):
  return x*tf.nn.sigmoid(x*beta)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def leaky_relu(x):
    alpha=0.2
    return tf.nn.leaky_relu(x,alpha)


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)
    
###############################SE-NET###########################################

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)
            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation
            return scale
        
####################################BN##################################################
def Batch_Normalization(x, training,scope):
    with arg_scope([batch_norm],
                   updates_collections=None,
                   scope=scope,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,lambda : batch_norm(inputs=x, is_training=training, reuse=None),lambda : batch_norm(inputs=x, is_training=training, reuse=True))



def conv_2d(x, ksize,stride,filters_out):
    filters_in = x.get_shape()[-1]
    #print(filters_in)
    shape = [ksize, ksize, filters_in, filters_out]
    weights = tf.get_variable('weights',
                         shape=shape,
                         dtype='float32',
                         initializer=tf.contrib.layers.xavier_initializer(),
                         
                         trainable=True)
    
    bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0, dtype='float'),regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY))
    #tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)(weights))#
    #tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)(bias))#
    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(x, bias)

def upconv2d_2x2(x, ksize,stride,filters_out,batch_size):
    filters_in = x.get_shape()[-1]
    # must have as_list to get a python list!
    x_shape = x.get_shape().as_list()
    height = x_shape[1] * stride
    width = x_shape[2] * stride
    output_shape = [batch_size, height, width, filters_out]
    strides = [1, stride, stride, 1]
    shape = [ksize, ksize, filters_out, filters_in]
    initializer = tf.contrib.layers.xavier_initializer()
    weights = tf.get_variable('weights',shape=shape,dtype='float32',initializer=initializer,trainable=True)
    bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float32'))
    
    #tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)(weights))#
    #tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)(bias))#
    x = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape, strides=strides, padding='SAME')
    tf.nn.bias_add(x, bias)
    return Relu(x)


def loss(pre, gt):
    with tf.variable_scope('loss'):
        pre=tf.squeeze(pre)
        gt=tf.squeeze(gt)
        pre=tf.to_float(pre)
        loss_=tf.abs(tf.subtract(pre, gt))
        loss_final=tf.reduce_mean(loss_)
        return loss_final
    

def corr(fmA,fmB,MAX_DISPLACEMENT,STRIDE_2,HEIGHT,WIDTH,BATCH_SIZE):
    out = []
    for j in range(-MAX_DISPLACEMENT + 1, MAX_DISPLACEMENT, STRIDE_2): # width
        if j>0:
            padded_a = tf.pad(fmA, [[0,0], [0, 0], [0, abs(j)], [0, 0]])
            #print("shape of padded_a")
            #print(padded_a.shape)
            padded_b = tf.pad(fmB, [[0, 0], [0, 0], [abs(j), 0], [0, 0]])
            #print("shape of padded_b")
            #print(padded_b.shape)
        elif j<0:
            padded_a = tf.pad(fmA, [[0,0], [0, 0], [abs(j),0], [0, 0]])
            #print("shape of padded_a")
            #print(padded_a.shape)
            padded_b = tf.pad(fmB, [[0, 0], [0, 0], [0,abs(j)], [0, 0]])
            #print("shape of padded_b")
            #print(padded_b.shape)
            
        m = padded_a * padded_b
        m=tf.image.resize_images(m,(HEIGHT,WIDTH),2)
        #print("m")
        #print(m.shape)

        #cut = tf.slice(m, [0, 0, 0,0], [BATCH_SIZE,HEIGHT, WIDTH, 3])

        #print("cut")
        #print(cut.shape)
        final = tf.reduce_sum(m, 3)
        #print("final")
        #print(final.shape)
        
            
        out.append(final)
    corr = tf.stack(out, 3)
    print ('Output size: ', corr.shape)
    return corr


def computeSoftArgMin(logits,BATCH_SIZE):
  softmax = tf.nn.softmax(logits,axis=3)
  #sess=tf.Session()
 #sftmax_=sess.run(softmax)
  #f1 = open("dispmap.txt", "w") #将值输出来看一下
# print(softmax_,file=f1)
# 
  IMG_H=tf.shape(logits)[1]
  IMG_W=tf.shape(logits)[2]
  sess=tf.Session()
  IMG_H_=sess.run(IMG_H)
  IMG_W_=sess.run(IMG_W)
  DISPARITY=192
  #print("softmax shape")
  #print(softmax.shape)
  disp = tf.range(1, (DISPARITY+1), 1)
  
  disp = tf.cast(disp, tf.float32)
  disp_mat = []
  for i in range(BATCH_SIZE*IMG_H_*IMG_W_):
    disp_mat.append(disp)
  #print("lenth")
  #print(len(disp_mat))
  disp_mat = tf.reshape(tf.stack(disp_mat), [BATCH_SIZE,IMG_H,IMG_W,DISPARITY])
  disp_mat=tf.to_float(disp_mat)
  
  #disp_mat_=sess.run(disp_mat)
  #print("disp_mat shape")
  #print(disp_mat_)
  #softmax_=sess.run(softmax)
  #print("softmax shape")
  #print(softmax_)
  #print(sess.run(disp_mat),file=f1)
  result = tf.multiply(softmax, disp_mat)
  #print("result shape before reduce")
  #print(result.shape)
  result = tf.reduce_sum(result, 3)
  #print("softmax result shape")
  result=tf.squeeze(result)
  #print(result.shape)
  #f1.close()
  sess.close()
  #sum_softmax=tf.reduce_sum(softmax,0)
  return result,softmax

def inference2(image_batch_left,image_batch_right,ground_truth_l,ground_truth_r,batch_size,training_flag):
     if training_flag=='training':
        is_training=tf.constant(True)
        KEPP_PROB=0.5
     if training_flag=='testing':
        is_training=tf.constant(False)
        KEPP_PROB=1.0
    
     DispRange=192
     final_left=[]
     final_right=[]
     
     #MAX_DISPLACEMENT=16
     STRIDE_2=1
     HEIGHT=256
     WIDTH=512

     #combine_image=corr(image_batch_left,image_batch_right,MAX_DISPLACEMENT,STRIDE_2,HEIGHT,WIDTH,1)
     
     combine_image = tf.concat([image_batch_left, image_batch_right], 3)
     #print("the size of lables" )
     #print(ground_truth.shape )
# conv layer
     with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
      ksize = 7
      stride = 2
      filter_out=64
      ratio=4
      
      x1 = conv_2d(combine_image, ksize,stride,filter_out)
      #x1=Squeeze_excitation_layer(x1, filter_out, ratio, 'Senet')
      x1=Relu6(x1)
      #x1=Batch_Normalization(x1, is_training,'conv1')
      #print("x1")
      #print(x1.shape)
      

     with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
      ksize = 5
      stride = 2
      filter_out=128
      ratio=4
      #x1=tf.nn.dropout(x1,KEPP_PROB)
      x2 = conv_2d(x1, ksize,stride,filter_out)
      #x2=Squeeze_excitation_layer(x2, filter_out, ratio, 'Senet')
      x2=Relu6(x2)
      #x2=Batch_Normalization(x2, is_training,'conv2')
      #print("x2")
      #print(x2.shape)

     with tf.variable_scope('conv3a',reuse=tf.AUTO_REUSE):
      ksize = 5
      stride = 2
      filter_out=256
      ratio=4
      #x2=tf.nn.dropout(x2,KEPP_PROB)
      x3 = conv_2d(x2, ksize,stride,filter_out)
      #x3=Squeeze_excitation_layer(x3, filter_out, ratio, 'Senet')
      x3=Relu6(x3)
      #x3=Batch_Normalization(x3, is_training,'conv3a')
      #print("x3")
      #print(x3.shape)

     with tf.variable_scope('conv3b',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 1
      filter_out=256
      ratio=4
      #x3=tf.nn.dropout(x3,KEPP_PROB)
      x4 = conv_2d(x3, ksize,stride,filter_out)
      #x4=Squeeze_excitation_layer(x4, filter_out, ratio, 'Senet')
      x4=Relu6(x4)
      #x4=Batch_Normalization(x4, is_training,'conv3b')
      #print("x4")
      #print(x4.shape)

     with tf.variable_scope('conv4a',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 2
      filter_out=512
      ratio=4
      #x4=tf.nn.dropout(x4,KEPP_PROB)
      x5 = conv_2d(x4, ksize,stride,filter_out)
      #x5=Squeeze_excitation_layer(x5, filter_out, ratio, 'Senet')
      x5=Relu6(x5)
      #x5=Batch_Normalization(x5, is_training,'conv4a')
      #print("x5")
      #print(x5.shape)

     with tf.variable_scope('conv4b',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 1
      filter_out=512
      ratio=4
      #x5=tf.nn.dropout(x5,KEPP_PROB)
      x6 = conv_2d(x5, ksize,stride,filter_out)
      #x6=Squeeze_excitation_layer(x6, filter_out, ratio, 'Senet')
      x6=Relu6(x6)
      #x6=Batch_Normalization(x6, is_training,'conv4b')
      #print("x6")
      #print(x6.shape)

     with tf.variable_scope('conv5a',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 2
      filter_out=512
      ratio=4
      #x6=tf.nn.dropout(x6,KEPP_PROB)
      x7 = conv_2d(x6, ksize,stride,filter_out)
      #x7=Squeeze_excitation_layer(x7, filter_out, ratio, 'Senet')
      x7=Relu6(x7)
      #x7=Batch_Normalization(x7, is_training,'conv5a')
      #print("x7")
      #print(x7.shape)

     with tf.variable_scope('conv5b',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 1
      filter_out=512
      ratio=4
      #x7=tf.nn.dropout(x7,KEPP_PROB)
      x8 = conv_2d(x7, ksize,stride,filter_out)
      #x8=Squeeze_excitation_layer(x8, filter_out, ratio, 'Senet')
      x8=Relu6(x8)
      #x8=Batch_Normalization(x8, is_training,'conv5b')
      #print("x8")
      #print(x8.shape)

     with tf.variable_scope('conv6a',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 1
      filter_out=1024
      ratio=4
      #x8=tf.nn.dropout(x8,KEPP_PROB)
      x9 = conv_2d(x8, ksize,stride,filter_out)
      #x9=Squeeze_excitation_layer(x9, filter_out, ratio, 'Senet')
      x9=Relu6(x9)
      #x9=Batch_Normalization(x9, is_training,'conv6a')
      #print("x9")
      #print(x9.shape)

     with tf.variable_scope('conv6b',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 2
      filter_out=1024
      ratio=4
      #x9=tf.nn.dropout(x9,KEPP_PROB)
      x10 = conv_2d(x9, ksize,stride,filter_out)
      #x10=Squeeze_excitation_layer(x10, filter_out, ratio, 'Senet')
      x10=Relu6(x10)
      #x10=Batch_Normalization(x10, is_training,'conv6b')
      #print("x10")
      #print(x10.shape)

     with tf.variable_scope('pr6_loss6',reuse=tf.AUTO_REUSE):
      ksize = 3
      stride = 1
      filter_out=DispRange
      ratio=4
      
      with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
          x11_l = conv_2d(x10, ksize,stride,filter_out)
          x11_l=Relu6(x11_l)
          
          #x11_l=tf.argmax(x11_l,axis=3)
          #x11_l=Batch_Normalization(x11_l, is_training,'pr6_loss6_l')
          #print("x11_l")
          #print(x11_l.shape)
          
          scope.reuse_variables()
      #with tf.variable_scope('Right',reuse=tf.AUTO_REUSE):
          x11_r = conv_2d(x10, ksize,stride,filter_out)
          x11_r=Relu6(x11_r)
          #x11_r=tf.argmax(x11_r,axis=3)
          #x11_r=Batch_Normalization(x11_r, is_training,'pr6_loss6_r')
          #print("x11_r")
          #print(x11_r.shape)
      
      pr6_l,_=computeSoftArgMin(x11_l,batch_size)
      pr6_r,_=computeSoftArgMin(x11_r,batch_size)
      
      final_left.append(pr6_l)
      final_right.append(pr6_r)
      
      gt6_l = tf.nn.max_pool(ground_truth_l, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6_l')
      gt6_r = tf.nn.max_pool(ground_truth_r, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6_r')
      #print("check the shape of pr6,gt6")
      #print("shape of pr6")
      #print(pr6_l.shape)
      #print("shape of gt6")
      #print(gt6_l.shape)
      #print("check the shape of pr6,gt6")
      #print(pr6_r.shape)
      #print("shape of gt6")
      #print(gt6_r.shape)
      
      loss6 = loss(pr6_l, gt6_l)+loss(pr6_r, gt6_r)

     # upconv5
     with tf.variable_scope('upconv5',reuse=tf.AUTO_REUSE):
        ksize = 4
        stride = 2
        filter_out=512
        ratio=4
        #W_upconv5 = weight_variable([4,4, 512,1024]) 
        #b_upconv5 = bias_variable([512])
        #x12=upconv2d_2x2(x10,  W_upconv5, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 512]) + b_upconv5, center=True, scale=True, is_training=True)
        x10=tf.nn.dropout(x10,KEPP_PROB)
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x12_l = upconv2d_2x2(x10, ksize,stride,filter_out,batch_size)
            x12_l=Relu6(x12_l)
            #x12_l=Batch_Normalization(x12_l, is_training,'upconv5')
            #print("x12_l")
            #print(x12_l.shape)
        
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x12_r = upconv2d_2x2(x10, ksize,stride,filter_out,batch_size)
            x12_r=Relu6(x12_r)
            #x12_r=Batch_Normalization(x12_r, is_training,'upconv5')
            #print("x12_r")
            #print(x12_r.shape)

     # iconv5
     with tf.variable_scope('iconv5',reuse=tf.AUTO_REUSE):
        #print("concat")
        #print("x12_l")
        #print(x12_l.shape)
        #print("x8")
        #print(x8.shape)
        cat1_l=tf.concat([x12_l, x8], 3)
        
        #print("concat")
        #print("x12_r")
        #print(x12_r.shape)
        #print("x8")
        #print(x12_r.shape)
        cat1_r=tf.concat([x12_r, x8], 3)
        
        ksize = 3
        stride = 1
        filter_out=512
        ratio=4
        
        #cat1_l=tf.nn.dropout(cat1_l,KEPP_PROB)
        #cat1_r=tf.nn.dropout(cat1_r,KEPP_PROB)
        
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x13_l = conv_2d(cat1_l, ksize,stride,filter_out)
            x13_l=Relu6(x13_l)
            #x13_l=Batch_Normalization(x13_l, is_training,'iconv5')
            #print("x13_l")
            #print(x13_l.shape)
        
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x13_r = conv_2d(cat1_r, ksize,stride,filter_out)
            x13_r=Relu6(x13_r)
            #x13_r=Batch_Normalization(x13_r, is_training,'iconv5')
            #print("x13_r")
            #print(x13_r.shape)

     with tf.variable_scope('pr5_loss5',reuse=tf.AUTO_REUSE):
        ksize = 3
        stride = 1
        filter_out=DispRange
        ratio=4
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x14_l = conv_2d(x13_l, ksize,stride,filter_out)
            x14_l=Relu6(x14_l)
            #x14_l_temp=tf.argmax(x14_l,axis=3)
            #x14_l=Batch_Normalization(x14_l, is_training,'pr5_loss5_l')
            print("x14_l")
            print(x14_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x14_r = conv_2d(x13_r, ksize,stride,filter_out)
            x14_r=Relu6(x14_r)
            #x14_r_temp=tf.argmax(x14_r,axis=3)
            #x14_r=Batch_Normalization(x14_r, is_training,'pr5_loss5_r')
            print("x14_r")
            print(x14_r.shape)
        
        pr5_l,_=computeSoftArgMin(x14_l,batch_size)
        pr5_r,_=computeSoftArgMin(x14_r,batch_size)
        
        final_left.append(pr5_l)
        final_right.append(pr5_r)
      
        # pr5 = pre(h_iconv5)
        gt5_l = tf.nn.max_pool(ground_truth_l, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5_l')
        gt5_r = tf.nn.max_pool(ground_truth_r, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5_r')
        loss5 = loss(pr5_l, gt5_l)+loss(pr5_r, gt5_r)

     # upconv4
     with tf.variable_scope('upconv4',reuse=tf.AUTO_REUSE):
        ksize = 4
        stride = 2
        filter_out=256
        ratio=4
        
        #x13_l=tf.nn.dropout(x13_l,KEPP_PROB)
        #x13_r=tf.nn.dropout(x13_r,KEPP_PROB)
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x15_l = upconv2d_2x2(x13_l, ksize,stride,filter_out,batch_size)
            x15_l=Relu6(x15_l)
            #x15_l=Batch_Normalization(x15_l, is_training,'upconv4')
            #print("x15_l")
            #print(x15_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x15_r = upconv2d_2x2(x13_r, ksize,stride,filter_out,batch_size)
            x15_r=Relu6(x15_r)
            #x15_r=Batch_Normalization(x15_r, is_training,'upconv4')
            #print("x15_r")
            #print(x15_r.shape)

     # iconv4
     with tf.variable_scope('iconv4',reuse=tf.AUTO_REUSE):
        #print("concat")
        #print("x6")
        #print(x6.shape)
        #print("x15")
        #print(x15_l.shape)
        
        #print("concat")
        #print("x6")
        #print(x6.shape)
        #print("x15")
        #print(x15_r.shape)
        
        
        cat2_l=tf.concat([x15_l, x6], 3)
        cat2_r=tf.concat([x15_r, x6], 3)
        
        ksize = 3
        stride = 1
        filter_out=256
        ratio=4
        
        #cat2_l=tf.nn.dropout(cat2_l,KEPP_PROB)
        #cat2_r=tf.nn.dropout(cat2_r,KEPP_PROB)
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x16_l = conv_2d(cat2_l, ksize,stride,filter_out)
            x16_l=Relu6(x16_l)
            #x16_l=Batch_Normalization(x16_l, is_training,'iconv4')
            #print("x16_l")
            #print(x16_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x16_r = conv_2d(cat2_r, ksize,stride,filter_out)
            x16_r=Relu6(x16_r)
            #x16_r=Batch_Normalization(x16_r, is_training,'iconv4')
            #print("x16_r")
            #print(x16_r.shape)

     with tf.variable_scope('pr4_loss4',reuse=tf.AUTO_REUSE):
        ksize = 3
        stride = 1
        filter_out=DispRange
        ratio=4
        
        
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x17_l = conv_2d(x16_l, ksize,stride,filter_out)
            x17_l=Relu6(x17_l)
            #x17_l=tf.argmax(x17_l,axis=3)
            #x17_l=Batch_Normalization(x17_l, is_training,'pr4_loss4_l')
            #print("x17_l")
            #print(x17_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x17_r = conv_2d(x16_r, ksize,stride,filter_out)
            x17_r=Relu6(x17_r)
            #x17_r=tf.argmax(x17_r,axis=3)
            #x17_r=Batch_Normalization(x17_r, is_training,'pr4_loss4_r')
            #print("x17_r")
            #print(x17_r.shape)
        
        pr4_l,_=computeSoftArgMin(x17_l,batch_size)
        pr4_r,_=computeSoftArgMin(x17_r,batch_size)
        
        final_left.append(pr4_l)
        final_right.append(pr4_r)
        
        # pr5 = pre(h_iconv5)
        gt4_l = tf.nn.max_pool(ground_truth_l, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
        gt4_r = tf.nn.max_pool(ground_truth_r, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
        loss4 = loss(pr4_l, gt4_l)+loss(pr4_r, gt4_r)


     # upconv3
     with tf.variable_scope('upconv3',reuse=tf.AUTO_REUSE) as scope:
        ksize = 4
        stride = 2
        filter_out=128
        ratio=4
        
        #x16_l=tf.nn.dropout(x16_l,KEPP_PROB)
        #x16_r=tf.nn.dropout(x16_r,KEPP_PROB)
        
        with tf.variable_scope('Left'):
            x18_l = upconv2d_2x2(x16_l, ksize,stride,filter_out,batch_size)
            x18_l=Relu6(x18_l)
            #x18_l=Batch_Normalization(x18_l, is_training,'upconv3')
            #print("x18_l")
            #print(x18_l.shape)
             
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x18_r = upconv2d_2x2(x16_r, ksize,stride,filter_out,batch_size)
            x18_r=Relu6(x18_r)
            #x18_r=Batch_Normalization(x18_r, is_training,'upconv3')
            #print("x18_r")
            #print(x18_r.shape)

     # iconv3
     with tf.variable_scope('iconv3',reuse=tf.AUTO_REUSE) as scope:
        #print("concat")
        #print("x4")
        #print(x4.shape)
        #print("x18_l")
        #print(x18_l.shape)
        
        #print("concat")
        #print("x4")
        #print(x4.shape)
        #print("x18_r")
        #print(x18_r.shape)
        
        cat3_l=tf.concat([x18_l, x4], 3)
        cat3_r=tf.concat([x18_r, x4], 3)
        
        #cat3_l=tf.nn.dropout(cat3_l,KEPP_PROB)
        #cat3_r=tf.nn.dropout(cat3_r,KEPP_PROB)
        
        ksize = 4
        stride = 1
        filter_out=128
        ratio=4
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x19_l = conv_2d(cat3_l, ksize,stride,filter_out)
            x19_l=Relu6(x19_l)
            #x19_l=Batch_Normalization(x19_l, is_training,'iconv3')
            #print("x19_l")
            #print(x19_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x19_r = conv_2d(cat3_r, ksize,stride,filter_out)
            x19_r=Relu6(x19_r)
            #x19_r=Batch_Normalization(x19_r, is_training,'iconv3')
            #print("x19_r")
            #print(x19_r.shape)


     with tf.variable_scope('pr3_loss3',reuse=tf.AUTO_REUSE) as scope:
        ksize = 3
        stride = 1
        filter_out=DispRange
        ratio=4
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE):
            x20_l = conv_2d(x19_l, ksize,stride,filter_out)
            x20_l=Relu6(x20_l)
            #x20_l=tf.argmax(x20_l,axis=3)
            #x20_l=Batch_Normalization(x20_l, is_training,'pr4_loss3_l')
            #print("x20_l")
            #print(x20_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x20_r = conv_2d(x19_r, ksize,stride,filter_out)
            x20_r=Relu6(x20_r)
            #x20_r=tf.argmax(x20_r,axis=3)
            #x20_r=Batch_Normalization(x20_r, is_training,'pr4_loss3_r')
            #print("x20_r")
            #print(x20_r.shape)
        
        pr3_l,_=computeSoftArgMin(x20_l,batch_size)
        pr3_r,_=computeSoftArgMin(x20_r,batch_size)
        
        final_left.append(pr3_l)
        final_right.append(pr3_r)
        # pr5 = pre(h_iconv5)
        gt3_l = tf.nn.max_pool(ground_truth_l, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt3')
        gt3_r = tf.nn.max_pool(ground_truth_r, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt3')
        loss3 = loss(pr3_l, gt3_l)+loss(pr3_r, gt3_r)

     # upconv2
     with tf.variable_scope('upconv2',reuse=tf.AUTO_REUSE) as scope:
        ksize = 4
        stride = 2
        filter_out=64
        ratio=4
        
        #x19_l=tf.nn.dropout(x19_l,KEPP_PROB)
        #x19_r=tf.nn.dropout(x19_r,KEPP_PROB)
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE):
            x21_l = upconv2d_2x2(x19_l, ksize,stride,filter_out,batch_size)
            x21_l=Relu6(x21_l)
            #x21_l=Batch_Normalization(x21_l, is_training,'upconv2')
            #print("x21_l")
            #print(x21_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x21_r = upconv2d_2x2(x19_r, ksize,stride,filter_out,batch_size)
            x21_r=Relu6(x21_r)
            #x21_r=Batch_Normalization(x21_r, is_training,'upconv2')
            #print("x21_r")
            #print(x21_r.shape)
        

     # iconv2
     with tf.variable_scope('iconv2',reuse=tf.AUTO_REUSE) :
        #print("concat")
        #print("x2")
        #print(x2.shape)
        #print("x21")
        #print(x21_l.shape)
        
        #print("concat")
        #print("x2")
        #print(x2.shape)
        #print("x21")
        #print(x21_r.shape)
        
        cat4_l=tf.concat([x21_l, x2], 3)
        cat4_r=tf.concat([x21_r, x2], 3)
        
        ksize = 4
        stride = 1
        filter_out=64
        ratio=4
        
        #cat4_l=tf.nn.dropout(cat4_l,KEPP_PROB)
        #cat4_r=tf.nn.dropout(cat4_r,KEPP_PROB)
        
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            
            x22_l = conv_2d(cat4_l,ksize,stride,filter_out)
            x22_l=Relu6(x22_l)
            x22_l=Batch_Normalization(x22_l, is_training,'iconv2')
            #print("x22_l")
            #print(x22_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x22_r = conv_2d(cat4_r,ksize,stride,filter_out)
            x22_r=Relu6(x22_r)
            #x22_r=Batch_Normalization(x22_r, is_training,'iconv2')
            #print("x22_r")
            #print(x22_r.shape)

#################2018.12.20#################
        
     with tf.variable_scope('pr2_loss2',reuse=tf.AUTO_REUSE) :
        ksize = 3
        stride = 1
        filter_out=DispRange
        ratio=4
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x23_l = conv_2d(x22_l, ksize,stride,filter_out)
            x23_l=Relu6(x23_l)
            #x23_l=tf.argmax(x23_l,axis=3)
            #x23_l=Batch_Normalization(x23_l, is_training,'pr2_loss2_l')
            #print("x23_l")
            #print(x23_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x23_r = conv_2d(x22_r, ksize,stride,filter_out)
            x23_r=Relu6(x23_r)
            #x23_r=tf.argmax(x23_r,axis=3)
            #x23_r=Batch_Normalization(x23_r, is_training,'pr2_loss2_r')
            #print("x23_r")
            #print(x23_r.shape)

        pr2_L,_=computeSoftArgMin(x23_l,batch_size)
        pr2_r,_=computeSoftArgMin(x23_r,batch_size)
        
        final_left.append(pr2_L)
        final_right.append(pr2_r)
        # pr5 = pre(h_iconv5)
        gt2_l = tf.nn.max_pool(ground_truth_l, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt2')
        gt2_r = tf.nn.max_pool(ground_truth_r, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt2')
        loss2 = loss(pr2_L, gt2_l)+loss(pr2_r, gt2_r)

     # upconv1
     with tf.variable_scope('upconv1',reuse=tf.AUTO_REUSE):
        ksize = 4
        stride = 2
        filter_out=32
        ratio=4
        
        #x22_l=tf.nn.dropout(x22_l,KEPP_PROB)
        #x22_r=tf.nn.dropout(x22_r,KEPP_PROB)
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x24_l = upconv2d_2x2(x22_l, ksize,stride,filter_out,batch_size)
            x24_l=Relu6(x24_l)
            #x24_l=Batch_Normalization(x24_l, is_training,'upconv1')
            #print("x24_l")
            #print(x24_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x24_r = upconv2d_2x2(x22_r, ksize,stride,filter_out,batch_size)
            x24_r=Relu6(x24_r)
            #x24_r=Batch_Normalization(x24_r, is_training,'upconv1')
            #print("x24_r")
            #print(x24_r.shape)

     # iconv1
     with tf.variable_scope('iconv1',reuse=tf.AUTO_REUSE):
        #print("concat")
        #print("x1")
        #print(x1.shape)
        #print("x24")
        #print(x24_l.shape)
        
        #print("concat")
        #print("x1")
        #print(x1.shape)
        #print("x24")
        #print(x24_r.shape)
        
        cat5_l=tf.concat([x24_l, x1], 3)
        cat5_r=tf.concat([x24_r, x1], 3)
        
        ksize = 4
        stride = 1
        filter_out=32
        ratio=4
        
        #cat5_l=tf.nn.dropout(cat5_l,KEPP_PROB)
        #cat5_r=tf.nn.dropout(cat5_r,KEPP_PROB)
        
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x25_l = conv_2d(cat5_l, ksize,stride,filter_out)
            x25_l=Relu6(x25_l)
            x25_l=Batch_Normalization(x25_l, is_training,'iconv1')
            #print("x25_l")
            #print(x25_l.shape)
        
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x25_r = conv_2d(cat5_r, ksize,stride,filter_out)
            x25_r=Relu6(x25_r)
            #x25_r=Batch_Normalization(x25_r, is_training,'iconv1')
            #print("x25_r")
            #print(x25_r.shape)

     with tf.variable_scope('pr1_loss1',reuse=tf.AUTO_REUSE):
        ksize = 3
        stride = 1
        filter_out=DispRange
        ratio=4
        
        with tf.variable_scope('Left',reuse=tf.AUTO_REUSE) as scope:
            x26_l = conv_2d(x25_l, ksize,stride,filter_out)
            x26_l=Relu6(x26_l)
            #x26_l=tf.argmax(x26_l,axis=3)
            #x26_l=Batch_Normalization(x26_l, is_training,'pr1_loss1_l')
            #print("x26_l")
            #print(x26_l.shape)
            
            scope.reuse_variables()
        #with tf.variable_scope('right',reuse=tf.AUTO_REUSE):
            x26_r = conv_2d(x25_r, ksize,stride,filter_out)
            x26_r=Relu6(x26_r)
            #x26_r=tf.argmax(x26_r,axis=3)
            #x26_r=Batch_Normalization(x26_r, is_training,'pr1_loss1_r')
            #print("x26_r")
            #print(x26_r.shape)
        
        pr1_l,_=computeSoftArgMin(x26_l,batch_size)
        pr1_r,_=computeSoftArgMin(x26_r,batch_size)
        
        final_left.append(pr1_l)
        final_right.append(pr1_r)
        # pr5 = pre(h_iconv5)
        gt1_l = tf.nn.max_pool(ground_truth_l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt5')
        gt1_r = tf.nn.max_pool(ground_truth_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt5')
        
       
        loss1 = loss(pr1_l, gt1_l)+loss(pr1_r, gt1_r)
        #return final_left,final_right
     #dispmap=computeSoftArgMin(logits,BATCH_SIZE)
     with tf.name_scope('loss'):
        #total_loss = ( 1/2 * loss1 + 1/4 * loss2 + 1/8 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6)/2/batch_size
        total_loss=loss1
     return total_loss,final_left,final_right







def evalrate(logits, labels):
    with tf.variable_scope("accuracy"):
        print("accuracy")
        print("check the size of logits")
        print(logits.shape)
        print("check the size of labels")
        print(labels.shape)
        
        logits=tf.squeeze(logits)
        labels=tf.squeeze(labels)
        
    
        height=tf.shape(logits)[0]
        width=tf.shape(logits)[1]
        
        sub1=tf.abs(tf.subtract(logits,labels))
        sess=tf.Session()
        sub1_=sess.run(sub1)
        height_=sess.run(height)
        width_=sess.run(width)
        
        total_correct_num=0
        for i in sub1_:
            if i<=3:
                total_correct_num=total_correct_num+1

        accuracy = total_correct_num/height_/width_
        sess.close()
    return accuracy

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    from IO import read,write
#    from skimage import transform
    from PIL import Image
    import numpy as np
    import operator
    from load_data import *
    
    import cv2

    tf.reset_default_graph()
    IMG_SIZE = 256
    BATCH_SIZE = 1
    CAPACITY = 200
    IMG_W=512
    IMG_H=256

    #pwd=os.getcwd()
    #logs_dir = pwd+'\\logs_1\\'     # 检查点保存路径
    #train_l="000056_10l.jpg"
    #train_r="000056_10r.jpg"
    #train_g="000056_10d.jpg"
   # 
   # train_l2="000055_10l.jpg"
   # train_r2="000055_10r.jpg"
   # train_g2="000055_10d.jpg"
   # 
    MPI_path='./MPI-Sintel-stereo-training-20150305/training'
    recfile='doc1.txt'
    recfile2='doc2.txt'
   
    print("use my own function")
    fly3d_path='D:/flyingthings3d__frames_finalpass'
    disparity_path='D:/disparity'
    all_image_list=get_flingthings3d_list(fly3d_path,disparity_path)
    mini_batches=get_mini_batch(all_image_list,BATCH_SIZE)
    imgl,imgr,imggl,imggr=read_batch_image(mini_batches[0],IMG_H,IMG_W)
    
    test1=imgl[0]
    
    #test2=cv2.resize(test1,(300,300))
    #cv2.imshow('dst_image', test1)
    #cv2.waitKey(10)

    
    image_left=tf.convert_to_tensor(np.asarray(imgl))
    image_right=tf.convert_to_tensor(np.asarray(imgr))
    ground_truth_l=tf.convert_to_tensor(np.asarray(imggl))
    ground_truth_r=tf.convert_to_tensor(np.asarray(imggr))
    
    
    image_left=tf.to_float(image_left, name='ToFloat')
    image_right=tf.to_float(image_right, name='ToFloat')
    ground_truth_l=tf.to_float(ground_truth_l, name='ToFloat')
    ground_truth_r=tf.to_float(ground_truth_r, name='ToFloat')
    print("sad")

    #image_left = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 3], name='image_left')
    #image_right = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 3], name='image_right') 
    #ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1], name='ground_truth')
    
    
    #ground_truth = tf.image.resize_image_with_crop_or_pad(ground_truth, IMG_SIZE, IMG_SIZE)
    #image_right = tf.image.resize_image_with_crop_or_pad(image_right, IMG_SIZE, IMG_SIZE)
    #image_left = tf.image.resize_image_with_crop_or_pad(image_left, IMG_SIZE, IMG_SIZE)
    
    print("shape")
    print(image_left.shape)
    
    image_left = tf.cast(image_left, tf.float32)
    image_right = tf.cast(image_right, tf.float32)
    image_gt_l = tf.cast(ground_truth_l, tf.float32)
    image_gt_r = tf.cast(ground_truth_r, tf.float32)
    
    
    
    #inference(image_batch_left,image_batch_right,ground_truth_l,ground_truth_r,batch_size,training_flag)
    total_loss,final_left,final_right=inference2(image_left,image_right,image_gt_l,image_gt_r,BATCH_SIZE,'training')
    #total_loss=tf.reduce_mean(total_loss)
    #softmax_1=tf.reduce_sum(softmax_1,3)
    test=final_left[0]
    #print(len(image_batch)) 
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    #print("测试一下图片的大小")
    #print(imgl.shape)
    #pr1_, total_loss_ = sess.run([pr1, total_loss],feed_dict={image_left: np.asarray(imgl), image_right: np.asarray(imgr), ground_truth:np.asarray(imgg)})
    test_= sess.run([test])
    total_loss_= sess.run([total_loss])
    #softmax_1_=sess.run(softmax_1)
    
    print("test_")
    print(test_)

    print("total_loss_")
    print(total_loss_)
    