# Import Library
import tensorflow as tf


# ### If you want to understand the conv2d function and its inputs, go to these pages
# #### https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
# #### https://stackoverflow.com/questions/34642595/tensorflow-strides-argument
# 
# ### If you want to understand the residual blocks used in the generator, go to this page
# #### http://torch.ch/blog/2016/02/04/resnets.html

# #### Functions for Batch Normalization, Residual Bloacks and Generator

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def residual_block(X, XW1, XW2, XB1, XB2):
    Y_res1 = tf.nn.conv2d(X, XW1, strides=[1, 1, 1, 1], padding='SAME')
    Y_res1_bn, update_emaY_res1 = batchnorm(Y_res, tst, iter, XB1, convolutional=True)
    Y_res1_r = tf.nn.relu(Y_res_bn)
    Y_res2 = tf.nn.conv2d(Y_re1_r, XW2, strides=[1, 1, 1, 1], padding='SAME')
    Y_res2_bn, update_emaY_res2 = batchnorm(Y_res2, tst, iter, XB2, convolutional=True)
    Y_res2_r = tf.nn.relu(Y_res2_bn)
    return Y_res2_r + X

def generator(input, scope):

    # #### Generator Variables
    with tf.variable_scope(scope):
        # DEFINING GENERATOR VARIABLES
        # ````````````````````````````
        # Define Variables for downsampling
        D1 = tf.get_variable('D1', dtype=tf.float32, initializer = tf.truncated_normal([9,9,3,output_channels1] ,stddev=0.1))  
        #Kernel size of 9x9 from 3 input channels (rgb). Number of output channels depends on how many you want in next layer.
        DB1 = tf.get_variable('DB1', dtype=tf.float32, initializer = tf.ones([output_channels1])/10)
        D2 = tf.get_variable('D2', dtype=tf.float32, initializer = tf.truncated_normal([3,3,output_channels1,K], stddev=0.1))
        #Kernel size of 3x3 from "output_channels1" number of input channels. Number of output channels K depends on how many you want in next layer.
        DB2 = tf.get_variable('DB2', dtype=tf.float32, initializer = tf.ones([K])/10)

        # Define Variables for Residual Blocks
        res_dict = {}
        for i in range(1,NO_OF_RESIDUAL_BLOCKS+1):
            res_dict[i] = {'R1'+str(i) :tf.get_variable('R1'+str(i), dtype=tf.float32, initializer = tf.truncated_normal([3,3,K,K] ,stddev=0.1)), 
                           'RB1'+str(i) :tf.get_variable('RB1'+str(i), dtype=tf.float32, initializer = tf.Variable(tf.ones([J])/10),
                           'R2'+str(i) :tf.get_variable('R2'+str(i), dtype=tf.float32, initializer = tf.Variable(tf.truncated_normal([3,3,K,K] ,stddev=0.1)), 
                           'RB2'+str(i) :tf.get_variable('RB2'+str(i), dtype=tf.float32, initializer = tf.Variable(tf.ones([J])/10)
                              }
             # Need to put the right input and output layer numbers


        # Define Variables for upsampling
        U1 = tf.get_variable('U1', dtype=tf.float32, initializer = tf.truncated_normal([3,3,number_of_input_channels,L], stddev=0.1))
        UB1 = tf.get_variable('UB1', dtype=tf.float32, initializer = tf.ones([L])/10)
        U2 = tf.get_variable('U2', dtype=tf.float32, initializer = tf.truncated_normal([9,9,L,3] ,stddev=0.1))
        #Here L is the number of output channels from the previous layer and is the number of input channels in this layer
        UB2 = tf.Variable('UB2', dtype=tf.float32, initializer = tf.ones([3])/10)
            
        # DEFINING THE LAYERS
        # ```````````````````
        # For Downsampling
        stride = 2  # if input A x A then output A/2 x A/2
        # Firt Layer
        YD1 = tf.nn.conv2d(input, D1, strides=[1, stride, stride, 1], padding='SAME')
        YD1bn, update_emaYD1 = batchnorm(YD1, tst, iter, DB1, convolutional=True)
        YD1r = tf.nn.relu(YD1bn)
        # Second Layer
        YD2 = tf.nn.conv2d(YD1r, D2, strides=[1, stride, stride, 1], padding='SAME')
        YD2bn, update_emaYD2 = batchnorm(YD2, tst, iter, DB2, convolutional=True)
        YD2r = tf.nn.relu(YD2bn)
        
        # For Residual Blocks
        YR = YD2r
        for i in range(1,NO_OF_RESIDUAL_BLOCKS+1):
            YR = residual_block(YR, res_dict[i]['R1'], res_dict[i]['R2'], res_dict[i]['RB1'], res_dict[i]['RB2'])
            
        # For Upsampling
        stride = 1/2
        # Second Last Layer
        YU1 = tf.nn.conv2d_transpose(YR, U1, strides=[1, stride, stride, 1], padding='SAME')
        YU1bn, update_emaYU1 = batchnorm(YU1, tst, iter, UB1, convolutional=True)
        YU1r = tf.nn.relu(YU1bn)
        # Last Layer
        YU2 = tf.nn.conv2d_transpose(YU1r, U2, strides=[1, stride, stride, 1], padding='SAME')
        Y_out = tf.nn.tanh(YU2)
        
        return Y_out



    

