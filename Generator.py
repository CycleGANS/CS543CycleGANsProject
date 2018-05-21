# Import Library
import tensorflow as tf


# ### If you want to understand the conv2d function and its inputs, go to these pages
# #### https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
# #### https://stackoverflow.com/questions/34642595/tensorflow-strides-argument
#
# ### If you want to understand the residual blocks used in the generator, go to this page
# #### http://torch.ch/blog/2016/02/04/resnets.html

# #### Functions for Batch Normalization, Residual Bloacks and Generator

def generator(input_imgs, no_of_residual_blocks, scope, output_channels=64):

    # Function for Batch Normalization
    def batchnorm(Ylogits):
        bn = tf.contrib.layers.batch_norm(Ylogits, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)
        return bn

    # Function for Convolution Layer
    def convolution_layer(input_images, filter_size, stride, o_c=64, padding="VALID", scope_name="convolution"):
        # o_c = Number of output channels/filters
        with tf.variable_scope(scope_name):
            conv = tf.contrib.layers.conv2d(input_images, o_c, filter_size, stride, padding=padding, activation_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
            return conv

    # Function for deconvolution layer
    def deconvolution_layer(input_images, o_c, filter_size, stride, padding="VALID", scope_name="deconvolution"):
        with tf.variable_scope(scope_name):
            deconv = tf.contrib.layers.conv2d_transpose(input_images, o_c, filter_size, stride, activation_fn=None,
                                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
            return deconv

    # Function for Residual Block
    def residual_block(Y, scope_name="residual_block"):
        with tf.variable_scope(scope_name):
            Y_in = tf.pad(Y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            Y_res1 = tf.nn.relu(batchnorm(convolution_layer(Y_in, filter_size=3, stride=1, o_c=output_channels * 4, scope_name="C1")))
            Y_res1 = tf.pad(Y_res1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            Y_res2 = batchnorm(convolution_layer(Y_res1, filter_size=3, stride=1, padding="VALID", o_c=output_channels * 4, scope_name="C2"))

            return Y_res2 + Y

    # #### Generator Variables
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # Need to pad the images first to get same sized image after first convolution
        input_imgs = tf.pad(input_imgs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        # print((input_imgs).shape)
        # print('-' * 40)
        # print(scope)
        # For Downsampling
        YD0 = tf.nn.relu(batchnorm(convolution_layer(input_imgs, filter_size=7, stride=1, o_c=output_channels, scope_name="D1")))
        YD1 = tf.nn.relu(batchnorm(convolution_layer(YD0, filter_size=3, stride=2, o_c=output_channels * 2, padding="SAME", scope_name="D2")))
        YD2 = tf.nn.relu(batchnorm(convolution_layer(YD1, filter_size=3, stride=2, o_c=output_channels * 4, padding="SAME", scope_name="D3")))

        # For Residual Blocks
        for i in range(1, no_of_residual_blocks + 1):
            Y_res = residual_block(YD2, scope_name="R" + str(i))

        # For Upsampling
        YU1 = tf.nn.relu(batchnorm(deconvolution_layer(Y_res, output_channels * 2, filter_size=3, stride=2, padding="SAME", scope_name="U1")))
        YU2 = tf.nn.relu(batchnorm(deconvolution_layer(YU1, output_channels, filter_size=3, stride=2, padding="SAME", scope_name="U2")))
        Y_out = tf.nn.tanh(convolution_layer(YU2, filter_size=7, stride=1, o_c=3, padding="SAME", scope_name="U3"))
        # import pdb
        # pdb.set_trace()
        return Y_out
