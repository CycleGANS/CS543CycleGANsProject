import tensorflow as tf

# General convolution layer.
def conv2d_layer(inputconv, num_filter=64, filter_h=7, filter_w=7, stride_h=1, stride_w=1, stddev=0.02, 
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        
        conv = tf.contrib.layers.conv2d(inputconv, num_filter, filter_h, stride_h, padding, activation_fn=None, 
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        
        if do_norm:
            conv = instance_norm(conv)
            
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

# Build model: simplified discriminator
def build_gen_discriminator(input_src, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        layer1 = conv2d_layer(input_src, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        layer2 = conv2d_layer(layer1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        layer3 = conv2d_layer(layer2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        layer4 = conv2d_layer(layer3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        layer5 = conv2d_layer(layer4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return layer5



