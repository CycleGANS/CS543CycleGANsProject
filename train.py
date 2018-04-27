import tensorflow as tf
import Generator as gen
import simple_discriminator as dis
import os
from PIL import Image
import numpy as np

def training(image_shape, G_cyc_loss_lambda = 10.0, F_cyc_loss_lambda = 10.0, learning_rate=0.0002 ):

    if image_shape == 256:
        no_of_residual_blocks = 9
    elif image_shape == 128:
        no_of_residual_blocks = 6

    # Creating placeholder for images
    X = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    Y = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    GofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    FofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    #GofFofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    #FofGofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])


    """ We will have 2 generators: G and F
    	G : X -> Y
    	F : Y -> X

    	and 2 Discriminators: DX and DY

    	DX aims to distinguish between images from {x} & translated images {F(y)}
    	DY aims to distinguish between images from {y} & translated images {G(x)}
    """

    # Creating the generators and discriminator networks
    GofX = gen.generator(X, no_of_residual_blocks, scope='G', output_channels=64)
    FofY = gen.generator(Y, no_of_residual_blocks, scope='F', output_channels=64)
    GofFofY = gen.generator(FofY, no_of_residual_blocks, scope='G', output_channels=64)
    FofGofX = gen.generator(GofX, no_of_residual_blocks,  scope='F', output_channels=64)

    D_Xlogits = dis.build_gen_discriminator(X, scope='DX')
    D_FofYlogits = dis.build_gen_discriminator(FofY, scope='DX')
    D_Ylogits = dis.build_gen_discriminator(Y, scope='DY')
    D_GofXlogits = dis.build_gen_discriminator(GofX, scope='DY')

    # Setting up losses for generators and discriminators
    """ adv_losses are adversary losses
        cyc_losses are cyclic losses
        real_losses are losses from real images
        fake_losses are from generated images
    """
    # https://arxiv.org/pdf/1611.04076.pdf this paper states that using cross entropy as loss
    # causes the gradient to vanish. To avoid this problem, least square losses are used as suggested by the paper.

    # Adversary and Cycle Losses for G
    G_adv_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.ones_like(D_GofXlogits)))
    G_cyc_loss = tf.reduce_mean(tf.abs(GofFofY - Y)) * G_cyc_loss_lambda        # Put lambda for G cyclic loss here
    G_tot_loss = G_adv_loss + G_cyc_loss

    # Adversary and Cycle Losses for F
    F_adv_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.ones_like(D_FofYlogits)))
    F_cyc_loss = tf.reduce_mean(tf.abs(FofGofX - X)) * F_cyc_loss_lambda        # Put lambda for F cyclic loss here
    F_tot_loss = F_adv_loss + F_cyc_loss

    # Total Losses for G and F
    GF_tot_loss = G_tot_loss + F_tot_loss


    # Losses for DX
    DX_real_loss = tf.reduce_mean(tf.squared_difference(D_Xlogits, tf.ones_like(D_Xlogits)))
    DX_fake_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.zeros_like(D_FofYlogits)))
    DX_tot_loss = (DX_real_loss + DX_fake_loss) / 2

    # Losses for DY
    DY_real_loss = tf.reduce_mean(tf.squared_difference(D_Ylogits, tf.ones_like(D_Ylogits)))
    DY_fake_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.zeros_like(D_GofXlogits)))
    DY_tot_loss = (DY_real_loss + DY_fake_loss) / 2

    # Optimization
    # Getting all the variables that belong to the different networks
    # I.e. The weights and biases in G, F, DX and DY
    network_variables = tf.trainable_variables()  # This gets all the variables that will be initialized
    GF_variables = [variables for variables in network_variables if 'G' in variables.name or 'F' in variables.name]
    DX_variables = [variables for variables in network_variables if 'DX' in variables.name]
    DY_variables = [variables for variables in network_variables if 'DY' in variables.name]

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)  # Put the learning rate here
    GF_train_step = optimizer.minimize(GF_tot_loss, var_list=GF_variables)
    DX_train_step = optimizer.minimize(DX_tot_loss, var_list=DX_variables)
    DY_train_step = optimizer.minimize(DY_tot_loss, var_list=DY_variables)


    # Summary for Tensor Board
    GF_summary = tf.summary.scalar("GF_tot_loss", GF_tot_loss)
    DX_summary = tf.summary.scalar("DX_tot_loss", DX_tot_loss)
    DY_summary = tf.summary.scalar("DY_tot_loss", DY_tot_loss)


    # Training
    # Initialization
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter( summmary_path + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter( summmary_path + '/test', sess.graph)

    dataset = 'horse2zebra'
    Xpath = glob('./datasets/' + dataset + '/trainA/*.jpg')
    Ypath = glob('./datasets/' + dataset + '/trainB/*.jpg')
    X_data = getdata(sess, Xpath, batch_size)
    Y_data = getdata(sess, Ypath, batch_size)

    for i in range(epochs):
        for j in range(no_of_batches):
            X_batch = batch(sess, X_data)
            Y_batch = batch(sess, Y_data)

            GofXforDis, FofYforDis = sess.run([GofX, FofY], feed_dict={X: X_batch,Y: Y_batch})

            DX_output, DX_vis_summary = sess.run([DX_train_step, DX_summary], feed_dict={X: X_batch, FofY: FofYforDis})

            DY_output, DY_vis_summary = sess.run([DY_train_step, DY_summary], feed_dict={Y: Y_batch, GofX: GofXforDis})

            GF_output, GF_vis_summ = sess.run([GF_train_step, GF_summary], feed_dict={X: X_batch, Y: Y_batch})

            train_summary_writer.add_summary(DX_vis_summary, j)
            train_summary_writer.add_summary(DY_vis_summary, j)
            train_summary_writer.add_summary(GF_vis_summ, j)

            if (j+1)%1000==0:
            	[GofX_sample, FofY_sample, GofFofY_sample, FofGofX_sample] = sess.run([GofX, FofY, GofFofY, FofGofX], feed_dict={X: X_batch, Y: Y_batch})

            	#Saving sample training images
            	# Works only for batch size 1
            	GofX_sample = Image.fromarray(GofX_sample, "RGB")
            	FofY_sample = Image.fromarray(FofY_sample, "RGB")
            	GofFofY_sample = Image.fromarray(GofFofY_sample, "RGB")
            	FofGofX_sample = Image.fromarray(FofGofX_sample, "RGB")

            	X_batch.save(os.path.join(cur_dir,"X"+str(i)+str(j)+".jpeg"))     #Need to define current directory and path
            	Y_batch.save(os.path.join(cur_dir,"Y"+str(i)+str(j)+".jpeg"))     #Need to define current directory and path
            	GofX_sample.save(os.path.join(cur_dir,"GofX"+str(i)+str(j)+".jpeg"))     #Need to define current directory and path
            	FofY_sample.save(os.path.join(cur_dir,"FofY"+str(i)+str(j)+".jpeg"))     #Need to define current directory and path
            	GofFofY_sample.save(os.path.join(cur_dir,"GofFofY"+str(i)+str(j)+".jpeg"))     #Need to define current directory and path
            	FofGofX_sample.save(os.path.join(cur_dir,"FofGofX"+str(i)+str(j)+".jpeg"))     #Need to define current directory and path

