import tensorflow as tf
import Generator as gen
import simple_discriminator as dis
import os
from PIL import Image
import numpy as np
import glob
import io_tools as io


def training(dataset, epochs, image_shape, batch_size, G_cyc_loss_lambda=10.0, F_cyc_loss_lambda=10.0, learning_rate=0.0002):

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
    FofGofX = gen.generator(GofX, no_of_residual_blocks, scope='F', output_channels=64)

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

    # For saving the model, the max_to_keep parameter saves just 5 models. I did this so that we don't run out of memory.
    saver = tf.train.Saver(max_to_keep=5)

    # Session on GPU
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Obtaining dataset
    # Training data
    """ Need to define getdata"""
    # dataset = 'horse2zebra'
    Xpath = glob.glob('./Datasets/' + dataset + '/trainA/*.jpg')
    Ypath = glob.glob('./Datasets/' + dataset + '/trainB/*.jpg')
    X_data = io.getdata(sess, Xpath, batch_size)     # Need to define getdata
    Y_data = io.getdata(sess, Ypath, batch_size)

    # Test data
    X_test_path = glob.glob('./Datasets/' + dataset + '/testA/*.jpg')
    Y_test_path = glob.glob('./Datasets/' + dataset + '/testB/*.jpg')
    X_test_data = io.getdata(sess, X_test_path, batch_size)     # Need to define getdata
    Y_test_data = io.getdata(sess, Y_test_path, batch_size)     # Need to define getdata

    # Creating a file to write the summaries for tensorboard
    train_summary_writer = tf.summary.FileWriter('./Summary/train/' + dataset, sess.graph)

    # Initialization if starting from scratch, else restore the variables
    try:
        saver.restore(sess, "/Checkpoints/" + dataset)
    except:
        init = tf.global_variables_initializer()
        sess.run(init)
    no_of_batches = min(len(Xpath), len(Ypath)) // batch_size
    # Training
    no_of_iterations = 0
    for i in range(1, epochs + 1):
        for j in range(1, no_of_batches + 1):
            no_of_iterations += 1

            X_batch = io.batch(sess, X_data)  # Define batch
            Y_batch = io.batch(sess, Y_data)

            # Creating fake images for the discriminators
            GofXforDis, FofYforDis = sess.run([GofX, FofY], feed_dict={X: X_batch, Y: Y_batch})

            DX_output, DX_vis_summary = sess.run([DX_train_step, DX_summary], feed_dict={X: X_batch, FofY: FofYforDis})

            DY_output, DY_vis_summary = sess.run([DY_train_step, DY_summary], feed_dict={Y: Y_batch, GofX: GofXforDis})

            GF_output, GF_vis_summ = sess.run([GF_train_step, GF_summary], feed_dict={X: X_batch, Y: Y_batch})

            train_summary_writer.add_summary(DX_vis_summary, no_of_iterations)
            train_summary_writer.add_summary(DY_vis_summary, no_of_iterations)
            train_summary_writer.add_summary(GF_vis_summ, no_of_iterations)

            # Creating Checkpoint
            if no_of_iterations % 800 == 0:
                save_path = saver.save(sess, '/Checkpoints/' + dataset + '/Epoch_(%d)_(%dof%d).ckpt' % (i, j, no_of_batches))
                print('Model saved in file: % s' % save_path)

            # To see what some of the test images look like after certain number of iterations
            if no_of_iterations % 150 == 0:
                X_test_batch = io.batch(sess, X_test_data)  # Define batch
                Y_test_batch = io.batch(sess, Y_test_data)
                [GofX_sample, FofY_sample, GofFofY_sample, FofGofX_sample] = sess.run([GofX, FofY, GofFofY, FofGofX], feed_dict={X: X_test_batch, Y: Y_test_batch})

                # Saving sample test images
                for l in range(batch_size):
                    X_test_image = Image.fromarray(X_test_batch[l], "RGB")
                    Y_test_image = Image.fromarray(Y_test_batch[l], "RGB")
                    GofX_image = Image.fromarray(GofX_sample[l], "RGB")
                    FofY_image = Image.fromarray(FofY_sample[l], "RGB")
                    GofFofY_image = Image.fromarray(GofFofY_sample[l], "RGB")
                    FofGofX_image = Image.fromarray(FofGofX_sample[l], "RGB")

                    new_im_X = Image.new('RGB', (image_shape * 3, image_shape))
                    new_im_X.paste(X_test_image, (0, 0))
                    new_im_X.paste(GofX_image, (image_shape, 0))
                    new_im_X.paste(FofGofX_image, (image_shape * 2, 0))

                    new_im_Y = Image.new('RGB', (image_shape * 3, image_shape))
                    new_im_Y.paste(Y_test_image, (0, 0))
                    new_im_Y.paste(FofY_image, (image_shape, 0))
                    new_im_Y.paste(GofFofY_image, (image_shape * 2, 0))

                    new_im_X.save('./Output/Train/X' + str(l) + '_Epoch_(%d)_(%dof%d).jpg' % (i, j, no_of_batches))
                    new_im_Y.save('./Output/Train/Y' + str(l) + '_Epoch_(%d)_(%dof%d).jpg' % (i, j, no_of_batches))

        print("Epoch: (%3d) Batch Number: (%5d/%5d)" % (i, j, no_of_batches))

    save_path = saver.save(sess, '/Checkpoints/' + dataset + '/Epoch_(%d)_(%dof%d).ckpt' % (i, j, no_of_batches))
    print('Model saved in file: % s' % save_path)
    sess.close()
