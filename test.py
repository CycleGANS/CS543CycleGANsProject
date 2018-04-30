import tensorflow as tf
import Generator as gen
from io_tools import *
import glob
import scipy.misc


# ---- Need this !!! -----
from utils import utils
from utils import image_utils as im


def test(dataset_str='horse2zebra', img_width=256, img_height=256):
    """Test and save output images.
    
    Args:
        dataset_str: Name of the dataset
        X_path, Y_path: Path to data in class X or Y
    """
    image_shape = img_width

    if image_shape == 256:
        no_of_residual_blocks = 9
    elif image_shape == 128:
        no_of_residual_blocks = 6

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # X and Y are for real images.
        X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
        Y = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
        
        # Build graph for generator to produce images from real data.
        GofX = gen.generator(X, no_of_residual_blocks, scope='G', output_channels=64)
        FofY = gen.generator(Y, no_of_residual_blocks, scope='F', output_channels=64)
        # Convert transformed images back to original one (cyclic).
        Fof_GofX = gen.generator(GofX, no_of_residual_blocks, scope='G', output_channels=64)
        Gof_FofY = gen.generator(FofY, no_of_residual_blocks,  scope='F', output_channels=64)

        # Restore checkpoint.
        # --------------- Need to implement utils!!!!! ----------------
        try:
            saver.restore(sess, "/Checkpoints/"+dataset)
        except:
            raise Exception('No checkpoint available!')
        
        # Load data and preprocess (resize and crop).
        X_path_ls = glob('./Datasets/' + dataset_str + '/testA/*.jpg')
        Y_path_ls = glob('./Datasets/' + dataset_str + '/testB/*.jpg')
        
        batch_size_X = len(X_path_ls)
        batch_size_Y = len(Y_path_ls)

        X_data = getdata(sess, X_path_ls, batch_size_X)
        Y_data = getdata(sess, Y_path_ls, batch_size_Y)

        # Get data into [batch_size, img_width, img_height, channels]
        X_batch = batch(sess, X_data)
        Y_batch = batch(sess, Y_data)

        # Feed into test procedure to test and save results.
        X_save_dir = './Outputs/Test/' + dataset_str + '/testA'
        Y_save_dir = './Outputs/Test/' + dataset_str + '/testB'
        # utils.mkdir([X_save_dir, Y_save_dir])

        _test_procedure(X_batch, sess, GofX, Fof_GofX, X, X_save_dir)
        _test_procedure(Y_batch, sess, FofY, Gof_FofY, Y, Y_save_dir)



def _test_procedure(batch, sess, gen_real, gen_cyc, real_placeholder, save_dir):
    """Procedure to perform test on a batch of real images and save outputs.
    Args:
        gen_real: Generator that maps real data to fake image.
        gen_cyc: Generator that maps fake image back to original image.
        real_placeholder: Placeholder for real image.
        save_dir: Directory to save output image.
    """
    for i in range(tf.shape(batch)[0]):
        # A single real image in batch.
        real_img = batch[i]
        # Generate fake and cyclic images.
        gen_real_out, gen_cyc_out = sess.run([gen_real, gen_cyc], 
                                        feed_dict={real_placeholder: real_img})
        # Concatenate 3 images into one.
        # out_img = np.concatenate((real_img, gen_real_out, gen_cyc_out), axis=0)
        # # Save result.
        # # --------------- Need the utils file!!! ---------------
        # # Temporarily use i as image name. Should change this.
        # im.imwrite(im.immerge(out_img, 1, 3), save_dir + '/' + str(i))
        


        gen_real_out_image = Image.fromarray(gen_real_out, "RGB")
        gen_cyc_out_image = Image.fromarray(gen_cyc_out, "RGB")

        new_im = Image.new('RGB', (image_shape*3, image_shape))
        new_im.paste(real_img, (0,0))
        new_im.paste(gen_real_out_image, (image_shape,0))
        new_im.paste(gen_cyc_out_image, (image_shape*2,0))

        new_im.save(save_dir+'(%d).jpg' % ( i )) 
        print("Save image.")
