import tensorflow as tf
import Generator as gen
from io_tools import *
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

    with tf.Session() as sess:
        # X and Y are for real images.
        X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
        Y = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
        
        # Build graph for generator to produce images from real data.
        GofX = gen.generator(X, scope='G')
        FofY = gen.generator(Y, scope='F')
        # Convert transformed images back to original one (cyclic).
        Fof_GofX = gen.generator(GofX, scope='F')
        Gof_FofY = gen.generator(FofY, scope='G')

        # Restore checkpoint.
        # --------------- Need to implement utils!!!!! ----------------
        try:
            checkpoint_path = utils.load_checkpoint('./outputs/checkpoints/' + dataset_str, sess)
        except:
            raise Exception('No checkpoint available!')
        
        # Load data and preprocess (resize and crop).
        X_path_ls = glob('./datasets/' + dataset_str + '/trainA/*.jpg')
        Y_path_ls = glob('./datasets/' + dataset_str + '/trainB/*.jpg')
        
        batch_size_X = len(X_path_ls)
        batch_size_Y = len(Y_path_ls)

        X_data = getdata(sess, X_path_ls, batch_size_X)
        Y_data = getdata(sess, Y_path_ls, batch_size_Y)

        # Get data into [batch_size, img_width, img_height, channels]
        X_batch = batch(sess, X_data)
        Y_batch = batch(sess, Y_data)

        # Feed into test procedure to test and save results.
        X_save_dir = './outputs/test_predictions/' + dataset_str + '/testA'
        Y_save_dir = './outputs/test_predictions/' + dataset_str + '/testB'
        utils.mkdir([X_save_dir, Y_save_dir])

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
        out_img = np.concatenate((real_img, gen_real_out, gen_cyc_out), axis=0)
        # Save result.
        # --------------- Need the utils file!!! ---------------
        # Temporarily use i as image name. Should change this.
        im.imwrite(im.immerge(out_img, 1, 3), save_dir + '/' + str(i))
        print("Save image.")