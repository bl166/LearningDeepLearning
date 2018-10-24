__author__ = 'tan_nguyen'

import os
import time

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import Tensorflow and start a session
import tensorflow as tf
tf.reset_default_graph()

learning_rate = 1e-3
batch_size = 50


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''
    
    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_max


def conv_layer(x, dim, name, activation=tf.nn.relu):
    
    with tf.name_scope(name):
        with tf.name_scope('input'):
            variable_summaries(x)
            
        with tf.name_scope('weights'):
            W = weight_variable(dim)
            variable_summaries(W)
            
        with tf.name_scope('bias'):
            b = bias_variable([dim[-1]])
            variable_summaries(b)
            
        with tf.name_scope('activation'):
            z = conv2d(x, W) + b
            h = activation(z)
            variable_summaries(h)
            
        with tf.name_scope('activation_pooling'):
            a = max_pool_2x2(h)
            variable_summaries(a)
            
    return a


def fc_layer(x, dim, name, activation=tf.nn.relu):
    
    with tf.name_scope(name):
        with tf.name_scope('input'):
            x_flat = tf.reshape(x, [-1, dim[0]])
            variable_summaries(x_flat)
            
        with tf.name_scope('weights'):
            W = weight_variable(dim)
            variable_summaries(W)
            
        with tf.name_scope('bias'):
            b = bias_variable([dim[-1]])
            variable_summaries(b)
                    
        with tf.name_scope('activation'):
            z = tf.matmul(x_flat, W) + b
            h = activation(z)
            variable_summaries(h)
            
    return h



def main():
    
    with tf.device('/device:GPU:0'):
        
        # Specify training parameters
        result_dir = './results/res'+str(learning_rate)+'+bs'+str(batch_size) # directory where the results from the training are saved
        max_step = 5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

        start_time = time.time() # start timing

        # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK

        # placeholders for input data and input labeles
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        # reshape the input image
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # first convolutional layer
        h_pool1 = conv_layer(x_image, [5, 5, 1, 32], 'conv1')
        
        # second convolutional layer
        h_pool2 = conv_layer(h_pool1, [5, 5, 32, 64], 'conv2')
        
        # densely connected layer
        h_fc1 = fc_layer(h_pool2,[7 * 7 * 64, 1024], 'fc1', tf.nn.relu)

        # dropout
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # softmax
        y_conv = fc_layer(h_fc1_drop,[1024, 10], 'fc2', tf.nn.softmax)

        # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

        # setup training
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]), name="loss_function")
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)), tf.float32, name="correct_prediction")
        accuracy = tf.reduce_mean(correct_prediction, name="accuracy") 

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar("cross_entropy", cross_entropy)
        tf.summary.scalar("val_accuracy", accuracy)
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

        # Run the Op to initialize the variables.
        sess.run(init)

        # run the training
        for i in range(max_step):
            batch = mnist.train.next_batch(batch_size) # make the data batch, which is used in the training iteration.
                                                # the batch size is 50
            if i%100 == 0:
                # output the training accuracy every 100 iterations
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_:batch[1], keep_prob: 1.0})
                
                test_accuracy = accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                
                print("step %d, training accuracy %g"%(i, train_accuracy),
                      "test accuracy %g"%test_accuracy)
                
                # Update the events file which is used to monitor the training (in this case,
                # only the training loss is monitored)
                summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()

            # save the checkpoints every 1100 iterations
            if i % 1100 == 0 or i == max_step:
                checkpoint_file = os.path.join(result_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=i)

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # run one train_step

            
        # print test error
        test_accuracy = accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

    stop_time = time.time()
    print('The training takes %f second to finish'%(stop_time - start_time))

if __name__ == "__main__":
    main()

