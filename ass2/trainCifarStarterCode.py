from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp

# --------------------------------------------------
# setup

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

# --------------------------------------------------
# Integrated layers
# --------------------------------------------------

def conv_layer(x, dim, name, activation = tf.nn.relu):
    
    with tf.name_scope(name):
        
        with tf.name_scope("weights"):
            W = weight_variable(dim)
            
        with tf.name_scope("bias"):
            b = bias_variable([dim[-1]])
            
        with tf.name_scope("activation"):
            z = conv2d(x, W) + b
            h = activation(z)
            
        with tf.name_scope("pooling"):
            a = max_pool_2x2(h)
            
    return a


def fc_layer(x, dim, name, activation=tf.nn.relu):

    with tf.name_scope(name):
        
        with tf.name_scope('input'):
            x_flat = tf.reshape(x, [-1, dim[0]])

        with tf.name_scope('weights'):
            W = weight_variable(dim)

        with tf.name_scope('bias'):
            b = bias_variable([dim[-1]])

        with tf.name_scope('activation'):
            z = tf.matmul(x_flat, W) + b
            if activation is not None:
                z = activation(z)

    return z


def dropout(x, kp, name):

    with tf.name_scope(name):

        h_fc1_drop = tf.nn.dropout(x, kp)

    return h_fc1_drop


if __name__=="__main__":
    ntrain = 1000 # per class
    ntest = 100 # per class
    nclass = 10 # number of classes
    imsize = 28
    nchannels = 1
    batchsize = 1000
    learning_rate = 1e-3

    Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
    Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
    LTrain = np.zeros((ntrain*nclass,nclass))
    LTest = np.zeros((ntest*nclass,nclass))

    itrain = -1
    itest = -1
    for iclass in range(0, nclass):
        for isample in range(0, ntrain):
            path = './CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
            im = misc.imread(path); # 28 by 28
            im = im.astype(float)/255
            itrain += 1
            Train[itrain,:,:,0] = im
            LTrain[itrain,iclass] = 1 # 1-hot lable
        for isample in range(0, ntest):
            path = './CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
            im = misc.imread(path); # 28 by 28
            im = im.astype(float)/255
            itest += 1
            Test[itest,:,:,0] = im
            LTest[itest,iclass] = 1 # 1-hot lable

    sess = tf.InteractiveSession()

    tf_data = tf.placeholder(tf.float32, [None, imsize, imsize, nchannels]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels] 
    tf_labels = tf.placeholder(tf.float32, [None, nclass]) #tf variable for labels
    keep_prob = tf.placeholder(tf.float32)

    # --------------------------------------------------
    # model
    #create your model
    """
    • Convolutional layer with kernel 5 x 5 and 32 filter maps followed by ReLU
    • Max Pooling layer subsampling 2
    • Convolutional layer with kernel 5 x 5 and 64 filter maps followed by ReLU
    • Max Pooling layer subsampling by 2
    • Fully Connected layer that has input 7*7*64 and output 1024
    • Fully Connected layer that has input 1024 and output 10 (for the classes)
    • Softmax layer (Softmax Regression + Softmax Nonlinearity)
    """

    # first convolutional layer
    h_pool1 = conv_layer(tf_data, [5, 5, nchannels, 32], 'conv1', tf.nn.relu)

    # second convolutional layer
    h_pool2 = conv_layer(h_pool1, [5, 5, 32, 64], 'conv2', tf.nn.relu)

    # densely connected layer
    h_fc1 = fc_layer(h_pool2,[7 * 7 * 64, 1024], 'fc1', tf.nn.relu)

    # dropout
    h_fc1_drop = dropout(h_fc1, keep_prob, 'drop')

    # softmax
    y_conv = fc_layer(h_fc1_drop, [1024, 10], 'fc2', tf.nn.softmax)


    # --------------------------------------------------
    # loss
    #set up the loss, optimization, evaluation, and accuracy

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]), name="loss_function")

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(tf_labels,1)), tf.float32, name="correct_prediction")

    accuracy = tf.reduce_mean(correct_prediction, name="accuracy")


    # --------------------------------------------------
    # optimization

    sess.run(tf.initialize_all_variables())
    batch_xs = np.zeros((batchsize, imsize, imsize, nchannels))#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_ys = np.zeros((batchsize, nclass))#setup as [batchsize, the how many classes] 

    nsamples = ntrain*nclass
    for i in range(500): # try a small iteration size once it works then continue
        perm = np.arange(nsamples)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
            batch_ys[j,:] = LTrain[perm[j],:]
        if i%10 == 0:
            #calculate train accuracy and print it
            train_acc = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
            train_loss = cross_entropy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
            print("train accuracy %g,"%train_acc, train_loss)

        optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

    # --------------------------------------------------
    # test

    print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))


    sess.close()
