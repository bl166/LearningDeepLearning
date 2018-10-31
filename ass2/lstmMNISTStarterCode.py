import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np 
from time import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#call mnist function

learningRate = 1e-3
trainingIters = 20000
batchSize = 128
displayStep = 200

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 128 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

def RNN(x, weights, biases, family="rnn"):
    
    x = tf.unstack(x, nSteps, 1)
#     x = tf.transpose(x, [1,0,2])
#     x = tf.reshape(x, [-1, nInput])
#     x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

    #find which lstm to use in the documentation
    if family=="rnn":
        rnnCell = rnn.BasicRNNCell(nHidden)
    elif family=="lstm":
        rnnCell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)
    elif family=="gru":
        rnnCell = rnn.GRUCell(nHidden)
    else:
        raise NotImplemented

    outputs, states = rnn.static_rnn(rnnCell, x, dtype=tf.float32)#for the rnn where to get the output and hidden state 

    return tf.matmul(outputs[-1], weights['out'])+ biases['out']


"""
Integrating training, testing and summary processes
"""
def train_rnn(family = 'rnn'):
    tf.reset_default_graph()

    x = tf.placeholder('float', [None, nSteps, nInput])
    y = tf.placeholder('float', [None, nClasses])

    weights = {
        'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([nClasses]))
    }

    pred = RNN(x, weights, biases, family)
    prediction = tf.nn.softmax(pred)

    rnn_result_dir = './results/rnn/'+family+'_res'+str(learningRate)+'+bs'+str(batchSize)+'+ep'+str(trainingIters)
    if not os.path.exists(rnn_result_dir): os.makedirs(rnn_result_dir)

    #optimization
    #create the cost, optimization, evaluation, and accuracy
    #for the cost softmax_cross_entropy_with_logits seems really good
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    tf.summary.scalar("loss", cost)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    # load test data
    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels

    start_time = time()
    with tf.Session() as sess:

        # Run the initializer
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(rnn_result_dir+'/train', sess.graph)
        test_writer = tf.summary.FileWriter(rnn_result_dir+'/test', sess.graph)

        # Save best model and tensorboard log files
        saver = tf.train.Saver()
        checkpoint_file = rnn_result_dir+'/checkpoint'

        best_acc = 0
        save = True
        for step in range(1, trainingIters+1):

            batchX, batchY = mnist.train.next_batch(batchSize)
            batchX = batchX.reshape((batchSize, nSteps, nInput))

            if step % displayStep != 0:

                sess.run(optimizer, feed_dict={x: batchX, y: batchY})

            else:
                # Calculate batch loss and accuracy
                train_summary_str, loss_train, acc_train = \
                    sess.run([summary_op, cost, accuracy], feed_dict={x: batchX, y: batchY}) # train
                test_summary_str, loss_test, acc_test = \
                    sess.run([summary_op, cost, accuracy], feed_dict={x: testData, y: testLabel}) # test

                # add to tensorboard summary
                train_writer.add_summary(train_summary_str, step)
                train_writer.flush()

                test_writer.add_summary(test_summary_str, step)
                test_writer.flush()

                print('''Step {}, Minibatch Loss= {:.6f}, Training Accuracy= {:.5f}, Test Loss= {:.6f}, Test Accuracy= {:.5f}'''\
                      .format(step, loss_train, acc_train, loss_test, acc_test))

                # keep tracking best accuracy
                if acc_test > best_acc:
                    best_acc = acc_test
                    if save:
                        saver.save(sess, checkpoint_file, global_step=step)
                        print("Saving...")

        print('Optimization finished')
        end_time = time()

        print("Runtime: {:.2f} s".format(end_time - start_time),
              ", Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))


