#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Trains a Convolutional Neural Network Classifier to predict the class of
MNIST images. The model paramters are saved to model/model_params.npz.

Adapted from https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py
"""
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import cnn_model

def train_model():
    """ Trains a CNN model for MNIST digit classification and saves the model
    parameters to  model/model_params.npz."""
    X_train, y_train, X_val, y_val, X_test, y_test = \
                        tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    sess = tf.InteractiveSession()

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 128

    # [batch_size, height, width, channels]
    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
    y_ = tf.placeholder(tf.int64, shape=[batch_size,])

    network = cnn_model.cnn_model_graph(x)
    y = network.outputs

    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = ce

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.global_variables_initializer())
    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)        # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop)    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop)    # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err; val_acc += ac; n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))

    #Save model parameters
    tl.files.exists_or_mkdir('model')
    tl.files.save_npz(network.all_params, name="model/model_params.npz")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
                                X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(network.all_drop)    # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err; test_acc += ac; n_batch += 1
    print("   test loss: %f" % (test_loss/n_batch))
    print("   test acc: %f" % (test_acc/n_batch))

if __name__ == '__main__':
    train_model()
