# -*- coding: utf-8 -*-
""" Convolutional Neural Network model definition for Classifying MNIST digits,
in TensorLayer.

cnn_model_graph(input_node) returns the network model.

Adapted from https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py
"""
import tensorflow as tf
import tensorlayer as tl

def cnn_model_graph(input_node):
    """Defines a CNN model for classifying MNIST digits

    Arguments:
        input_node : Tensorflow placeholder with shape [batch_size, 28, 28 ,1]

    Returns:
        TensorLayer layer representing the tf graph
    """
    network = tl.layers.InputLayer(input_node, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name ='cnn_layer1')     # output: (?, 28, 28, 32)
    network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer1',)   # output: (?, 14, 14, 32)
    network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name ='cnn_layer2')     # output: (?, 14, 14, 64)
    network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer2',)   # output: (?, 7, 7, 64)
    network = tl.layers.FlattenLayer(network, name='flatten_layer')   # output: (?, 3136)
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1') # output: (?, 3136)
    network = tl.layers.DenseLayer(network, n_units=256,
                                    act = tf.nn.relu, name='relu1')   # output: (?, 256)
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2') # output: (?, 256)
    network = tl.layers.DenseLayer(network, n_units=10,
                                    act = tf.identity,
                                    name='output_layer')    # output: (?, 10)
    return network
