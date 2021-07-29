""" Utility functions. """
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = tf.layers.batch_normalization(conv_output, reuse=reuse, name=scope)
    if activation is not None:
        normed = activation(normed)
    return normed

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.k_shot