#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import re

def variable_summary(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        name_var = re.split('/', var.name)[-1]
        name_var = re.split(':', name_var)[0]
        with tf.name_scope(name_var):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    return

def keras_layer_summary(layer):
    for var in layer.trainable_weights:
        variable_summary(var)
    return