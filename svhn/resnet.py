#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from scipy.io import loadmat
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

SVHN_LOCATION = "/mnt/Data/SVHN/"
BATCH_SIZE = 32
EPOCHS = 20

def preprocessing(img, lbl):
    gray_img = tf.image.rgb_to_grayscale(img)
    norm_img = tf.image.per_image_standardization(gray_img)

    tens_location = tf.equal(lbl, 10 * tf.ones(lbl.shape, dtype=tf.uint8))
    tens_with_zeroes = tf.where(tens_location, tf.zeros(lbl.shape, dtype=tf.uint8),
                               lbl)

    return gray_img, tens_with_zeroes

def train_input_fn(train_mat, extra_mat):
    train_dataset = tf.data.Dataset.from_tensor_slices((np.moveaxis(train_mat['X'],
        3, 0), train_mat['y'].flatten()))
    extra_dataset = tf.data.Dataset.from_tensor_slices((np.moveaxis(extra_mat['X'],
        3, 0), extra_mat['y'].flatten()))

    train_dataset = train_dataset.concatenate(extra_dataset).map(
        preprocessing).shuffle(10000).batch(
        BATCH_SIZE).repeat(EPOCHS - 1)

    return train_dataset

test_input_fn = lambda test_mat: tf.data.Dataset.from_tensor_slices((np.moveaxis(
    test_mat['X'], 3, 0), test_mat['y'].flatten())).map(preprocessing).batch(
    BATCH_SIZE)

def resnet_layer(inputs, filters=16, kernel_size=3, strides=1, activation=tf.nn.relu,
                 batch_normalization=True):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                         strides=strides, padding='same')
    if batch_normalization:
        x = tf.layers.batch_normalization(x)
    if activation is not None:
        x = activation(x)
    return x

def resnet_model(features, labels, mode):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    filters = 16
    num_res_blocks = 9

    features = tf.cast(features, dtype=tf.float16)
    x = resnet_layer(inputs=features)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, filters=filters, strides=strides)
            y = resnet_layer(inputs=y, filters=filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, filters=filters, kernel_size=1,
                                 strides=strides, activation=None,
                                 batch_normalization=False)
            x = tf.nn.relu(x + y)
        filters *= 2

    x = tf.layers.average_pooling2d(x, pool_size=8, strides=8)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=10)

    classes = tf.argmax(input=x, axis=1)
    correct_prediction = tf.equal(tf.cast(classes, tf.uint8), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
    acc = tf.identity(acc, name='accuracy_tensor')

    predictions = {'classes': classes, 'accuracy': acc}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), 10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=x)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels,
                                                       predictions=classes)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)

print("Loading training data...")
train_mat = loadmat(SVHN_LOCATION + "train_32x32.mat")
extra_mat = loadmat(SVHN_LOCATION + "extra_32x32.mat")
print("Loading testing data...")
test_mat = loadmat(SVHN_LOCATION + "test_32x32.mat")
print("All data loaded")

tensors_to_log = {'accuracy': 'accuracy_tensor'}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

model = tf.estimator.Estimator(model_fn=resnet_model, model_dir='./')

model.train(input_fn=lambda:train_input_fn(train_mat, extra_mat), hooks=[logging_hook])

print(model.evaluate(input_fn=lambda:test_input_fn(test_mat)))

