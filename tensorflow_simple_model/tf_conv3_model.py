# coding:utf-8

import tensorflow as tf

from data_input import fer2013_input

CLASSES_NUM = fer2013_input.CLASSES_NUM


def conv2d(img, w, b):
    """conv2d"""
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    """max_pool"""
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 48, 48, 1])

    # 卷积层
    # 最大池化层
    # 保留层
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = max_pool(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, dropout)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = max_pool(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, dropout)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = max_pool(conv3, k=2)
    conv3 = tf.nn.dropout(conv3, dropout)

    # 全连接层
    dense1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(
        tf.add(tf.matmul(dense1, weights['wd1']), biases['bd1']))
    dense1 = tf.nn.dropout(dense1, dropout)  # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, weights['out']), biases['out'])
    return out


# 参数
WEIGHTS = {
    # 5x5 卷积 1输入 32输出
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 256])),
    # 全连接 12*12*64输入 1024输出
    'wd1': tf.Variable(tf.random_normal([12 * 12 * 64, 1024])),

    'out': tf.Variable(tf.random_normal([1024, CLASSES_NUM]))
}

BIASES = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([CLASSES_NUM]))
}
