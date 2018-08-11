# coding:utf-8

import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from data_input import fer2013_input
from vgg_net_model import vgg_net_model as model

# 参数
EPOCH_NUM = 200  # training iter = epoch * batch
BATCH_SIZE = 128

LEARNING_RATE = 0.0001
# IMAGE_SIZE = IMG_SIZE
IMAGE_SIZE = fer2013_input.IMG_SIZE

TRAIN_DATA_SRC = fer2013_input.TRAIN_DATA_SRC
TEST_DATA_SRC = fer2013_input.TEST_DATA_SRC
MODEL_SAVE_PATH = "drive/machine_learning/fer2013/model/vgg_net/"


X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='x')
Y_CORRECT = tf.placeholder(dtype=tf.int64, shape=[None], name='y_correct')

# using local response normalization 使用 lrn
NETWORK, COST, _ = model.vgg_net_model(X, Y_CORRECT, False)
_, COST_TEST, ACCURACY = model.vgg_net_model(X, Y_CORRECT, True)
# you may want to try batch normalization 使用 bn
# NETWORK, COST, _ = vgg_net_model_bn(X, Y_CORRECT, False, is_train=True)
# _, COST_TEST, ACCURACY = vgg_net_model_bn(X, Y_CORRECT, True, is_train=False)


TRAIN_PARAMS = NETWORK.all_params
TRAIN_OP = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                  use_locking=False).minimize(COST, var_list=TRAIN_PARAMS)
# 初始化变量
INIT = tf.global_variables_initializer()

SAVER = tf.train.Saver()


def train():
    # 加载会话
    with tf.Session() as sess:
        sess.run(INIT)
        NETWORK.print_params(False)
        NETWORK.print_layers()
        print("------------------------------")
        print("learning_rate: %f" % LEARNING_RATE)
        print("batch_size: %d" % BATCH_SIZE)
        print("EPOCH_NUM: %d" % EPOCH_NUM)

        data, labels = fer2013_input.load_data(TRAIN_DATA_SRC, "train", one_hot=False)

        epoch = 0

        while epoch < EPOCH_NUM:
            start_time = time.time()
            epoch_i = 0
            for batch_xs, batch_ys in tl.iterate.minibatches(data, labels, BATCH_SIZE, shuffle=True):
                # data augmentation for training
                # batch_xs = tl.prepro.threading_data(
                # batch_xs, fn=distort_fn, is_train=True)
                batch_xs = np.reshape(batch_xs, (-1, 48, 48, 1))
                sess.run(TRAIN_OP, feed_dict={X: batch_xs, Y_CORRECT: batch_ys})

                if epoch_i == 127:
                    acc = sess.run(ACCURACY, feed_dict={X: batch_xs, Y_CORRECT: batch_ys})

                    loss = sess.run(COST, feed_dict={X: batch_xs, Y_CORRECT: batch_ys})
                    print("epoch {:d} of {:d} took {:.2fs}".format(
                        epoch + 1, EPOCH_NUM, time.time() - start_time) +
                        ", Minibatch Loss= {:.6f}".format(loss) +
                        ", Training Accuracy= {:.5f}".format(acc))
                epoch_i += 1
            epoch += 1

        print("------------------------------")
        print("train finished")

        # 模型保存目录
        save_path = MODEL_SAVE_PATH

        # 创建目录
        if tf.gfile.Exists(save_path):
            tf.gfile.DeleteRecursively(save_path)
        tf.gfile.MakeDirs(save_path)

        SAVER.save(sess, save_path, epoch)
        print("save model success")


def eval_data():
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        SAVER.restore(sess, module_file)

        test_x, test_y = fer2013_input.load_data(TEST_DATA_SRC, "test", one_hot=False)

        test_x = np.reshape(test_x, (-1, 48, 48, 1))
        print("Accuracy rate:", sess.run(ACCURACY, feed_dict={X: test_x, Y_CORRECT: test_y}))
