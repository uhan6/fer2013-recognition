# coding:utf-8

import tensorflow as tf
import numpy as np

import fer2013_input
import vgg_net_model.vgg_net_model_train as fer2013

TEST_DATA_SRC = fer2013.TEST_DATA_SRC
SAVE_PATH = fer2013.MODEL_SAVE_PATH
MODULE_FILE = tf.train.latest_checkpoint(SAVE_PATH)
SAVER = tf.train.Saver()


def eval_data():
    with tf.Session() as sess:
        SAVER.restore(sess, MODULE_FILE)

        accuracy = fer2013.ACCURACY
        y_correct = fer2013.Y_CORRECT
        x = fer2013.X

        test_x, test_y = fer2013_input.load_data(TEST_DATA_SRC, "test", one_hot=False)

        test_x = np.reshape(test_x, (-1, 48, 48, 1))
        print("Accuracy rate:", sess.run(accuracy, feed_dict={x: test_x, y_correct: test_y}))


if __name__ == '__main__':
    eval_data()
