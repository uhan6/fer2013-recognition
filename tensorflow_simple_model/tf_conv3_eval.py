# coding:utf-8

import tensorflow as tf

import fer2013_input
import tensorflow_simple_model.tf_conv3_model_train as fer2013

SAVE_PATH = fer2013.MODEL_SAVE_PATH
MODULE_FILE = tf.train.latest_checkpoint(SAVE_PATH)
SAVER = tf.train.Saver()


def eval_data():
    with tf.Session() as sess:
        SAVER.restore(sess, MODULE_FILE)

        accuracy = fer2013.ACCURACY
        y = fer2013.Y_CORRECT
        x = fer2013.X
        keep_prob = fer2013.KEEP_RATE

        input_test = fer2013_input.Eval(one_hot=True)
        test_x, test_y = input_test.get_data()
        print("测试准确率:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.}))


if __name__ == '__main__':
    eval_data()
