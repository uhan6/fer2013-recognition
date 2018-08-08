# coding:utf-8

import tensorflow as tf

import fer2013_input
import fer2013

save_path = "drive/machine_learning/fer2013/model/"
module_file = tf.train.latest_checkpoint(save_path)
saver = tf.train.Saver()


def eval_data():
    with tf.Session() as sess:
        saver.restore(sess, module_file)

        accuracy = fer2013.accuracy
        y = fer2013.y
        x = fer2013.x
        keep_prob = fer2013.keep_prob

        input_test = fer2013_input.Eval()
        test_x, test_y = input_test.get_data()
        print("测试准确率:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.}))


if __name__ == '__main__':
    eval_data()
