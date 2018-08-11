# coding:utf-8

import tensorflow as tf

from data_input import fer2013_input
from tensorflow_simple_model import tf_conv3_model as model

# 参数
EPOCH_NUM = 100  # training iter = epoch * batch
BATCH_SIZE = 128

DISPLAY_STEP = 10
LEARNING_RATE = 0.001

N_INPUT = fer2013_input.IMG_SIZE ** 2  # 2304
CLASSES_NUM = fer2013_input.CLASSES_NUM
DROPOUT_RATE = 0.75  # Dropout 保留单元的概率

# 路径
TRAIN_DATA_SRC = fer2013_input.TRAIN_DATA_SRC
TEST_DATA_SRC = fer2013_input.TEST_DATA_SRC
MODEL_SAVE_PATH = "drive/machine_learning/fer2013/model/tf_simple_net/"


# 占位符定义
X = tf.placeholder(tf.float32, [None, N_INPUT])
Y_CORRECT = tf.placeholder(tf.float32, [None, CLASSES_NUM])
KEEP_RATE = tf.placeholder(tf.float32)  # dropout rate 保留的概率

# 定义参数
PREDICT = model.conv_net(X, model.WEIGHTS, model.BIASES, KEEP_RATE)

# 定义损失函数和训练函数
COST = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=PREDICT, labels=Y_CORRECT))
TRAIN_OP = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(COST)

# 验证模型
CORRECT_PRED = tf.equal(tf.argmax(PREDICT, 1), tf.argmax(Y_CORRECT, 1))
ACCURACY = tf.reduce_mean(tf.cast(CORRECT_PRED, tf.float32))

# 初始化变量
INIT = tf.global_variables_initializer()

SAVER = tf.train.Saver()


def train():
    # 加载会话
    with tf.Session() as sess:
        sess.run(INIT)
        epoch = 0
        input_train = fer2013_input.Train(one_hot=True)
        while epoch < EPOCH_NUM:
            batch_xs, batch_ys = input_train.next_batch(BATCH_SIZE)
            # 训练一批次
            sess.run(TRAIN_OP, feed_dict={
                X: batch_xs, Y_CORRECT: batch_ys, KEEP_RATE: DROPOUT_RATE})
            if (epoch + 1) % DISPLAY_STEP == 0:
                # 准确率
                acc = sess.run(ACCURACY, feed_dict={X: batch_xs, Y_CORRECT: batch_ys, KEEP_RATE: 1.})

                loss = sess.run(COST, feed_dict={X: batch_xs, Y_CORRECT: batch_ys, KEEP_RATE: 1.})
                print("Step " + str(epoch) +
                      ", Iter " + str(epoch * BATCH_SIZE) +
                      ", Minibatch Loss= " + "{:.6f}".format(loss) +
                      ", Training Accuracy= " + "{:.5f}".format(acc))
            epoch += 1

        # 模型保存目录
        save_path = MODEL_SAVE_PATH

        # 创建目录
        if tf.gfile.Exists(save_path):
            tf.gfile.DeleteRecursively(save_path)
        tf.gfile.MakeDirs(save_path)

        SAVER.save(sess, save_path, epoch)
        print("训练结束")


def eval_data():
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        SAVER.restore(sess, module_file)

        input_test = fer2013_input.Eval(one_hot=True)
        test_x, test_y = input_test.get_data()
        print("测试准确率:", sess.run(ACCURACY, feed_dict={X: test_x, Y_CORRECT: test_y, KEEP_RATE: 1.}))
