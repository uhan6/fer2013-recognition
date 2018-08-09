import tensorflow as tf

import fer2013_input

# 参数
EPOCH_NUM = 100  # training iter = epoch * batch
BATCH_SIZE = 128

DISPLAY_STEP = 10
LEARNING_RATE = 0.001

N_INPUT = fer2013_input.IMG_SIZE ** 2  # 2304
N_CLASSES = fer2013_input.CLASSES_NUM
DROPOUT_RATE = 0.75  # Dropout 保留单元的概率

# 占位符定义
X = tf.placeholder(tf.float32, [None, N_INPUT])
Y_CORRECT = tf.placeholder(tf.float32, [None, N_CLASSES])
KEEP_RATE = tf.placeholder(tf.float32)  # dropout rate 保留的概率

TRAIN_DATA_SRC = fer2013_input.TRAIN_DATA_SRC
TEST_DATA_SRC = fer2013_input.TEST_DATA_SRC
MODEL_SAVE_PATH = "../drive/machine_learning/fer2013/model/tf_simple_net/"


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

    'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
}

BIASES = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

PREDICT = conv_net(X, WEIGHTS, BIASES, KEEP_RATE)

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


if __name__ == '__main__':
    train()
