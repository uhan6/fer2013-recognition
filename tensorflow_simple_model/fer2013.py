"""
fer2013模型
"""
import tensorflow as tf

import fer2013_input

# 参数
LEARNING_RATE = 0.001
TRAINING_STEP = 100  # training iter = step * batch
BATCH_SIZE = 128
DISPLAY_STEP = 10

# 神经网络参数
N_INPUT = fer2013_input.IMAGE_SIZE ** 2  # 2304
N_CLASSES = fer2013_input.NUM_CLASSES
DROPOUT_RATE = 0.75  # Dropout 保留单元的概率

# 占位符定义
x = tf.placeholder(tf.float32, [None, N_INPUT])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)  # dropout rate 保留的概率


def conv2d(img, w, b):
    """conv2d"""
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    """max_pool"""
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(_x, _weights, _biases, _dropout):
    _x = tf.reshape(_x, shape=[-1, 48, 48, 1])

    # 卷积层
    # 最大池化层
    # 保留层
    conv1 = conv2d(_x, _weights['wc1'], _biases['bc1'])
    conv1 = max_pool(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, _dropout)

    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pool(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, _dropout)

    conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
    conv3 = max_pool(conv3, k=2)
    conv3 = tf.nn.dropout(conv3, _dropout)

    # 全连接层
    dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(
        tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out


# 参数
weights = {
    # 5x5 卷积 1输入 32输出
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 256])),
    # 全连接 12*12*64输入 1024输出
    'wd1': tf.Variable(tf.random_normal([12 * 12 * 64, 1024])),

    'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

pred = conv_net(x, weights, biases, keep_prob)

# 定义损失函数和训练函数
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# 验证模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def train():
    # 加载会话
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        input_train = fer2013_input.Train()
        # while step * BATCH_SIZE < TRAINING_ITERS:
        while step < TRAINING_STEP:
            batch_xs, batch_ys = input_train.next_batch(BATCH_SIZE)
            # 训练一批次
            sess.run(optimizer, feed_dict={
                x: batch_xs, y: batch_ys, keep_prob: DROPOUT_RATE})
            if (step + 1) % DISPLAY_STEP == 0:
                # 准确率
                acc = sess.run(accuracy, feed_dict={
                    x: batch_xs, y: batch_ys, keep_prob: 1.})

                loss = sess.run(cost, feed_dict={
                    x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Step " + str(step) + ", Iter " + str(step * BATCH_SIZE) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

        # 模型保存目录
        save_path = "/drive/machine_learning/fer2013/model/"

        # 创建目录
        if tf.gfile.Exists(save_path):
            tf.gfile.DeleteRecursively(save_path)
        tf.gfile.MakeDirs(save_path)

        saver.save(sess, save_path, step)
        print("训练结束")


if __name__ == '__main__':
    train()
