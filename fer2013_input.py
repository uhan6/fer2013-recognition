# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# 处理这个尺寸的图像
# 如果更改此数字，则整个模型架构将更改，并且任何模型都需要重新训练
IMG_SIZE = 48
# 数据集标签和描述
CLASSES_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
# 数据类别数
CLASSES_NUM = 7
# 样本数量
TRAIN_EXAMPLES_NUM = 28709
TEST_EXAMPLES_NUM = 3589
# 转化为 one hot 标签
ONE_HOT = False

# 测试集, 训练集存储路径
__TRAIN_DATA_SRC_LOCAL = "../drive/machine_learning/fer2013/data/fer2013.csv"
__TEST_DATA_SRC_LOCAL = "../drive/machine_learning/fer2013/data/fer2013.csv"

# 在 google drive 上存储的路径
# __TRAIN_DATA_SRC_DRIVE = "drive/machine_learning/fer2013/data/fer2013.csv"
# __EVAL_DATA_SRC_DRIVE = "drive/machine_learning/fer2013/data/fer2013.csv"

# 设置在google drive 上运行
TRAIN_DATA_SRC, TEST_DATA_SRC = __TRAIN_DATA_SRC_LOCAL, __TEST_DATA_SRC_LOCAL


def load_data(src, mode, train_index=TRAIN_EXAMPLES_NUM, img_size=IMG_SIZE, one_hot=ONE_HOT, classes=CLASSES_NUM):
    """
    读取整合数据
    训练和测试数据不重复，数据内部不重复
    """

    images = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.float32)

    # 有第一行描述行，跳过 header
    data = pd.read_csv(src, header=0)

    # 0~28708  28709个训练 28709~35886  3589个测试
    if mode == "train":
        data_train = data[:train_index]
        labels = data_train["emotion"].values
        images = data_train["pixels"].str.split(" ").values
        images = np.concatenate(images)
    elif mode == "test":
        data_test = data[train_index:]
        labels = data_test["emotion"].values
        images = data_test["pixels"].str.split(" ").values
        images = np.concatenate(images)

    images = np.reshape(images, (-1, img_size ** 2))

    if one_hot:
        # 转换为 one hot 标签
        labels = pd.get_dummies(labels)
        labels = labels.values
        labels = np.reshape(labels, (-1, classes))

    print("data shape:{}".format(images.shape))
    print("label shape:{}".format(labels.shape))

    return images, labels


class Train:
    """训练"""
    _images = np.array([], dtype=np.float32)
    _labels = np.array([], dtype=np.float32)
    _batch_index = 0

    def __init__(self, src=TRAIN_DATA_SRC,
                 mode="train",
                 train_index=TRAIN_EXAMPLES_NUM,
                 img_size=IMG_SIZE,
                 one_hot=ONE_HOT,
                 classes=CLASSES_NUM):
        self._images, self._labels = load_data(src, mode, train_index, img_size, one_hot, classes)

    def next_batch_old(self, batch_size=128):
        """随机获得一批数据"""
        random_index = np.random.choice(self._images.shape[0], size=batch_size)
        rand_x = self._images[random_index]
        rand_y = self._labels[random_index]
        return rand_x, rand_y

    def next_batch(self, batch_size):
        """随机获得一批数据"""
        max_data = self._images.shape[0]

        if self._batch_index * batch_size > max_data:
            self._batch_index = 0

        if self._batch_index == 0:
            indices = np.arange(max_data)
            np.random.shuffle(indices)

        start_idx = self._batch_index * batch_size
        rand_x = self._images[start_idx:start_idx + batch_size]
        rand_y = self._labels[start_idx:start_idx + batch_size]

        return rand_x, rand_y


class Eval:
    """验证"""
    _images = np.array([], dtype=np.float32)
    _labels = np.array([], dtype=np.float32)

    def __init__(self, src=TRAIN_DATA_SRC,
                 mode="test",
                 train_index=TRAIN_EXAMPLES_NUM,
                 img_size=IMG_SIZE,
                 one_hot=ONE_HOT,
                 classes=CLASSES_NUM):
        self._images, self._labels = load_data(src, mode, train_index, img_size, one_hot, classes)

    def get_data(self):
        """获取数据"""
        return self._images, self._labels
