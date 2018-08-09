# 表情识别
使用各种模型解决 fer2013 数据集的表情识别问题

[![license](https://img.shields.io/github/license/go88/fer2013-recognition.svg?style=for-the-badge)](https://choosealicense.com/licenses/mit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](https://github.com/go88/fer2013-recognition/pulls)
[![GitHub (pre-)release](https://img.shields.io/github/release/go88/fer2013-recognition/all.svg?style=for-the-badge)](https://github.com/go88/fer2013-recognition/releases)

# 目录
```
/drive/machine_learning/fer2013/ 存放模型和数据
/*_model/
         *_eval.py   验证模型
         *_model_train.py    模型和训练
/fer2013_input.py   数据读取
/images/    图片资料
```
## 1. 简单模型

TensorFlow 原生API实现3层卷积3层池化

## 2. VGGNet 模型
我在 D 模型基础上,在每一层卷积后加入 LRN 层或 BN 层使训练效果更好

![VGGNet](images/VGGNet.png)
