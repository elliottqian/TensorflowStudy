# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(type(mnist))


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    """
    权重参数
    :param shape: 数据的shape
    :return:
    """
    initial = tf.truncated_normal(shape, stddev=0.1)  # 方差唯一的标准正态分布?
    return tf.Variable(initial)


def bias_variable(shape):
    """
    偏置参数
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)   # 为0.1的常量为偏置量
    return tf.Variable(initial)


def conv2d(x, W):
    """
    二维卷积
    :param x:
    :param W:
    :return:

    第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
    具有[batch, in_height, in_width, in_channels]这样的shape，
    具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
    注意这是一个4维的Tensor，要求类型为float32和float64其中之一

    第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
    具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
    具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
    有一个地方需要注意，第三维in_channels，就是参数input的第四维
    """
    # strides[0]=strides[3]=1.   第二和第三个参数分别表示 横着和竖着的步长
    # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')




