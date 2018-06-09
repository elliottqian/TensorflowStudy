# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class SimpleCnn(object):

    def __init__(self):
        self.X = None
        self.y = None
        self.filter_one = None
        pass

    def build_net(self):

        pass

    def build_varible(self):
        with tf.variable_scope("input_data"):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        with tf.variable_scope("parameter"):
            self.filter_one = tf.Variable(initial_value=tf.random_uniform([3, 3, 1]))
            pass

    def build_cov(self):
        with tf.variable_scope("cov"):
            tf.nn.conv2d(self.X, self.filter_one, strides=[1, 1, 1, 1], padding='SAME')

    def build_loss(self):
        pass

    def train(self):
        pass

    def save_model(self):
        pass


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# print(type(mnist))
#
#
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
#
# def weight_variable(shape):
#     """
#     权重参数
#     :param shape: 数据的shape
#     :return:
#     """
#     initial = tf.truncated_normal(shape, stddev=0.1)  # 方差唯一的标准正态分布?
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     """
#     偏置参数
#     :param shape:
#     :return:
#     """
#     initial = tf.constant(0.1, shape=shape)   # 为0.1的常量为偏置量
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     """
#     二维卷积
#     :param x:
#     :param W:
#     :return:
#
#     第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
#     具有[batch, in_height, in_width, in_channels]这样的shape，
#     具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
#     注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#
#     第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
#     具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
#     具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
#     有一个地方需要注意，第三维in_channels，就是参数input的第四维
#
#     4、For the SAME padding, the output height and width are computed as:
#     out_height = ceil(float(in_height) / float(strides[1]))
#     out_width = ceil(float(in_width) / float(strides[2]))
#     For the VALID padding, the output height and width are computed as:
#     out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
#     out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
#     """
#     # strides[0]=strides[3]=1.   第二和第三个参数分别表示 横着和竖着的步长
#     # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     """
#
#     :param x:
#     :return:
#     """
#     return tf.nn.max_pool(x,
#                           ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')
#
#
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#
#
# # -1 表示缺省, 会根据其他三个参数填充   这里是 未知个数  每个数据是 28长  28宽  1个高
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
#
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   for i in range(2000):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#       train_accuracy = accuracy.eval(feed_dict={
#           x: batch[0], y_: batch[1], keep_prob: 1.0})
#       print('step %d, training accuracy %g' % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
#   print('test accuracy %g' % accuracy.eval(feed_dict={
#       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#
#
#
# class MNIST(object):
#
#     def __init__(self):
#         pass
#
#     def load_data(self):
#         self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#         self.x = tf.placeholder(tf.float32, shape=[None, 784])
#         y_ = tf.placeholder(tf.float32, shape=[None, 10])
#         pass
#
#     def get_bias_variable(self, shape):
#         """
#         偏置参数
#         :param shape:
#         :return:
#         """
#         initial = tf.constant(0.1, shape=shape)  # 为0.1的常量为偏置量
#         return tf.Variable(initial)
#
#     def get_conv2d(self, input_, convolution_kernel):
#         """
#         二维卷积
#         :param x:
#         :param W:
#         :return:
#
#         第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
#         具有[batch, in_height, in_width, in_channels]这样的shape，
#         具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
#         注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#
#         第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
#         具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
#         具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
#         有一个地方需要注意，第三维in_channels，就是参数input的第四维
#
#         4、For the SAME padding, the output height and width are computed as:
#         out_height = ceil(float(in_height) / float(strides[1]))
#         out_width = ceil(float(in_width) / float(strides[2]))
#         For the VALID padding, the output height and width are computed as:
#         out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
#         out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
#         """
#         # strides[0]=strides[3]=1.   第二和第三个参数分别表示 横着和竖着的步长
#         # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
#         return tf.nn.conv2d(input_, convolution_kernel, strides=[1, 1, 1, 1], padding='SAME')
#
#     def a(self):
#         pass




