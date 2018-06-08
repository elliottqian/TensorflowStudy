# coding: utf-8

import tensorflow as tf
import numpy as np

with tf.name_scope("data"):
    x_data = np.random.rand(100).astype(np.float32)
    y_data = 0.3 * x_data + np.random.random() * 0.05 + 0.1


with tf.name_scope("parameters"):
    with tf.variable_scope("weight_and_bias"):
        weight = tf.get_variable(name="weight", initializer=tf.random_uniform([1], minval=-1, maxval=1))
        tf.summary.histogram('weight', weight)
        bias = tf.get_variable(name="bias", initializer=tf.random_uniform([1], minval=-1, maxval=1))
        tf.summary.histogram('bias', bias)


with tf.name_scope("prediction"):
    y_prediction = tf.add((weight * x_data), bias)


with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y_prediction - y_data))
    tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

with tf.name_scope("init"):
    init = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("linear_regression/", session.graph)
    session.run(init)

    for step in range(1001):
        session.run(train)
        rs = session.run(merged)
        writer.add_summary(rs, step)
        if step % 10 == 0:
            print(step, 'weight:', session.run(weight), 'bias:', session.run(bias))


