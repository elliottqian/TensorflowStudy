# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 200

INPUT_SIZE = 28
TIME_STEP_SIZE = 28
HIDDEN_SIZE = 256
LAYER_NUM = 2
CLASS_NUM = 10

KEEP_PROB = 0.5


# 设置 GPU 按需增长
session_config = tf.ConfigProto(log_device_placement=False)
session_config.gpu_options.allow_growth = True
run_config = tf.estimator.RunConfig().replace(session_config=session_config)


def build_forward_model(input_layer):

    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_drop_1 = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_1, input_keep_prob=1.0, output_keep_prob=KEEP_PROB)

    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_drop_2 = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_2, input_keep_prob=1.0, output_keep_prob=KEEP_PROB)

    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_drop_1, lstm_cell_drop_2], state_is_tuple=True)
    init_state = mlstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    outputs = list()
    state = init_state

    with tf.variable_scope('RNN'):
        for time_step in range(TIME_STEP_SIZE):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(input_layer[:, time_step, :], state)
            outputs.append(cell_output)
    h_state = outputs[-1]
    dense_layer_1 = tf.layers.dense(inputs=h_state, units=512, activation=tf.nn.relu)
    output_layer = tf.layers.dense(inputs=dense_layer_1, units=CLASS_NUM)
    return output_layer


def simple_rnn_model(features, labels, mode):
    input_X = tf.reshape(features["X"], [-1, 28, 28])
    lstm_forward_result = build_forward_model(input_X)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=lstm_forward_result)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    predictions = {
        "classes": tf.argmax(input=lstm_forward_result, axis=1),
        "probabilities": tf.nn.softmax(lstm_forward_result, name="soft_max_tensor")
    }
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                                        predictions=predictions["classes"], name="accuracy")
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    mnist = input_data.read_data_sets("../cnn/MNIST_data/", one_hot=True)
    train_data = mnist.train.images  # Returns np.array
    train_labels = mnist.train.labels
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = mnist.test.labels

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": train_data},
                                                        y=train_labels,
                                                        batch_size=BATCH_SIZE,
                                                        num_epochs=None,
                                                        shuffle=True)

    mnist_classifier = tf.estimator.Estimator(model_fn=simple_rnn_model,
                                              model_dir="./_rnn/mnist_convnet_model2",
                                              config=run_config)

    mnist_classifier.train(input_fn=train_input_fn,
                           steps=3000,
                           hooks=[])

    eval_input_train_fn = tf.estimator.inputs.numpy_input_fn(x={"X": train_data},
                                                             y=train_labels,
                                                             batch_size=BATCH_SIZE,
                                                             num_epochs=1,
                                                             shuffle=False)
    eval_results_train = mnist_classifier.evaluate(input_fn=eval_input_train_fn)
    print(eval_results_train)

    # print(train_data.shape)
    # print(train_labels.shape)
    # print(eval_data.shape)
    # print(type(train_input_fn()))
    #
    # print(train_input_fn()[0])
    #
    # X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    # y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    #
    # input_layer = tf.reshape(X, shape=[-1, 28, 28])
    # lstm_out = build_forward_model(input_layer)
    #
    # print(lstm_out)
    #
    # with tf.Session(config=session_config) as session:
    #     input_X, y = session.run(train_input_fn())
    #     print(input_X.shape)
    #     result = session.run(lstm_out, feed_dict={X: input_X})
    #     print(result)

    pass
