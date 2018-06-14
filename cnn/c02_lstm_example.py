# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

tf.logging.set_verbosity(tf.logging.INFO)

def simple_lstm_model(features, labels, mode):
    """
     定长LSTM
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    batch_size = features["X"].shape.dims[0].value

    weight_out = tf.Variable(tf.random_normal([10, 10]))
    biases_out = tf.Variable(tf.constant(0.1, shape=(10,)))

    input_layer = tf.reshape(features["X"], shape=[-1, 28, 28])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=10, forget_bias=1.0, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_layer, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weight_out) + biases_out

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=results)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=results, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        "probabilities": tf.nn.softmax(results, name="softmax_tensor_1")
    }

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions["classes"])
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    pass


if __name__ == "__main__":
    print("build_net")

    mnist = input_data.read_data_sets("../cnn/MNIST_data/", one_hot=True)
    train_data = mnist.train.images  # Returns np.array
    print(type(train_data))
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    print(type(train_labels))
    print(train_labels[0:2])
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=simple_lstm_model, model_dir="./lstm/mnist_convnet_model")

    tensors_to_log = {"probabilities": "softmax_tensor_1"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=None)
    # eval_input_train_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"X": train_data},
    #     y=train_labels,
    #     num_epochs=1,
    #     shuffle=False)
    eval_input_test_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    # eval_results_train = mnist_classifier.evaluate(input_fn=eval_input_train_fn)
    eval_results_test = mnist_classifier.evaluate(input_fn=eval_input_test_fn)
    # print(eval_results_train)
    print(eval_results_test)
    pass
