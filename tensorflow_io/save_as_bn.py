# coding: utf-8

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
model_path = "C:\\Users\\ElliottQian\\PycharmProjects\\TensorflowStudy\\tensorflow_io\\bp_model12"


def simple_model(features, labels, mode):
    first_layer = features["X"]


    # label = tf.reshape(labels["y"], [-1, 1])
    # label = labels["y"]

    with tf.variable_scope("weight"):
        w = tf.Variable(initial_value=tf.random_normal([2, 1], stddev=0.35))
        b = tf.Variable(initial_value=tf.random_normal([1, 1], stddev=0.35))
    y_predict = tf.add(x=tf.matmul(first_layer, w), y=b, name="model_predictions")

    predictions = {"predictions": y_predict}

    regression_output = tf.estimator.export.RegressionOutput(value=predictions["predictions"])
    export_outputs = {"regression_output": regression_output}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=y_predict)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=2)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=[logging_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels=labels, predictions=y_predict)
        eval_metric_ops = {"eval_metric_ops_loss": tf.metrics.mean_squared_error(labels=labels, predictions=y_predict)}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)


if __name__ == "__main__":
    test_data = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6]
        ],
        dtype=np.float32
    )
    test_label = np.array([5, 8, 11, 14, 16.5], dtype=np.float32).reshape([5, 1])

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)
    simple_regression = tf.estimator.Estimator(model_fn=simple_model,
                                               model_dir="./model/simple_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": test_data},
                                                        y=test_label,
                                                        batch_size=2,
                                                        num_epochs=300,
                                                        shuffle=True)
    tensors_to_log = {"log_name": "model_predictions"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2)
    simple_regression.train(input_fn=train_input_fn,
                            steps=20,
                            hooks=[])

    eval_results_train = simple_regression.evaluate(input_fn=train_input_fn, steps=1)

    feature_spec = {
        'X': tf.placeholder(dtype=tf.float32, shape=[None, 2], name="input_x"),
        'Not_X': tf.placeholder(dtype=tf.float32, shape=[None, 5])
    }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    simple_regression.export_savedmodel(model_path, serving_input_receiver_fn)
    pass
