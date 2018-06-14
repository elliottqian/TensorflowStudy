# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow_io.save_as_bn import model_path


with tf.Session() as session:
    meta_graph_def = tf.saved_model.loader.load(session, ['serve'], model_path + "\\1528930715")
    x = session.graph.get_operations()
    model_predictions = session.graph.get_tensor_by_name('model_predictions:0')
    #sig = meta_graph_def.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    sig = meta_graph_def.signature_def["regression_output"]
    print(sig)
    # input_x = session.graph.get_tensor_by_name('Placeholder:0')
    # print(input_x.shape)
    #
    # test_data = np.array(
    #     [
    #         [1, 2],
    #         [2, 3],
    #         [3, 4],
    #         [4, 5],
    #         [5, 6]
    #     ],
    #     dtype=np.float32
    # )
    #
    # print(session.run(model_predictions, feed_dict={input_x: test_data}))
    for o in x:
        print(o)