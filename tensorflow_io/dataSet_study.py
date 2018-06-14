# coding: utf-8


import tensorflow as tf

"""
read data from text File
"""
file = tf.data.TextLineDataset("C:\\Users\\ElliottQian\\Downloads\\ml-latest-small\\movies.csv")\
    .map(lambda x: tf.decode_csv(x, record_defaults=["string", "string", "string"]))
dataSet = file.batch(20)
dataSet = dataSet.repeat(2)


def split_fun(x):
    temp = x.split(",")
    tf.decode_raw()
    dict_ = dict()
    dict_["movieId"] = temp[3]
    dict_["title"] = temp[2]
    dict_["genres1"] = temp[1]
    return dict_


iterator = dataSet.make_one_shot_iterator()


def parse_csv(line, cols_types, split_symbol):
    columns = tf.decode_csv(line, record_defaults=cols_types, field_delim=split_symbol)
    return tf.stack(columns)


def mk_dlt(csv_filename, header_lines, split_symbol, batch_size, cols_type):
    """
    :param csv_filename:
    :param header_lines: 跳过的headline的行数
    :param split_symbol:        分隔符
    :param batch_size:
    :param cols_type: 列的类型
    :return:
    """
    data_set = tf.data.TextLineDataset(filenames=csv_filename).skip(header_lines)
    data_set = data_set.map(lambda line: parse_csv(line, cols_types, split_symbol)).batch(batch_size)
    return data_set

