# -*- coding: utf-8 -*-

from tensorflow.contrib.learn.python.learn.preprocessing import text
import codecs
import gensim
import os

# sentence = [["a", "b"], ["c", "a"]]
# dic = gensim.corpora.dictionary.Dictionary(documents=sentence, prune_at=None)
# print(dic.keys())
# print(dic.values())
# print(dic.token2id.get("a"))


def read_file_to_dict(file_path, sentence_size=10, min_word_freq=0):
    """
    # 保存: vocab_processor.save(filename)
    # 读取: text.VocabularyProcessor.restore(filename)
    # fit相当于训练, transform相当于转换
    # 具体用法见: https://github.com/tensorflow/tensorflow/tree/r1.3/tensorflow/contrib/learn/python/learn/preprocessing/tests
    # 转换一句话: vocab_processor.transform(["我 喜欢 吃饭", "b c a"]), 返回类型为numpy.ndarray的列表
    # 转换一个单词: vocab_processor.vocabulary_.get("我")
    # embedding_size = len(vocab_processor.vocabulary_), index 是从1开始的, embedding_size比单词数量多1
    :param file_path:
    :param sentence_size:
    :param min_word_freq:
    :return:
    """
    with codecs.open(file_path) as f:
        vocab_processor = text.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
        # 输入一个迭代器, 迭代器的数据是一个文本, 文本的词用空格隔开
        vocab_processor.fit(f)
    return vocab_processor


class MySentences(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with codecs.open(self.path) as f:
            for line in f:
                yield line.strip().split()

    def open_dir(self):
        for f_name in os.listdir(self.path):
            for line in codecs.open(os.path.join(self.path, f_name)):
                yield line.strip().split()


class MySentenceDir(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for f_name in os.listdir(self.path):
            for line in codecs.open(os.path.join(self.path, f_name)):
                yield line.strip().split()



class Vocab(object):
    """
    构造词典的类, 用gensim的gensim.corpora.dictionary.Dictionary
    """

    @staticmethod
    def get_sentences(file_path):
        with codecs.open(file_path) as f:
            for line in f:
                yield line.strip().split()

    def __init__(self):
        self.dic = None

    def build(self, sentences, prune_at=None):
        self.dic = gensim.corpora.dictionary.Dictionary(documents=sentences, prune_at=prune_at)

    def save_vocab(self, saved_path):
        self.dic.save(saved_path)

    def load(self, saved_path):
        self.dic = gensim.corpora.dictionary.Dictionary.load(saved_path)

    def get_key(self, item):
        return self.dic.token2id.get(item)

    def get_item(self, key):
        return self.dic.get(key)


class WordToVec(object):
    """

    """

    def __init__(self):
        """
        :type model: gensim.models.Word2Vec
        """
        self.model = None

    def build(self, sentences):
        """
        :type self.model: gensim.models.Word2Vec
        :param sentences:
        :return:
        """

        self.model = gensim.models.Word2Vec(sentences, size=100, iter=14, window=5, min_count=0, workers=8)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        """
        :type path: gensim.models.Word2Vec
        :param path:
        :return:
        """

        self.model = gensim.models.Word2Vec.load(path)


if __name__ == "__main__":
    ttbb_path = "/mnt/D/Ubuntu/PycharmProjects/nlp_study_py/nlpTools/resource/tlbb"
    saved_file = "/home/elliottqian/Documents/PycharmProjects/deeplearning_notebook/utils/tlbb_vocab"
    saved_file2 = "/home/elliottqian/Documents/PycharmProjects/deeplearning_notebook/utils/tlbb_wor2vec"

    my_sentences = MySentences(ttbb_path)
    w2v = WordToVec()
    w2v.build(my_sentences)
    w2v.save(saved_file2)
    w2v.load(saved_file2)
    print(w2v.model.most_similar("乔峰"))

    # vocab = Vocab()
    # # sentences = Vocab.get_sentences(ttbb_path)
    # # vocab.build(sentences)
    # # vocab.save_vocab(saved_file)
    # vocab.load(saved_file)
    # print(vocab.get_key("乔峰"))
    # # read_file_to_sentence("/mnt/D/Ubuntu/PycharmProjects/nlp_study_py/nlpTools/resource/tlbb")
    pass