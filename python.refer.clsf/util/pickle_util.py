# coding:utf-8
import pickle


def dump_object(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'), protocol=4)


def load_object(pklfile):
    return pickle.load(open(pklfile, 'rb'))
