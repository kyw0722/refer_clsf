# coding:utf-8
import os
from tensorflow import keras

from util import path_util
from util import csv_util
from util import pickle_util
from util.config import config


class DatasetMaker:

    def __init__(self, pjt_code, pdt_code=None):
        self.pjt_code = pjt_code
        self.pdt_code = pdt_code

    def makeValidDataset(self, model_type: str, docu_feature: int, model: object):
        root_path = path_util.get_project_root_path()
        valid_csv_path = os.path.join(root_path, f'dataset/{self.pjt_code}/intermediate_csv/valid.csv')
        clean_doc_path = os.path.join(root_path, f'dataset/{self.pjt_code}/intermediate_csv/03.csv')
        tokenizer_path = os.path.join(root_path, f'dataset/{self.pjt_code}/final_pkl/tokenizer_{docu_feature}.pkl')

        valid_list = csv_util.read_csv_to_list(csv_file=valid_csv_path,
                                               field_idx_list=[0, 1, 2, 3],     # appl_no, ctry, ptype, label_idx
                                               fline_skip=True)
        clean_doc_list = csv_util.read_csv_to_list(csv_file=clean_doc_path,
                                                   field_idx_list=[0, 1, 2, 3, 4, 5, 6],    # appl_no, ctry, ptype, ti, ab, clr, cla
                                                   fline_skip=True)
        tokenizer = pickle_util.load_object(tokenizer_path)


        clean_doc_dict = {}     # 전체 데이터
        for clean_doc in clean_doc_list:
            appl_no, ctry, ptype, ti, ab, clr, cla = clean_doc
            key = f'{appl_no}_{ctry}_{ptype}'
            value = [ti, ab, clr, cla]
            clean_doc_dict[key] = value


        text_total_list = []
        for valid in valid_list:
            appl_no, ctry, ptype, label_idx = valid
            key = f'{appl_no}_{ctry}_{ptype}'
            text_list = clean_doc_dict[key]
            text_total_list.append(text_list)

        docu_list = None
        if docu_feature == 0:
            docu_list= [f'{x[0]} {x[1]}' for x in text_total_list]
        elif docu_feature == 1:
            docu_list= [f'{x[0]} {x[1]} {x[2]}' for x in text_total_list]
        elif docu_feature == 2:
            docu_list = [f'{x[0]} {x[1]} {x[3]}' for x in text_total_list]


        if model_type == 'mlp_singlex':
            x_valid = tokenizer.texts_to_matrix(docu_list, mode='freq')
        else:
            doc_seq = tokenizer.texts_to_sequences(docu_list)
            max_doc_length = model.input[0].shape[-1]
            x_valid = keras.preprocessing.sequence.pad_sequences(doc_seq, maxlen=max_doc_length, padding='post')

        return x_valid


    def makePredictDataset(self, model_type: str, docu_feature: int, model: object):
        root_path = path_util.get_project_root_path()
        clean_doc_path = os.path.join(root_path, f'dataset/{self.pjt_code}/predict/{self.pdt_code}/intermediate_csv/03.csv')
        tokenizer_path = os.path.join(root_path, f'dataset/{self.pjt_code}/final_pkl/tokenizer_{docu_feature}.pkl')

        clean_doc_list = csv_util.read_csv_to_list(csv_file=clean_doc_path,
                                                   field_idx_list=[3, 4, 5, 6],
                                                   # ti, ab, clr, cla
                                                   fline_skip=True)
        tokenizer = pickle_util.load_object(tokenizer_path)

        if docu_feature == 0:
            docu_list = [f'{x[0]} {x[1]}' for x in clean_doc_list]
        elif docu_feature == 1:
            docu_list = [f'{x[0]} {x[1]} {x[2]}' for x in clean_doc_list]
        elif docu_feature == 2:
            docu_list = [f'{x[0]} {x[1]} {x[3]}' for x in clean_doc_list]

        if model_type == 'mlp_singlex':
            x_predict = tokenizer.texts_to_matrix(docu_list, mode='freq')
        else:
            doc_seq = tokenizer.texts_to_sequences(docu_list)
            max_doc_length = model.input[0].shape[-1]
            x_predict = keras.preprocessing.sequence.pad_sequences(doc_seq, maxlen=max_doc_length, padding='post')

        return x_predict


