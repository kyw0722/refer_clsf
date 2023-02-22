# coding:utf-8
"""전체 데이터를 8:2 로 나눠서 train_dataset.csv, valid_dataset.csv 를 만든다."""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from util import path_util
from util import csv_util


def main(pjt_code):

    valid_split_ratio = 0.2

    root_path = path_util.get_project_root_path()
    cleandoc_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/03.csv')

    cleandoc_list = csv_util.read_csv_to_list(csv_file=cleandoc_path,
                                              field_idx_list=[0, 1, 2, 7],
                                              fline_skip=True)

    x = []
    y = []
    for cleandoc in cleandoc_list:
        appl_no, ctry, ptype, label_idx = cleandoc
        x.append([appl_no, ctry, ptype])
        y.append(label_idx)

    x = np.array(x)
    y = np.array(y)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_split_ratio, shuffle=True)

    train_list = []
    for idx, x_train_each in enumerate(x_train):
        y_train_each = y_train[idx]
        train_each_list = list(x_train_each)
        train_each_list.append(y_train_each)
        train_list.append(train_each_list)

    valid_list = []
    for idx, x_valid_each in enumerate(x_valid):
        y_valid_each = y_valid[idx]
        valid_each_list = list(x_valid_each)
        valid_each_list.append(y_valid_each)
        valid_list.append(valid_each_list)

    train_dataset_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/train.csv')
    valid_dataset_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/valid.csv')

    csv_util.write_csv_from_list(train_list, header_text_list=['appl_no', 'ctry', 'ptype', 'label_idx'], csv_file=train_dataset_path)
    csv_util.write_csv_from_list(valid_list, header_text_list=['appl_no', 'ctry', 'ptype', 'label_idx'], csv_file=valid_dataset_path)


if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    main(pjt_code)

