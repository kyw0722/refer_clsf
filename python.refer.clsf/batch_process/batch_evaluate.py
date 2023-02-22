# coding:utf-8
import collections
import os
import numpy as np
import pandas as pd
from tensorflow import keras

from util.config import config

from util import path_util
from util import analysis_util
from util import dataset_util
from util import csv_util


def _evaluate_model(pjt_code, model_filename):

    root_path = path_util.get_project_root_path()
    model_type = '_'.join([x for x in model_filename.split('_')[:2]])
    model_filename_path = os.path.join(root_path, f'dataset/{pjt_code}/model_save/{model_type}/{model_filename}')

    model = keras.models.load_model(model_filename_path)

    docu_feature = int(model_filename.split('-')[0].split('_')[-1])

    # model에 맞는 valid_data를 생성한다.

    datasetMaker = dataset_util.DatasetMaker(pjt_code=pjt_code)
    valid_x = datasetMaker.makeValidDataset(model_type=model_type, docu_feature=docu_feature, model=model)

    if model_type == 'cnn_singlex':
        results = model.predict([valid_x, valid_x, valid_x])
    else:
        results = model.predict(valid_x)

    return results


def main(pjt_code):

    model_type_list = [x.strip() for x in config['NORMAL']['MODELS'].split(',')]

    total_model_file_list = []
    for model_type in model_type_list:
        model_list = analysis_util.get_ensemble_participation_model(pjt_code, model_type)
        total_model_file_list += model_list

    total_result = []
    total_weighting_result = []

    for model_file in total_model_file_list:
        model_info = analysis_util.model_info_from_file(model_file)
        val_acc = model_info[2]
        result = _evaluate_model(pjt_code, model_file)
        total_result.append(result)
        total_weighting_result.append(result * (val_acc * 100))

    # 정답을 가져온다.
    root_path = path_util.get_project_root_path()
    valid_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/valid.csv')
    true_list = csv_util.read_csv_to_list(csv_file=valid_csv_path,
                                           field_idx_list=[3],  # appl_no, ctry, ptype, label_idx
                                           fline_skip=True)
    true_valid = np.array([int(x[0]) for x in true_list])

    # summation
    summation_result = np.sum(total_result, axis=0)
    y_summation = np.argmax(summation_result, axis=1)
    summation_acc = np.sum(true_valid == y_summation) / len(true_valid)
    print("summation_acc : ", summation_acc)

    # weighting
    weighting_result = np.sum(total_weighting_result, axis=0)
    y_weighting = np.argmax(weighting_result, axis=1)
    weighting_acc = np.sum(true_valid == y_weighting) / len(true_valid)
    print("weighting_acc : ", summation_acc)

    # voting
    y_voting = []
    voting_index = np.argmax(total_result, axis=2)
    trans_voting_index = np.transpose(voting_index)

    for idx, voting in enumerate(trans_voting_index):
        voting_collection = collections.Counter(voting).most_common()
        if len(voting_collection) == 1:     # 모든 분류기가 동일한 결과일 경우
            voting_result = voting_collection[0][0]
        else:
            if voting_collection[0][1] == voting_collection[1][1]:
                voting_result = y_weighting[idx]
            else:
                voting_result = voting_collection[0][0]
        y_voting.append(voting_result)

    voting_acc = np.sum(true_valid == y_voting) / len(true_valid)

    print("voting_acc : ", voting_acc)

    pd_ensemble_acc_list = []  # type, acc, priority
    pd_ensemble_acc_list.append(['summation', summation_acc, 2])
    pd_ensemble_acc_list.append(['weighting', weighting_acc, 1])
    pd_ensemble_acc_list.append(['voting', voting_acc, 0])

    pd_acc_dataset = pd.DataFrame(pd_ensemble_acc_list, columns=['type', 'acc', 'priority'])
    sorted_acc_dataset = pd_acc_dataset.sort_values(
        by=['acc', 'priority'], ascending=[False, True]
    )

    ensemble_result_path = os.path.join(root_path, f'dataset/{pjt_code}/train_result/ensemble_result.csv')
    best_ensemble_type_path = os.path.join(root_path, f'dataset/{pjt_code}/train_result/best_ensemble_type.csv')

    csv_util.write_csv_from_list(sorted_acc_dataset.values, ['type', 'acc', 'priority'], ensemble_result_path)
    csv_util.write_csv_from_list([sorted_acc_dataset.values[:1][0][:1]], ['type'], best_ensemble_type_path)


if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    main(pjt_code)