# coding:utf-8
import os
import numpy as np
from tensorflow import keras
import collections

from util.config import config
from util import analysis_util
from util import path_util
from util import csv_util
from util import dataset_util
from util import excel_util


def _make_predict_excel(pjt_code, pdt_code, total_results, total_weighting_results):
    root_path = path_util.get_project_root_path()
    # best ensemble type를 읽어온다.
    best_ensemble_type_path = os.path.join(root_path, f'dataset/{pjt_code}/train_result/best_ensemble_type.csv')
    best_ensemble_types = csv_util.read_csv_to_list(best_ensemble_type_path, [0])
    best_ensemble_type = best_ensemble_types[0][0]

    label_list_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/02.csv')
    label_list = csv_util.read_csv_to_list(label_list_csv_path, [0])
    label_dict = {idx: x[0] for idx, x in enumerate(label_list)}

    if best_ensemble_type == 'summation':
        result = np.round(np.sum(total_results, axis=0) / len(total_results), 4)
    elif best_ensemble_type == 'weighting':
        total_weighting_result = np.sum(total_weighting_results, axis=0)
        result = np.round(np.true_divide(total_weighting_result, total_weighting_result.sum(axis=1, keepdims=True)), 4)
    elif best_ensemble_type == 'voting':
        y_voting = []
        voting_index = np.argmax(total_results, axis=2)
        trans_voting_index = np.transpose(voting_index)
        for idx, voting in enumerate(trans_voting_index):
            voting_collection = collections.Counter(voting).most_common()
            voting_cnt_list = np.zeros(len(label_list))
            for voting in voting_collection:
                voting_cnt_list[voting[0]] = voting[1]
            y_voting.append(voting_cnt_list / sum(voting_cnt_list))

        result = np.round(np.array(y_voting), 4)

    clean_doc_path = os.path.join(root_path, f"dataset/{pjt_code}/predict/{pdt_code}/intermediate_csv/03.csv")
    clean_doc_list = csv_util.read_csv_to_list(clean_doc_path,
                                               field_idx_list=[0, 1, 2],
                                               fline_skip=True)

    predict_list = []
    for idx, patent_key in enumerate(clean_doc_list):
        appl_num, ctry, ptype = patent_key
        cate_probs = result[idx]

        cate_sorted_index = sorted(range(len(cate_probs)), key=lambda k: cate_probs[k], reverse=True)
        cate_disp_count = min(3, len(label_dict.keys()))

        predict_item = [appl_num, ctry, ptype]
        for x in range(cate_disp_count):
            cate_index = cate_sorted_index[x]
            cate_name = label_dict[cate_index]
            cate_prob = cate_probs[cate_index]
            predict_item.append(cate_name)
            predict_item.append(cate_prob)

        predict_list.append(predict_item)


    result_excel_path = os.path.join(root_path, f'dataset/{pjt_code}/predict/{pdt_code}/result_excel/{pjt_code}_predict.xlsx')

    excel_util.write_excel_from_list_with_style(excel_filename=result_excel_path,
                                                data_list=predict_list,
                                                head_list=['출원번호', '국가', '특허/실용 구분', 'top1', 'prob1', 'top2', 'prob2', 'top3', 'prob3'])




def _predict_model(pjt_code, pdt_code, model_filename):

    root_path = path_util.get_project_root_path()
    model_type = '_'.join([x for x in model_filename.split('_')[:2]])
    model_filename_path = os.path.join(root_path, f'dataset/{pjt_code}/model_save/{model_type}/{model_filename}')

    model = keras.models.load_model(model_filename_path)

    docu_feature = int(model_filename.split('-')[0].split('_')[-1])

    # model에 맞는 predict_data를 생성한다.
    datasetMaker = dataset_util.DatasetMaker(pjt_code=pjt_code, pdt_code=pdt_code)
    predict_x = datasetMaker.makePredictDataset(model_type=model_type, docu_feature=docu_feature, model=model)

    if model_type == 'cnn_singlex':
        results = model.predict([predict_x, predict_x, predict_x])
    else:
        results = model.predict(predict_x)

    return results


def main(pjt_code, pdt_code):
    model_type_list = [x.strip() for x in config['NORMAL']['MODELS'].split(',')][:]

    total_model_file_list = []
    for model_type in model_type_list:
        model_list = analysis_util.get_ensemble_participation_model(pjt_code, model_type)
        total_model_file_list += model_list

    total_result = []
    total_weighting_result = []

    for model_file in total_model_file_list:
        model_info = analysis_util.model_info_from_file(model_file)
        val_acc = model_info[2]
        result = _predict_model(pjt_code, pdt_code, model_file)
        total_result.append(result)
        total_weighting_result.append(result * (val_acc * 100))

    _make_predict_excel(pjt_code, pdt_code, total_result, total_weighting_result)



if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    pdt_code = 'PRD_CPC_C01'
    main(pjt_code, pdt_code)