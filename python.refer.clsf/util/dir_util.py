import os
from typing import List
from util import path_util
from util.config import config


def make_dataset_dir(pjt_code):
    root_path = path_util.get_project_root_path()

    model_types = config['NORMAL']['MODELS']
    train_dataset_dirs = config['NORMAL']['DATASET_DIR']
    have_model_dirs = config['NORMAL']['HAVE_MODELS_DIR']

    model_type_list = [x.strip() for x in model_types.split(',')]
    train_dataset_dir_list = [x.strip() for x in train_dataset_dirs.split(',')]
    have_model_dir_list = [x.strip() for x in have_model_dirs.split(',')]

    will_make_dirs = []
    for train_dataset_dir in train_dataset_dir_list:
        if train_dataset_dir in have_model_dir_list:
            for model_type in model_type_list:
                sub_path = f'dataset/{pjt_code}/{train_dataset_dir}/{model_type}'
                full_path = os.path.join(root_path, sub_path)
                will_make_dirs.append(full_path)
        else:
            sub_path = f'dataset/{pjt_code}/{train_dataset_dir}'
            full_path = os.path.join(root_path, sub_path)
            will_make_dirs.append(full_path)

    for make_dir in will_make_dirs:
        os.makedirs(make_dir, exist_ok=True)


def make_predict_dataset_dir(pjt_code, pdt_code):
    root_path = path_util.get_project_root_path()
    predict_dataset_dirs = config['NORMAL']['PREDICT_DATASET_DIR']
    predict_dataset_dir_list = [x.strip() for x in predict_dataset_dirs.split(',')]

    will_make_dirs: List[str] = []
    for predict_dataset_dir in predict_dataset_dir_list:
        prd_path = os.path.join(root_path, f'dataset/{pjt_code}/predict/{pdt_code}/{predict_dataset_dir}')
        will_make_dirs.append(prd_path)

    for make_dir in will_make_dirs:
        os.makedirs(make_dir, exist_ok=True)



if __name__ == '__main__':
    # make_dataset_dir('TV')
    make_predict_dataset_dir(pjt_code='CPC_C01', pdt_code='PRD_CPC_C01')

