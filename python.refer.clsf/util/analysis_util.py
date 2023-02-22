# coding:utf-8
import os
import pandas as pd
from util import path_util


def _gen_filepath_list(best_arrays):
    file_path_list = []
    print(best_arrays)

    for best_array in best_arrays:
        suffix = best_array[0]
        epoch = str(best_array[1])
        val_acc = format(best_array[2], '.3f')
        val_loss = format(best_array[3], '.4f')
        acc = format(best_array[4], '.3f')
        loss = format(best_array[5], '.4f')

        file_path = '{}-{}-{}-{}-{}-{}.h5'.format(suffix, epoch, val_acc, val_loss, acc, loss)
        file_path_list.append(file_path)

    return file_path_list


def model_info_from_file(model_file):
    values = model_file[:-3].split('-')
    suffix = values[0]
    epoch = int(values[1])
    val_acc = float(values[2])
    val_loss = float(values[3])
    acc = float(values[4])
    loss = float(values[5])
    diff = abs(loss - val_loss)

    return [suffix, epoch, val_acc, val_loss, acc, loss, diff]

def file_from_model_info(model_info):
    suffix = model_info[0]
    epoch = str(model_info[1])
    val_acc = format(model_info[2], '.3f')
    val_loss = format(model_info[3], '.4f')
    acc = format(model_info[4], '.3f')
    loss = format(model_info[5], '.4f')

    file_name = file_path = '{}-{}-{}-{}-{}-{}.h5'.format(suffix, epoch, val_acc, val_loss, acc, loss)

    return file_name



def select_best_model(save_folder):
    BEST_MODEL_CNT = 10

    pd_list = []
    for (path, dir, files_list) in os.walk(save_folder):
        for file in files_list:
            values = file[:-3].split('-')
            suffix = values[0]
            epoch = int(values[1])
            val_acc = float(values[2])
            val_loss = float(values[3])
            acc = float(values[4])
            loss = float(values[5])

            pd_list.append([suffix, epoch, val_acc, val_loss, acc, loss])

    pd_dataset = pd.DataFrame(pd_list, columns=['suffix', 'epoch', 'val_acc', 'val_loss', 'acc', 'loss'])
    sorted_dataset = pd_dataset.sort_values(
        by=['val_acc', 'val_loss', 'epoch'], ascending=[False, True, False])

    min_cnt = min(BEST_MODEL_CNT, len(sorted_dataset.values))

    best_model_array = sorted_dataset.values[:min_cnt]
    best_model_filename_list = _gen_filepath_list(best_model_array)

    # 해당되지 않는 파일은 모두 삭제한다.
    for (path, dir, files_list) in os.walk(save_folder):
        for file in files_list:
            if file in best_model_filename_list:
                pass
            else:
                file_path = '{}{}{}'.format(path, os.sep, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        print('remove error')

    return best_model_filename_list


def get_ensemble_participation_model(pjt_code, model_type):
    """
    앙상블에 포함될 모델을 가져온다.
    val_acc 가장 높은 것, val_loss 가장 낮은것, |val_loss - loss| 가장 낮은것.
    :param pjt_code:
    :param model_type:
    :return:
    """
    model_list = []     # ensemble에 참여할 모델파일 리스트
    root_path = path_util.get_project_root_path()
    save_folder_path = os.path.join(root_path, f'dataset/{pjt_code}/model_save/{model_type}')

    pd_list = []
    for path, dir, files_list in os.walk(save_folder_path):
        for file in files_list:
            model_info_list = model_info_from_file(file)
            pd_list.append(model_info_list)
    pd_dataset = pd.DataFrame(pd_list, columns=['suffix', 'epoch', 'val_acc', 'val_loss', 'acc', 'loss', 'diff'])

    # val_acc 가 가장 높은 것
    val_acc_model_info = \
        pd_dataset.sort_values(by=['val_acc', 'val_loss', 'epoch'], ascending=[False, True, False]).values[0]
    val_acc_model_file = file_from_model_info(val_acc_model_info)
    model_list.append(val_acc_model_file)
    suffix = val_acc_model_info[0]
    epoch = val_acc_model_info[1]
    index_name = pd_dataset[(pd_dataset['suffix'] == suffix) & (pd_dataset['epoch'] == epoch)].index
    pd_dataset.drop(index_name, inplace=True)

    # val_loss 가 가장 낮은 것
    val_loss_model_info = \
        pd_dataset.sort_values(by=['val_loss', 'val_acc', 'epoch'], ascending=[True, False, False]).values[0]
    val_loss_model_file = file_from_model_info(val_loss_model_info)
    model_list.append(val_loss_model_file)
    suffix = val_loss_model_info[0]
    epoch = val_loss_model_info[1]
    index_name = pd_dataset[(pd_dataset['suffix'] == suffix) & (pd_dataset['epoch'] == epoch)].index
    pd_dataset.drop(index_name, inplace=True)

    # |val_loss - loss| 가 가장 낮은 것
    val_diff_model_info = \
        pd_dataset.sort_values(by=['diff', 'val_acc', 'val_loss'], ascending=[True, False, True]).values[0]
    val_diff_model_file = file_from_model_info(val_diff_model_info)
    model_list.append(val_diff_model_file)

    return model_list

