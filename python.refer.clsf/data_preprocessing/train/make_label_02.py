# coding:utf-8
'''분류명을 중복제거해서 02.csv 를 만든다.'''
import os
from util import path_util
from util import csv_util


def main(pjt_code):
    root_path = path_util.get_project_root_path()
    standard_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/01.csv')
    standard_list = csv_util.read_csv_to_list(csv_file=standard_path,
                                              field_idx_list=[7],
                                              fline_skip=True)

    label_set = {x[0].strip() for x in standard_list}
    label_list = sorted(label_set)
    label_csv_list = [[x] for x in label_list]

    label_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/02.csv')
    csv_util.write_csv_from_list(contents_list=label_csv_list,
                                 header_text_list=['label'],
                                 csv_file=label_path)





if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    main(pjt_code)