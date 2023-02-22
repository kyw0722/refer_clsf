# coding:utf-8
"""텍스트 정리"""
import os
from util import path_util
from util import csv_util
from util import doc_util


def main(pjt_code):

    root_path = path_util.get_project_root_path()
    standard_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/01.csv')

    label_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/02.csv')
    label_list = csv_util.read_csv_to_list(csv_file=label_path,
                                           field_idx_list=[0],
                                           fline_skip=True)

    label_dict = {x[0]: idx for idx, x in enumerate(label_list)}

    clean_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/03.csv')

    standard_list = csv_util.read_csv_to_list(csv_file=standard_path,
                                              field_idx_list=[0, 1, 2, 3, 4, 5, 6, 7],
                                              fline_skip=True)


    clean_total_list = []
    for standard_item in standard_list:
        appl_no, ctry, ptype, ti, ab, cla, clr, label = standard_item
        ti_tokens = doc_util.clean_text_tokens(ti)
        ab_tokens = doc_util.clean_text_tokens(ab)
        cla_tokens = doc_util.clean_text_tokens(cla)
        clr_tokens = doc_util.clean_text_tokens(clr)
        label_idx = label_dict[label]

        ti_clean = ' '.join(ti_tokens)
        ab_clean = ' '.join(ab_tokens)
        cla_clean = ' '.join(cla_tokens)
        clr_clean = ' '.join(clr_tokens)

        clean_list = [appl_no, ctry, ptype, ti_clean, ab_clean, clr_clean, cla_clean, label_idx]
        clean_total_list.append(clean_list)


    csv_util.write_csv_from_list(contents_list=clean_total_list,
                                 header_text_list=['appl_no', 'ctry', 'ptype', 'ti', 'ab', 'clr', 'cla', 'label_idx'],
                                 csv_file=clean_path)




if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    main(pjt_code)