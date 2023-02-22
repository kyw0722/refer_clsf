# coding:utf-8
import os
from util import path_util
from util import csv_util
from util import doc_util


def main(pjt_code, pdt_code):

    root_path = path_util.get_project_root_path()
    standard_path = os.path.join(root_path, f'dataset/{pjt_code}/predict/{pdt_code}/intermediate_csv/01.csv')
    clean_path = os.path.join(root_path, f'dataset/{pjt_code}/predict/{pdt_code}/intermediate_csv/03.csv')

    standard_list = csv_util.read_csv_to_list(csv_file=standard_path,
                                              field_idx_list=[0, 1, 2, 3, 4, 5, 6],
                                              fline_skip=True)

    clean_total_list = []
    for standard_item in standard_list:
        appl_no, ctry, ptype, ti, ab, clr, cla = standard_item
        ti_tokens = doc_util.clean_text_tokens(ti)
        ab_tokens = doc_util.clean_text_tokens(ab)
        cla_tokens = doc_util.clean_text_tokens(cla)
        clr_tokens = doc_util.clean_text_tokens(clr)

        ti_clean = ' '.join(ti_tokens)
        ab_clean = ' '.join(ab_tokens)
        cla_clean = ' '.join(cla_tokens)
        clr_clean = ' '.join(clr_tokens)

        clean_list = [appl_no, ctry, ptype, ti_clean, ab_clean, clr_clean, cla_clean]
        clean_total_list.append(clean_list)

    csv_util.write_csv_from_list(contents_list=clean_total_list,
                                 header_text_list=['appl_no', 'ctry', 'ptype', 'ti', 'ab', 'clr', 'cla'],
                                 csv_file=clean_path)


if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    pdt_code = 'PRD_CPC_C01'
    main(pjt_code, pdt_code)
