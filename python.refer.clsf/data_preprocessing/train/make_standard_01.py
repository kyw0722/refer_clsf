# coding:utf-8
"""source excel을 읽어서 01.csv 를 만든다."""
import os
from util import path_util
from util import excel_util
from util import csv_util



def main(pjt_code):
    root_path = path_util.get_project_root_path()

    source_excel_dir = os.path.join(root_path, f'dataset/{pjt_code}/source_excel')
    source_excel_file = os.listdir(source_excel_dir)[0]
    source_excel_path = os.path.join(source_excel_dir, source_excel_file)

    standard_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/01.csv')

    patent_total_list = excel_util.read_excel_to_list(excel_filename=source_excel_path,
                                                      sheet_index=0,
                                                      field_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
                                                      min_row=2)

    csv_util.write_csv_from_list(contents_list=patent_total_list,
                                 header_text_list=['출원번호', 'DB종류', '특허/실용 구분', '발명의 명칭', '요약', '전체청구항', '대표청구항', '분류명'],
                                 csv_file=standard_csv_path)




if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    main(pjt_code)
