# coding:utf-8
"""raw_excel_data 의 분리된 excel 파일을 합쳐서 source_excel 파일 디렉토리에 생성한다."""
import os
from util import path_util
from util import excel_util

def main(raw_excel_dir: str, target_excel_dir: str, target_excel_fie: str):

    raw_file_list = os.listdir(raw_excel_dir)
    patent_total_list = []
    for raw_file in raw_file_list:
        file_path = os.path.join(raw_excel_dir, raw_file)
        patent_list = excel_util.read_excel_to_list(excel_filename=file_path,
                                                    sheet_index=0,
                                                    field_idxs=[0, 1, 2, 3, 4, 5, 6, 7],    # 출원번호, DB종류, 특허/실용 구분, 발명의 명칭, 요약, 전체청구항, 대표청구항, 사용자태그
                                                    min_row=2)
        patent_total_list += patent_list

    target_path = os.path.join(target_excel_dir, target_excel_file)

    excel_util.write_excel_from_list_with_style(excel_filename=target_path,
                                                data_list=patent_total_list,
                                                head_list=['출원번호', 'DB종류', '특허/실용 구분', '발명의 명칭', '요약', '전체청구항', '대표청구항', '분류명'])






if __name__ == '__main__':

    root_path = path_util.get_project_root_path()
    pjt_code = 'CPC_C01'

    raw_excel_dir = os.path.join(root_path, f'raw_excel_data/{pjt_code}')
    target_excel_dir = os.path.join(root_path, f'dataset/{pjt_code}/source_excel')
    target_excel_file = 'train_cpc_c01.xlsx'


    main(raw_excel_dir, target_excel_dir, target_excel_file)



