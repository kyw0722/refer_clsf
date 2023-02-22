# coding:utf-8
import csv
import sys


def read_csv_to_list(csv_file:str, field_idx_list:list, fline_skip:bool=True) -> list:
    """
    csv file을 읽어서 리스트를 만든다.
    :param csv_file:
    :param field_idx_list:
    :param fline_skip:
    :return:
    """
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    csv_list = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        if fline_skip: next(csv_reader, None)
        for line in csv_reader:
            line_list = []
            for field_idx in field_idx_list:
                line_list.append(line[field_idx])
            csv_list.append(line_list)
    return csv_list


def write_csv_from_list(contents_list:list, header_text_list:list, csv_file:str):
    """list를 header가 있는 csv 파일로 만든다."""
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    with open(csv_file, 'w', encoding='utf-8', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerow(header_text_list)
        for content in contents_list:
            writer.writerow(content)



if __name__ == '__main__':
    pass