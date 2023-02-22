# coding:utf-8
"""ti_ab, ti_ab_clr, ti_ab_cla 를 이용한 3개의 tokenizer를 만든다."""
import os
import random
from keras_preprocessing.text import Tokenizer
from collections import Counter

from util import path_util
from util import csv_util
from util import pickle_util


def _make_tokenizer(cleandoc_list:list, word_min_occur:int):

    random.seed(42)

    vocab_0 = Counter()     # ti, ab
    vocab_1 = Counter()     # ti, ab, clr
    vocab_2 = Counter()     # ti, ab, cla

    for cleandoc in cleandoc_list:
        ti, ab, clr, cla = cleandoc
        text_0 = f'{ti} {ab}'
        text_1 = f'{ti} {ab} {clr}'
        text_2 = f'{ti} {ab} {cla}'

        vocab_0.update(text_0.split())
        vocab_1.update(text_1.split())
        vocab_2.update(text_2.split())

    word_list_0 = [k for k, c in vocab_0.items() if c >= word_min_occur]
    word_list_1 = [k for k, c in vocab_1.items() if c >= word_min_occur]
    word_list_2 = [k for k, c in vocab_2.items() if c >= word_min_occur]

    random.shuffle(word_list_0)
    random.shuffle(word_list_1)
    random.shuffle(word_list_2)

    tokenizer_0 = Tokenizer()
    tokenizer_0.fit_on_texts(word_list_0)

    tokenizer_1 = Tokenizer()
    tokenizer_1.fit_on_texts(word_list_1)

    tokenizer_2 = Tokenizer()
    tokenizer_2.fit_on_texts(word_list_2)

    return [tokenizer_0, tokenizer_1, tokenizer_2]


def main(pjt_code):
    root_path = path_util.get_project_root_path()
    word_min_occur = 10

    cleandoc_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/03.csv')
    cleandoc_list = csv_util.read_csv_to_list(csv_file=cleandoc_path,
                                              field_idx_list=[3, 4, 5, 6])
    tokenizer_list = _make_tokenizer(cleandoc_list, word_min_occur)

    for idx, tokenizer in enumerate(tokenizer_list):
        tokenizer_path = os.path.join(root_path, f'dataset/{pjt_code}/final_pkl/tokenizer_{idx}.pkl')
        pickle_util.dump_object(tokenizer, file_path=tokenizer_path)


if __name__ == '__main__':
    pjt_code = 'CPC_C01'
    main(pjt_code)
