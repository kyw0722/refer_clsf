# coding:utf-8
""" LSTM 모델을 훈련시킨다."""
import os
import numpy as np
from tensorflow import keras

from util import path_util
from util.config import config
from util import csv_util
from util import pickle_util
from util import analysis_util


def _onehot_encoding(label_idx, class_cnt):
    label_onehot_encoded = list(np.zeros(class_cnt, dtype=np.int))
    for label_id in label_idx.split(','):
        try:
            label_onehot_encoded[int(label_id)] = 1
        except:
            pass
    return label_onehot_encoded


def _define_model(max_doc_length, vocab_size, learning_rate, class_cnt, layer_count, dropout_rate):

    embedding_size = 100
    inputs = keras.layers.Input(shape=(max_doc_length,))
    embedding = keras.layers.Embedding(input_dim=vocab_size,
                                       output_dim=embedding_size,
                                       trainable=True)(inputs)
    lstms = keras.layers.Bidirectional(keras.layers.LSTM(layer_count, return_sequences=True))(embedding)
    lstms = keras.layers.LSTM(layer_count, return_sequences=False)(lstms)

    dense = keras.layers.Dense(class_cnt * 2)(lstms)
    batch_norm = keras.layers.BatchNormalization()(dense)
    drop = keras.layers.ELU()(batch_norm)
    drop = keras.layers.Dropout(dropout_rate)(drop)

    outputs = keras.layers.Dense(class_cnt, activation='softmax')(drop)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model




def main(pjt_code):
    root_path = path_util.get_project_root_path()
    model_type = "ltm_singlex"

    docu_features = config[model_type]["DOCU_FEATURES"]
    max_doc_length = int(config[model_type]["MAX_DOC_LENGTH"])
    learning_rates = config[model_type]['LEARNING_RATES']
    batch_sizes = config[model_type]['BATCH_SIZES']
    layer_counts = config[model_type]['LAYER_COUNTS']
    dropout_rates = config[model_type]['DROPOUT_RATES']
    epochs = int(config[model_type]['epochs'])
    patience = int(config[model_type]['patience'])

    docu_features = [int(x.strip()) for x in docu_features.split(',')]
    learning_rates = [int(x.strip()) for x in learning_rates.split(',')]
    batch_sizes = [int(x.strip()) for x in batch_sizes.split(',')]
    layer_counts = [int(x.strip()) for x in layer_counts.split(',')]
    dropout_rates = [int(x.strip()) for x in dropout_rates.split(',')]

    hyperparams = [
        (learning_rate, batch_size, layer_count, dropout_rate)
        for learning_rate in learning_rates
        for batch_size in batch_sizes
        for layer_count in layer_counts
        for dropout_rate in dropout_rates
    ]

    total_run = len(hyperparams) * len(docu_features)
    print("------------------>", total_run)

    for docu_idx in docu_features:
        train_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/train.csv')
        valid_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/valid.csv')
        clean_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/03.csv')
        label_csv_path = os.path.join(root_path, f'dataset/{pjt_code}/intermediate_csv/02.csv')
        tokenizer_pkl_path = os.path.join(root_path, f'dataset/{pjt_code}/final_pkl/tokenizer_{docu_idx}.pkl')

        # appl_num, ctry, ptype, label_idx
        train_list = csv_util.read_csv_to_list(train_csv_path, field_idx_list=[0, 1, 2, 3], fline_skip=True)
        valid_list = csv_util.read_csv_to_list(valid_csv_path, field_idx_list=[0, 1, 2, 3], fline_skip=True)

        clean_list = csv_util.read_csv_to_list(clean_csv_path, field_idx_list=[0, 1, 2, 3, 4, 5, 6, 7], fline_skip=True)
        label_list = csv_util.read_csv_to_list(label_csv_path, field_idx_list=[0], fline_skip=True)
        tokenizer = pickle_util.load_object(tokenizer_pkl_path)

        # 부류 갯수 파악
        class_cnt = len(label_list)

        # clean 관련 dict 만들기 key: {appl_no}_{ctry}_{ptype}  value: [ti, ab, clr, cla]
        text_dict = {}
        for clean_doc in clean_list:
            appl_no, ctry, ptype, ti, ab, clr, cla, _ = clean_doc
            key = f'{appl_no}_{ctry}_{ptype}'
            value = [ti, ab, clr, cla]
            text_dict[key] = value

        train_doc = []
        y_train = []
        for train_one in train_list:
            appl_no, ctry, ptype, label_idx = train_one
            key = f'{appl_no}_{ctry}_{ptype}'
            train_doc_one = text_dict[key]
            ti, ab, clr, cla = train_doc_one
            if docu_idx == 0:
                text = f'{ti} {ab}'
            elif docu_idx == 1:
                text = f'{ti} {ab} {clr}'
            elif docu_idx == 2:
                text = f'{ti} {ab} {cla}'
            train_doc.append(text)
            label_onehot = _onehot_encoding(label_idx, class_cnt)
            y_train.append(label_onehot)

        valid_doc = []
        y_valid = []
        for valid_one in valid_list:
            appl_no, ctry, ptype, label_idx = valid_one
            key = f'{appl_no}_{ctry}_{ptype}'
            valid_doc_one = text_dict[key]
            ti, ab, clr, cla = valid_doc_one
            if docu_idx == 0:
                text = f'{ti} {ab}'
            elif docu_idx == 1:
                text = f'{ti} {ab} {clr}'
            elif docu_idx == 2:
                text = f'{ti} {ab} {cla}'
            valid_doc.append(text)
            label_onehot = _onehot_encoding(label_idx, class_cnt)
            y_valid.append(label_onehot)

        seq_train = tokenizer.texts_to_sequences(train_doc)
        seq_valid = tokenizer.texts_to_sequences(valid_doc)

        max_doc_length_seq = max([len(x) for x in seq_train])
        max_doc_length = min(max_doc_length_seq, max_doc_length)

        vocab_size = len(tokenizer.word_counts) + 1

        x_train = keras.preprocessing.sequence.pad_sequences(seq_train,
                                                             maxlen=max_doc_length,
                                                             padding='post')
        x_valid = keras.preprocessing.sequence.pad_sequences(seq_valid,
                                                             maxlen=max_doc_length,
                                                             padding='post')

        for idx, hyperparam in enumerate(hyperparams):
            now_run = docu_idx * len(hyperparams) + idx + 1
            print('진행상황 {} / {}'.format(now_run, total_run))

            learning_rate, batch_size, layer_count, dropout_rate = hyperparam

            suffix_model_name = '_'.join([str(x) for x in hyperparam])
            suffix_model_name = '{}_'.format(model_type) + suffix_model_name + '_' + str(docu_idx)

            learning_rate = 1 / pow(10, learning_rate)
            dropout_rate = dropout_rate / 10

            model = _define_model(max_doc_length=max_doc_length,
                                  vocab_size=vocab_size,
                                  learning_rate=learning_rate,
                                  class_cnt=class_cnt,
                                  layer_count=layer_count,
                                  dropout_rate=dropout_rate)

            # 모델 구조 확인하기
            keras.utils.plot_model(model, 'ltm.jpg', show_shapes=True)

            model_filename = os.path.join(root_path,
                                          'dataset/{}/model_save/{}/{}-'.format(pjt_code, model_type,
                                                                                suffix_model_name))
            model_filename = model_filename + '{epoch}-{val_acc:.3f}-{val_loss:.4f}-{acc:.3f}-{loss:.4f}.h5'

            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
            mc = keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', mode='min', save_best_only=False)

            model.fit(x_train, np.array(y_train),
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_valid, np.array(y_valid)),
                      callbacks=[es, mc])

            save_folder_path = os.path.join(root_path, f'dataset/{pjt_code}/model_save/{model_type}')
            analysis_util.select_best_model(save_folder_path)



if __name__ == "__main__":
    pjt_code = "CPC_C01"
    main(pjt_code)