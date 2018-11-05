# -*- coding: utf-8 -*-

import sys
import os
CURRENT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.dirname(CURRENT_PATH))
import numpy as np
import json
import random
from utils.data_process import read_chars_dict, read_words_dict, mul_snli_jsonl_encoded, words_embedding_file

snli_train_path = mul_snli_jsonl_encoded[0]
snli_dev_path = mul_snli_jsonl_encoded[1]
snli_test_path = mul_snli_jsonl_encoded[2]

def read_vocab_size():
    """Don't include PADDING and UNKNOWN"""
    words_dict = read_words_dict()
    return len(words_dict)

def read_chars_vocab_size():
    """Don't include PADDING"""
    chars_dict = read_chars_dict()
    return len(chars_dict)

def read_embedding_table():
    """
    The function of read embedding table.
    Notes:
        Don't include PADDING and UNKNOWN.
    """
    embedding_obj = open(words_embedding_file, 'r')
    embedding_table = []
    for i, emb in enumerate(embedding_obj):
        embedding_table.append(eval(emb))

    return np.array(embedding_table).astype(np.float32)
     
def shuffle_data(data_jsonl, shuffled_file):
    data_obj = open(data_jsonl, 'r')
    data_list = list()
    for i, line in enumerate(data_obj):
        data_list.append(line)
    data_obj.close()
    random.shuffle(data_list)
    shuffled_data = open(shuffled_file, 'w')
    for line in data_list:
        shuffled_data.write(line)
    shuffled_data.close


def padding_batch_data(
    unalign_data, pad_index=0,
    deminsion=2, max_dim_2=None):
    """
    Padding unalign data.
    """
    if deminsion == 2 and max_dim_2 != None:
        max_len = max_dim_2
    else:
        max_len = max(map(len, unalign_data))

    padded_data = list()
    for cell in unalign_data:
        cell_len = len(cell)
        pad_num = max_len - cell_len
        if deminsion == 2:
            cell = cell + [pad_index]*pad_num
            padded_data.append(cell)
        elif deminsion == 3:
            cell, _ = padding_batch_data(cell,max_dim_2=max_dim_2)
            
            cell = cell + [ [pad_index]*len(cell[0]) ] * pad_num
            padded_data.append(cell)
        else:
            print("Padding error")
    return padded_data, max_len


def read_data(data_jsonl):
    return open(data_jsonl, 'r')


def read_batch_data(data_obj,batch_size):
    """
    The function of read batch data.
    Params:
        data_obj: the file object which is read_data function returned.
        batch_size: the number of batch size.
    Returns:
        batch_data: the dict of batch data.
        read_end: a flag indicating whether or not it has been read.
    """
    read_end = False
    batch_data = dict()
    batch_data["label"] = list()
    batch_data["sentence1_word"] = list()
    batch_data["sentence1_char"] = list()
    batch_data["sentence2_word"] = list()
    batch_data["sentence2_char"] = list()
    max_len_char_1 = 0
    max_len_char_2 = 0
    for i in range(batch_size):
        json_str = data_obj.readline()
        # 是否读完
        if json_str == '':
            read_end = True
            data_obj.seek(0)
            json_str = data_obj.readline()
        elem = json.loads(json_str)
        batch_data["label"].append(elem["label"])
        batch_data["sentence1_word"].append(elem["sentence1_word"])
        batch_data["sentence2_word"].append(elem["sentence2_word"])

        padded_char_1,max_len_1 = padding_batch_data(elem["sentence1_char"])
        max_len_char_1 = max_len_char_1 if max_len_char_1 > max_len_1 else  max_len_1
        batch_data["sentence1_char"].append(padded_char_1)

        padded_char_2,max_len_2 = padding_batch_data(elem["sentence2_char"])
        max_len_char_2 = max_len_char_2 if max_len_char_2 > max_len_2 else max_len_2
        batch_data["sentence2_char"].append(padded_char_2)
    
    batch_data["label"] = np.array(batch_data["label"])
    # sentence word
    batch_data["sentence1_word"], _ = padding_batch_data(
        batch_data["sentence1_word"]
    )
    batch_data["sentence1_word"] = np.array(
        batch_data["sentence1_word"],
        dtype=np.int32
    )
    batch_data["sentence2_word"], _ = padding_batch_data(
        batch_data["sentence2_word"]
    )
    batch_data["sentence2_word"] = np.array(
        batch_data["sentence2_word"],
        dtype=np.int32
    )
    # sentence character
    # print(batch_data["sentence1_char"])
    batch_data["sentence1_char"],_ = padding_batch_data(
        batch_data["sentence1_char"],
        deminsion=3,
        max_dim_2=max_len_char_1
    )
    batch_data["sentence1_char"] = np.array(
        batch_data["sentence1_char"],
        dtype=np.int32
    )
    batch_data["sentence2_char"],_ = padding_batch_data(
        batch_data["sentence2_char"],
        deminsion=3,
        max_dim_2=max_len_char_2
    )
    batch_data["sentence2_char"] = np.array(
        batch_data["sentence2_char"],
        dtype=np.int32
    )

    return batch_data, read_end
    

if __name__ == "__main__":
    data_obj = read_data(snli_train_path)
    read_end = False
    batch_i = 0
    while(read_end == False):
        batch_i += 1
        batch_data, read_end = read_batch_data(data_obj, batch_size=3)
        if(batch_i == 4):
            print(batch_data["label"].shape)
            print(batch_data["sentence1_word"].shape)
            print(batch_data["sentence1_char"].shape)
            print(batch_data["sentence2_word"].shape)
            print(batch_data["sentence2_char"].shape)
    print(read_end)
    print(batch_data["label"].shape)
    print(batch_data["sentence1_word"].shape)
    print(batch_data["sentence1_char"].shape)
    print(batch_data["sentence2_word"].shape)
    print(batch_data["sentence2_char"].shape)
