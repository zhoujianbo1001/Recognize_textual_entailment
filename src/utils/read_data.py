# -*- coding: utf-8 -*-
import sys
import os
CURRENT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_PATH)
sys.path.append(os.path.dirname(CURRENT_DIR))

import numpy as np
import json
import random

from utils.data_process import read_chars_dict, read_words_dict
import utils.params as params

CONFIGS = params.load_configs()

@DeprecationWarning
def read_vocab_size():
    """Don't include PADDING and UNKNOWN"""
    words_dict = read_words_dict()
    return len(words_dict)

@DeprecationWarning
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
    embedding_obj = open(CONFIGS.words_embedding_file, 'r')
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


def batch_data_pad_or_clip(
    unalign_data, pad_index=0,
    deminsions=2, len_dim1=None, len_dim2=None):
    """
    Padding or clipping unalign data.
    Params:
        unalign_data: the list of unalign_data.
        pad_index: the index of padded data.
        deminsions: the deminsions of data, which is 2 or 3.
        len_dim1: the dim1's length needed to pad or clip.
        len_dim2: the dim2's length needed to pad or clip.
    """
    if len_dim1 == None:
        max_len = max(map(len, unalign_data))
    elif len_dim1 != None:
        max_len = len_dim1
    
    if deminsions == 3 and len_dim2 == None or deminsions < 2 or deminsions > 3:
        raise NotImplementedError

    num_clipped_1 = 0
    num_clipped_2 = 0

    padded_data = list()
    for cell in unalign_data:
        cell_len = len(cell)
        pad_num = max_len - cell_len
        if deminsions == 2:
            if pad_num > 0:
                # pad
                cell = cell + [pad_index]*pad_num
            elif pad_num < 0:
                # clip
                cell = cell[0:max_len]
                num_clipped_1 += 1

            padded_data.append(cell)
        elif deminsions == 3:
            cell, _, num_cliped_dim1, _ = batch_data_pad_or_clip(
                cell, pad_index=pad_index, deminsions=2, len_dim1=len_dim2)
            num_clipped_2 += num_cliped_dim1
            if pad_num > 0:
                # pad
                cell = cell + [ [pad_index]*len(cell[0]) ] * pad_num
            else:
                # clip
                cell = cell[0:max_len]
                num_clipped_1 += 1

            padded_data.append(cell)
        else:
            print("Padding error")
    return padded_data, max_len, num_clipped_1, num_clipped_2


def read_data(data_jsonl):
    return open(data_jsonl, 'r')


def read_batch_data(data_obj, batch_size, seq_len=None, chars_len=None):
    """
    The function of read batch data.
    Params:
        data_obj: the file object which is read_data function returned.
        batch_size: the number of batch size.
        seq_len: the max length of sequences.
        chars_len: the max length of word's characters.

    Returns:
        batch_data: the dict of batch data.
        read_end: a flag indicating whether or not it has been read.
        read_batch: the number of readed data.
        num_clipped_seq: the number of clipped sequence.
        num_clipped_chars: the number of clipped words.
    """
    num_clipped_seq = 0
    num_clipped_chars = 0
    read_end = False
    batch_data = dict()
    batch_data["label"] = list()
    batch_data["sentence1_word"] = list()
    batch_data["sentence1_char"] = list()
    batch_data["sentence2_word"] = list()
    batch_data["sentence2_char"] = list()
    read_batch = 0
    max_len_char_1 = 0
    max_len_char_2 = 0
    for i in range(batch_size):
        json_str = data_obj.readline()
        # 是否读完
        if json_str == '':
            read_end = True
            data_obj.seek(0)
            break
        elem = json.loads(json_str)
        # append label
        if elem["label"] == 4:
            continue
        batch_data["label"].append(elem["label"])
        
        # append sentence words
        batch_data["sentence1_word"].append(elem["sentence1_word"])
        batch_data["sentence2_word"].append(elem["sentence2_word"])

        # appeed sentence characters
        padded_char_1, max_len_1, num_clipped_c1, _ = batch_data_pad_or_clip(
            elem["sentence1_char"], deminsions=2, len_dim1=chars_len)
        max_len_char_1 = max_len_char_1 if max_len_char_1 > max_len_1 else  max_len_1
        batch_data["sentence1_char"].append(padded_char_1)
        num_clipped_chars += num_clipped_c1

        padded_char_2,max_len_2, num_clipped_c2, _ = batch_data_pad_or_clip(
            elem["sentence2_char"], deminsions=2, len_dim1=chars_len)
        max_len_char_2 = max_len_char_2 if max_len_char_2 > max_len_2 else max_len_2
        batch_data["sentence2_char"].append(padded_char_2)
        num_clipped_chars += num_clipped_c2

        read_batch += 1
    
    batch_data["label"] = np.array(batch_data["label"])
    # sentence word
    batch_data["sentence1_word"], _, num_clipped_s1, _ = batch_data_pad_or_clip(
        batch_data["sentence1_word"], deminsions=2, len_dim1=seq_len
    )
    batch_data["sentence1_word"] = np.array(
        batch_data["sentence1_word"],
        dtype=np.int32
    )
    num_clipped_seq += num_clipped_s1

    batch_data["sentence2_word"], _, num_clipped_s2, _ = batch_data_pad_or_clip(
        batch_data["sentence2_word"], deminsions=2, len_dim1=seq_len
    )
    batch_data["sentence2_word"] = np.array(
        batch_data["sentence2_word"],
        dtype=np.int32
    )
    num_clipped_seq += num_clipped_s2

    # sentence character
    # print(batch_data["sentence1_char"])
    batch_data["sentence1_char"], _, _, _ = batch_data_pad_or_clip(
        batch_data["sentence1_char"],
        deminsions=3,
        len_dim1=seq_len,
        len_dim2=max_len_char_1
    )
    batch_data["sentence1_char"] = np.array(
        batch_data["sentence1_char"],
        dtype=np.int32
    )
    batch_data["sentence2_char"], _, _, _ = batch_data_pad_or_clip(
        batch_data["sentence2_char"],
        deminsions=3,
        len_dim1=seq_len,
        len_dim2=max_len_char_2
    )
    batch_data["sentence2_char"] = np.array(
        batch_data["sentence2_char"],
        dtype=np.int32
    )

    return batch_data, read_end, read_batch, num_clipped_seq, num_clipped_chars
    

if __name__ == "__main__":
    data_obj = read_data(CONFIGS.dev_encoded_path)
    read_end = False
    batch_i = 0
    num_clipped_seq = 0
    num_clipped_chars = 0
    while(read_end == False):
        batch_i += 1
        batch_data, read_end, _, num_clipped_s, num_clipped_c = read_batch_data(
            data_obj, batch_size=3, seq_len=58, chars_len=16)
        num_clipped_seq += num_clipped_s
        num_clipped_chars += num_clipped_c
        if(batch_i == 4):
            print(batch_data["label"].shape)
            print(batch_data["sentence1_word"])
            print(batch_data["sentence1_char"].shape)
            print(batch_data["sentence2_word"])
            print(batch_data["sentence2_char"].shape)
            print(num_clipped_seq)
            print(num_clipped_chars)
    print(read_end)
    print(batch_data["label"].shape)
    print(batch_data["sentence1_word"].shape)
    print(batch_data["sentence1_char"].shape)
    print(batch_data["sentence2_word"].shape)
    print(batch_data["sentence2_char"].shape)
    print(num_clipped_seq)
    print(num_clipped_chars)
