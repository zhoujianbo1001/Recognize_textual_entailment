"""
The hyperparameters for a model are defined here. 
All paramters and arguments can be changed by calling flags in the command line.
"""

import argparse
import io
import os
import json

parser = argparse.ArgumentParser()

# Add training params
parser.add_argument("--lr", type=float, default=1e-4)

# Add model params
# Regularization
parser.add_argument("--weight_decay", type=float, default=0.0)
# input params
parser.add_argument("--seq_len", type=int, default=48)
parser.add_argument("--chars_len", type=int, default=16)#optional
# word embedding params
parser.add_argument("--emb_dropout_kp", type=float, default=1.0)
parser.add_argument("--emb_train", type=bool, default=True)
parser.add_argument("--vocab_size", type=int, default=100)#
parser.add_argument("--emb_dim", type=int, default=50)#
# optional params: characters embedding params
parser.add_argument("--use_char_emb", type=bool, default=False)
parser.add_argument("--chars_dropout_kp", type=float, default=0.8)
parser.add_argument("--chars_vocab_size", type=int, default=50)#
parser.add_argument("--chars_emb_dim", type=int, default=50)#
parser.add_argument("--chars_filters_out_channels", type=list, default=[100])#
parser.add_argument("--chars_filters_width", type=list, default=[5])#
parser.add_argument("--chars_out_size", type=int, default=100)#
# using pos
parser.add_argument("--use_pos", type=bool, default=False)
# using extract match
parser.add_argument("--use_em", type=bool, default=False)
# highway network params
parser.add_argument("--hn_num_layers", type=int, default=1)
parser.add_argument("--hn_dropout_kp", type=float, default=1.0)
parser.add_argument("--hn_out_size", type=int, default=50)
# optional params: self attention params
parser.add_argument("--use_self_att", type=bool, default=True)
parser.add_argument("--self_attention_layers", type=int, default=1)
parser.add_argument("--func_self_att", type=str, default="dot-product")
# optional params: fully connected network
parser.add_argument("--use_fcn", type=bool, default=True)
parser.add_argument("--fcn_out_size", type=int, default=3)
parser.add_argument("--fcn_num_layers", type=int, default=2)
parser.add_argument("--fcn_dropout_kp", type=float, default=1.0)
# optional params: dense network params
parser.add_argument("--use_dn", type=bool, default=False)
parser.add_argument("--dn_out_size", type=bool, default=3)
parser.add_argument("--dn_first_scale_down_ratio", type=float, default=1.0)
parser.add_argument("--dn_first_scale_down_filter", type=int , default=1)
parser.add_argument("--dn_num_blocks", type=int, default=3)
parser.add_argument("--dn_grow_rate", type=int, default=20)
parser.add_argument("--dn_num_block_layers", type=int, default=8)
parser.add_argument("--dn_filter_height", type=int, default=3)
parser.add_argument("--dn_filter_width", type=int, default=3)
parser.add_argument("--dn_transition_rate", type=float, default=0.5)
# optional params: relational network params
parser.add_argument("--use_rn", type=bool, default=False)

def load_configs():
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    load_configs()

