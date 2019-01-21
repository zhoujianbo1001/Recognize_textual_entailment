"""
The hyperparameters for a model are defined here. 
All paramters and arguments can be changed by calling flags in the command line.
"""

import argparse
import sys
import os
CURRENT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_PATH)
sys.path.append(os.path.dirname(CURRENT_DIR))

####################################################################
# data path
DATA_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))+"/data"
WORDVEC_FILE = DATA_PATH + "/glove.6B/glove.6B.300d.txt"
NUM_WORDVEC = 400000

DATA_PROCESSED_PATH = DATA_PATH+"/data_processed"

WORDS_EMBEDDING_PATH = DATA_PROCESSED_PATH+"/words_embedding.txt"
WORDS_DICT_PATH = DATA_PROCESSED_PATH+"/words_dictionary.txt"
CHARS_DICT_PATH = DATA_PROCESSED_PATH+"/chars_dictionary.txt"

snli_train_path = DATA_PATH+"/snli_1.0/snli_1.0_train.jsonl"
snli_dev_path = DATA_PATH+"/snli_1.0/snli_1.0_dev.jsonl"
snli_test_path = DATA_PATH+"/snli_1.0/snli_1.0_test.jsonl"
num_snli_train_data = 550152
num_snli_dev_data = 10000
num_snli_test_data = 10000

snli_train_encoded_path = DATA_PROCESSED_PATH+"/snli_train_encoded.jsonl"
snli_dev_encoded_path = DATA_PROCESSED_PATH+"/snli_dev_encoded.jsonl"
snli_test_encoded_path = DATA_PROCESSED_PATH+"/snli_test_encoded.jsonl"

####################################################################
# saved model path
CKPT_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR)) + "/ckpt"

snli_ckpt_dir = CKPT_PATH + "/snli"
snli_ckpt_path = snli_ckpt_dir+"/snli.ckpt"
snli_ckpt_reader = snli_ckpt_dir+"/snli_ckpt_reader.txt"

snli_saved_model_dir = os.path.dirname(os.path.dirname(CURRENT_DIR)) + "/snli_saved_model"

snli_train_log = os.path.dirname(CURRENT_DIR) + "/snli_train.log"

#####################################################################
# model params
snli_vocab_size = 26615
multiNLI_vocab_size = 0 #
snli_multiNLI_vocab_size = 0 #

snli_chars_size = 60
multiNLI_chars_size = 0 #
snli_multiNLI_chars_size = 0 #

EMB_DIM = 300

parser = argparse.ArgumentParser()

##################################################################
# Add data process params.
parser.add_argument("--wordvec_file", type=str, default=WORDVEC_FILE)

parser.add_argument("--words_embedding_file", type=str, default=WORDS_EMBEDDING_PATH)
parser.add_argument("--num_wordvec", type=str, default=NUM_WORDVEC)

parser.add_argument("--words_dict_file", type=str, default=WORDS_DICT_PATH)
parser.add_argument("--chars_dict_file", type=str, default=CHARS_DICT_PATH)

parser.add_argument("--train_raw_path", type=str, default=snli_train_path)
parser.add_argument("--dev_raw_path", type=str, default=snli_dev_path)
parser.add_argument("--test_raw_path", type=str, default=snli_test_path)
parser.add_argument("--num_train_data", type=int, default=num_snli_train_data)
parser.add_argument("--num_dev_data", type=int, default=num_snli_dev_data)
parser.add_argument("--num_test_data", type=int, default=num_snli_test_data)

parser.add_argument("--train_encoded_path", type=str, default=snli_train_encoded_path)
parser.add_argument("--dev_encoded_path", type=str, default=snli_dev_encoded_path)
parser.add_argument("--test_encoded_path", type=str, default=snli_test_encoded_path)
##################################################################
# Add training params
parser.add_argument("--train_path", type=str, default=snli_train_encoded_path)
parser.add_argument("--ckpt_dir", type=str, default=snli_ckpt_dir)
parser.add_argument("--ckpt_path", type=str, default=snli_ckpt_path)
parser.add_argument("--ckpt_reader", type=str, default=snli_ckpt_reader)
parser.add_argument("--train_log", type=str, default=snli_train_log)
parser.add_argument("--saved_model_dir", type=str, default=snli_saved_model_dir)

parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--report_interval", type=int, default=200)

##################################################################
# Add testing params
parser.add_argument("--dev_path", type=str, default=snli_dev_encoded_path)
parser.add_argument("--test_path", type=str, default=snli_test_encoded_path)

parser.add_argument("--dev_batch_size", type=int, default=120)
parser.add_argument("--test_batch_size", type=int, default=120)

######################################################################
# Add model params
# Regularization
parser.add_argument("--weight_decay", type=float, default=0.0)

# input params
parser.add_argument("--seq_len", type=int, default=48)
parser.add_argument("--chars_len", type=int, default=16)#optional

parser.add_argument("--label_size", type=int, default=4)
# word embedding params
parser.add_argument("--emb_dropout_kp", type=float, default=0.7)
parser.add_argument("--emb_train", type=bool, default=True)
parser.add_argument("--vocab_size", type=int, default=snli_vocab_size)# Don't include PADDING and UNKNOWN
parser.add_argument("--emb_dim", type=int, default=EMB_DIM)#
# optional params: characters embedding params
parser.add_argument("--use_char_emb", type=bool, default=False)
parser.add_argument("--chars_dropout_kp", type=float, default=0.8)
parser.add_argument("--chars_vocab_size", type=int, default=snli_chars_size)# Don't include PADDING
parser.add_argument("--chars_emb_dim", type=int, default=EMB_DIM)#
parser.add_argument("--chars_filters_out_channels", type=list, default=[100])#
parser.add_argument("--chars_filters_width", type=list, default=[5])#
parser.add_argument("--chars_out_size", type=int, default=100)#
# using pos
parser.add_argument("--use_pos", type=bool, default=False)
# using extract match
parser.add_argument("--use_em", type=bool, default=False)
# highway network params
parser.add_argument("--hn_num_layers", type=int, default=1)
parser.add_argument("--hn_dropout_kp", type=float, default=0.7)
parser.add_argument("--hn_out_size", type=int, default=EMB_DIM)
# optional params: self attention params
parser.add_argument("--use_self_att", type=bool, default=True)
parser.add_argument("--self_attention_layers", type=int, default=1)
parser.add_argument("--func_self_att", type=str, default="dot-product")
# optional params: fully connected network
parser.add_argument("--use_fcn", type=bool, default=True)
parser.add_argument("--fcn_num_layers", type=int, default=3)
parser.add_argument("--fcn_func_activation", type=str, default="relu")
parser.add_argument("--fcn_dropout_kp", type=float, default=1.0)
# optional params: dense network params
parser.add_argument("--use_dn", type=bool, default=False)
parser.add_argument("--dn_first_scale_down_ratio", type=float, default=1.0)
parser.add_argument("--dn_first_scale_down_filter", type=int , default=1)
parser.add_argument("--dn_num_blocks", type=int, default=3)
parser.add_argument("--dn_grow_rate", type=int, default=20)
parser.add_argument("--dn_num_block_layers", type=int, default=8)
parser.add_argument("--dn_filter_height", type=int, default=3)
parser.add_argument("--dn_filter_width", type=int, default=3)
parser.add_argument("--dn_transition_rate", type=float, default=0.5)
parser.add_argument("--dn_dropout_kp", type=float, default=0.7)
# optional params: relational network params
parser.add_argument("--use_rn", type=bool, default=False)
parser.add_argument("--rn_dropout_kp", type=float, default=1.0)

def load_configs():
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    CONFIGS = load_configs()
    print(type(CONFIGS))

