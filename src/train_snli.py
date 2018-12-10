# -*- coding: utf-8 -*-
import sys
import os
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
import tensorflow as tf
import numpy as np
import sklearn as sl
from models.DIIN import DIIN
from utils.read_data import read_embedding_table, read_data, read_batch_data,\
snli_train_path, snli_dev_path, snli_test_path, read_vocab_size, read_chars_vocab_size

############################
# train params
epoch = 10000
batch_size = 60
is_train = True
dropout_p = 0.8

report_interval = 50

ckpt_path = CURRENT_PATH+"/ckpt/snli.ckpt"
ckpt_dir = CURRENT_PATH+"/ckpt"

############################
# model params
emb_train=True
embeddings=None
vocab_size=100 ##
emb_dim=50 ##
chars_vocab_size=50 ##
chars_emb_dim=50 ##
filters_out_channels=[100]
filters_width=[5]
char_out_size=100
weight_decay=0.0
highway_num_layers=2
self_attention_layers=1
label_size=4

seq_len=48
chars_len=16

def train_snli():
    snli_log = open(CURRENT_PATH+"/train_snli.log", 'w')

    # create compute graph.
    embeddings = read_embedding_table()
    vocab_size = embeddings.shape[0]# Don't include PADDING and UNKNOWN
    emb_dim = embeddings.shape[-1]

    chars_vocab_size = read_chars_vocab_size()# Don't include PADDING
    chars_emb_dim = emb_dim

    diin = DIIN(
        emb_train, embeddings, vocab_size, emb_dim, chars_vocab_size, chars_emb_dim,
        filters_out_channels, filters_width, char_out_size, 
        weight_decay, highway_num_layers, self_attention_layers, label_size)
    diin.build_graph()
    diin.build_loss()
    diin.build_train_op()

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if tf.train.get_checkpoint_state(ckpt_dir):
            print("Checkpoint exists.")
            print(tf.train.latest_checkpoint(ckpt_dir))
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print("Checkpoint does't exist.")
            sess.run(tf.global_variables_initializer())
        # read data set
        data_obj = read_data(snli_dev_path)

        current_epoch = 0
        current_batch = 1
        total_losses = 0
        while(current_epoch < epoch):
            # read batch data
            batch_data, read_end = read_batch_data(data_obj, batch_size)
            if read_end:
                current_epoch += 1
                current_batch = 0
            else:
                current_batch += 1
            ###########################
            premise_pos = np.random.randint(low=0, high=100, size=(batch_size, 10, 47), dtype=np.int32)
            hypothesis_pos = np.random.randint(low=0, high=100, size=(batch_size, 10, 47), dtype=np.int32)
            premise_exact_match = np.random.randint(low=0, high=2, size=(batch_size, 10, 1), dtype=np.int32)
            hypothesis_exact_match = np.random.randint(low=0, high=2, size=(batch_size, 10,1), dtype=np.int32)
            #############################

            ################## Words processing ######################
            def pad_words(sentence_word):
                if sentence_word.shape[-1] < seq_len:
                    pad_len = seq_len-sentence_word.shape[-1]
                    sentence_word = np.pad(
                        sentence_word, ((0,0),(0,pad_len)), 
                        "constant")
                return sentence_word

            batch_data["sentence1_word"] = pad_words(batch_data["sentence1_word"])
            batch_data["sentence2_word"] = pad_words(batch_data["sentence2_word"])
            print(batch_data["sentence1_word"].shape)

            ################## Characters process #####################
            def pad_chars(sentence_char):
                if sentence_char.shape[-1] < chars_len:
                    pad_word = seq_len - sentence_char.shape[1]
                    pad_char = chars_len - sentence_char.shape[-1]
                    sentence_char = np.pad(
                        sentence_char, ((0,0),(0,pad_word),(0,pad_char)), 
                        "constant", constant_values=(0,0))
                return sentence_char

            batch_data["sentence1_char"] = pad_chars(batch_data["sentence1_char"])
            batch_data["sentence2_char"] = pad_chars(batch_data["sentence2_char"])

            losses, predict, global_step, debug = diin.update(
                sess, batch_data["label"],
                batch_data["sentence1_word"],
                batch_data["sentence2_word"],
                batch_data["sentence1_char"],
                batch_data["sentence2_char"],
                premise_pos, hypothesis_pos,
                premise_exact_match, hypothesis_exact_match,
                is_train, dropout_p)
            

            if current_batch % report_interval == 0:
                # save model
                saver.save(sess, ckpt_path, global_step=global_step)

                predict_label = np.argmax(predict, axis=-1)
                accuracy = sl.metrics.accuracy_score(batch_data["label"], predict_label)
                average_accuracy = np.mean(accuracy)
                precise = sl.metrics.precision_score(batch_data["label"], predict_label, average="micro")
                average_precise = np.mean(precise)
                recall = sl.metrics.recall_score(batch_data["label"], predict_label, average='micro')
                average_recall = np.mean(recall)
                f1 = sl.metrics.f1_score(batch_data["label"],predict_label, average='micro')
                average_f1 = np.mean(f1)
                mean_losses = total_losses / report_interval
                print("Epoch_idx: {0}, Batch_idx: {1}, Mean_losses: {2}".format(current_epoch, current_batch, mean_losses))
                print("Accuracy: {:.2f}, Precise: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(average_accuracy, average_precise, average_recall, average_f1))
                print(
                    "Epoch_idx: {0}, Batch_idx: {1}, Mean_losses: {2}".format(current_epoch, current_batch, mean_losses), 
                    file=snli_log)
                print(
                    "Accuracy: {:.2f}, Precise: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(average_accuracy, average_precise, average_recall, average_f1),
                    file=snli_log)

                total_losses = 0
            else:
                total_losses += losses


if __name__ == "__main__":
    train_snli()