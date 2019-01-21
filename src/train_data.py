# -*- coding: utf-8 -*-
import sys
import os
CURRENT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_PATH)

import shutil
import tensorflow as tf
import numpy as np
import sklearn as sl

from models.DIIN import DIIN, debug_func
from utils.read_data import read_embedding_table, read_data, read_batch_data
from test_data import dev_func

import utils.params as params
CONFIGS = params.load_configs()

############################
is_train = True

ckpt_path = CONFIGS.ckpt_path
ckpt_dir = CONFIGS.ckpt_dir

def train_data():
    # create compute graph.
    embeddings = read_embedding_table()

    diin = DIIN(embeddings)
    diin.build_graph()
    diin.build_loss()
    diin.build_train_op(lr=CONFIGS.lr)

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # model variables restore.
        if tf.train.get_checkpoint_state(ckpt_dir):
            print("Checkpoint exists.")
            print(tf.train.latest_checkpoint(ckpt_dir))
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print("Checkpoint does't exist.")
            sess.run(tf.global_variables_initializer())
        # read data set
        data_obj = read_data(CONFIGS.train_path)

        current_epoch = 0
        current_batch = 0
        total_losses = 0
        num_clipped_seq = 0
        num_clipped_chars = 0
        train_log = open(CONFIGS.train_log, "w")
        while(current_epoch < CONFIGS.epoch):
            # read batch data
            batch_data, read_end, read_batch, clipped_seq, clipped_chars = read_batch_data(
                data_obj, CONFIGS.batch_size, seq_len=CONFIGS.seq_len, chars_len=CONFIGS.chars_len)
            num_clipped_seq += clipped_seq
            num_clipped_chars += clipped_chars

            if read_batch == 0:
                continue

            ###########################
            premise_pos = np.random.randint(
                low=0, high=100, size=(CONFIGS.batch_size, 10, 47), dtype=np.int32)
            hypothesis_pos = np.random.randint(low=0, high=100, size=(CONFIGS.batch_size, 10, 47), dtype=np.int32)
            premise_exact_match = np.random.randint(low=0, high=2, size=(CONFIGS.batch_size, 10, 1), dtype=np.int32)
            hypothesis_exact_match = np.random.randint(low=0, high=2, size=(CONFIGS.batch_size, 10,1), dtype=np.int32)
            #############################

            losses, predict, global_step, debug = diin.update(
                sess, batch_data["label"],
                batch_data["sentence1_word"],
                batch_data["sentence2_word"],
                batch_data["sentence1_char"],
                batch_data["sentence2_char"],
                premise_pos, hypothesis_pos,
                premise_exact_match, hypothesis_exact_match,
                is_train)

            # debug_func(debug, train_log)
            
            if current_batch % CONFIGS.report_interval == 0:
                # save model
                saver.save(sess, ckpt_path)

                report_metrics(
                    predict, batch_data["label"], total_losses, CONFIGS.report_interval, 
                    current_epoch, current_batch, train_log)

                total_losses = 0
                save_model_for_test(diin, sess, CONFIGS.saved_model_dir)
            else:
                total_losses += losses

            if read_end:
                print("num_clipped_seq: {0}, num_clipped_chars: {1}"\
                .format(num_clipped_seq, num_clipped_chars), file=train_log)
                # tag change
                current_epoch += 1
                current_batch = 0
                num_clipped_seq = 0 
                num_clipped_chars = 0
                
                # dev data
                dev_obj = read_data(CONFIGS.dev_path)
                accuracy_total = dev_func(sess, diin, dev_obj)
                print("Dev accuracy: ", accuracy_total, file=train_log)
                print("Dev accuracy: ", accuracy_total)
                # train data
                accuracy_total = dev_func(sess, diin, data_obj)
                print("Train accuracy: ", accuracy_total, file=train_log)
                print("Train accuracy: ", accuracy_total)
            else:
                current_batch += 1
        
        # save model for test
        save_model_for_test(diin, sess, CONFIGS.saved_model_dir)

def report_metrics(
    predict, labels, total_losses, report_interval, 
    current_epoch, current_batch, file_log):
    predict_label = np.argmax(predict, axis=-1)
    accuracy = sl.metrics.accuracy_score(labels, predict_label)
    average_accuracy = np.mean(accuracy)
    precise = sl.metrics.precision_score(labels, predict_label, average="micro")
    average_precise = np.mean(precise)
    recall = sl.metrics.recall_score(labels, predict_label, average='micro')
    average_recall = np.mean(recall)
    f1 = sl.metrics.f1_score(labels, predict_label, average='micro')
    average_f1 = np.mean(f1)
    mean_losses = total_losses / report_interval
    print("Epoch_idx: {0}, Batch_idx: {1}, Mean_losses: {2}".format(current_epoch, current_batch, mean_losses))
    print("Accuracy: {:.2f}, Precise: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(average_accuracy, average_precise, average_recall, average_f1))
    print(
        "Epoch_idx: {0}, Batch_idx: {1}, Mean_losses: {2}".format(current_epoch, current_batch, mean_losses), 
        file=file_log)
    print(
        "Accuracy: {:.2f}, Precise: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(average_accuracy, average_precise, average_recall, average_f1),
        file=file_log)

def save_model_for_test(model, sess, export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    inputs = {
        "prem_x" : tf.saved_model.utils.build_tensor_info(
            model.prem_x),
        "hyp_x" : tf.saved_model.utils.build_tensor_info(
            model.hyp_x),
        "is_train" : tf.saved_model.utils.build_tensor_info(
            model.is_train)
    }
    outputs = {
        "logits" : tf.saved_model.utils.build_tensor_info(
            model.logits)
    }
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs, outputs, "test_signature")
    builder.add_meta_graph_and_variables(
        sess, ["test_saved_model"],
        {"test_signature": signature})
    builder.save()

if __name__ == "__main__":
    train_data()