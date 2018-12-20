# -*- coding: utf-8 -*-
import sys
import os
CURRENT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_PATH)
import tensorflow as tf
import numpy as np
import sklearn as sl

from models.DIIN import DIIN
from utils.read_data import read_data, read_batch_data
import utils.params as params

CONFIGS = params.load_configs()

def predict_batch_data(sess, model, data_obj, batch_size=1, seq_len=None, chars_len=None):
    batch_data, read_end, read_batch, num_clipped_seq, num_clipped_chars = \
    read_batch_data(data_obj, batch_size, seq_len, chars_len)

    ###########################
    premise_pos = np.random.randint(low=0, high=100, size=(CONFIGS.batch_size, 10, 47), dtype=np.int32)
    hypothesis_pos = np.random.randint(low=0, high=100, size=(CONFIGS.batch_size, 10, 47), dtype=np.int32)
    premise_exact_match = np.random.randint(low=0, high=2, size=(CONFIGS.batch_size, 10, 1), dtype=np.int32)
    hypothesis_exact_match = np.random.randint(low=0, high=2, size=(CONFIGS.batch_size, 10,1), dtype=np.int32)
    #############################

    logits = model.predict(
        sess, batch_data["sentence1_word"], batch_data["sentence2_word"],
        batch_data["sentence1_char"], batch_data["sentence2_char"],
        premise_pos, hypothesis_pos, premise_exact_match, hypothesis_exact_match)

    p = np.argmax(logits, -1)
    accuracy = sl.metrics.accuracy_score(batch_data["label"], p)
    return p, read_end, accuracy

def dev_func(sess, model, data_obj):
    predict_end = False
    accuracy_total = 0
    num_predicted = 0
    while(predict_end == False):
        _, predict_end, accuracy = predict_batch_data(
            sess, model, data_obj, 
            batch_size=CONFIGS.dev_batch_size, 
            seq_len=CONFIGS.seq_len, 
            chars_len=CONFIGS.chars_len)
        # moving average
        accuracy_total = (accuracy*CONFIGS.dev_batch_size+accuracy_total*num_predicted)/(num_predicted+CONFIGS.dev_batch_size)
        num_predicted = num_predicted+CONFIGS.dev_batch_size
    return accuracy_total

def test_data(data_path):
    with tf.Session() as sess:
        # restore model
        meta_graph_def = tf.saved_model.loader.load(
            sess, ["test_saved_model"], 
            CONFIGS.saved_model_dir)

        test_signature = meta_graph_def.signature_def["test_signature"]
        prem_x_name = test_signature.inputs["prem_x"].name
        hyp_x_name = test_signature.inputs["hyp_x"].name
        is_train_name = test_signature.inputs["is_train"].name
        logits_name = test_signature.outputs["logits"].name

        prem_x = sess.graph.get_tensor_by_name(prem_x_name)
        hyp_x = sess.graph.get_tensor_by_name(hyp_x_name)
        is_train = sess.graph.get_tensor_by_name(is_train_name)
        logits = sess.graph.get_tensor_by_name(logits_name)

        # read data
        data_obj = read_data(data_path)
        accuracy_total = 0
        num_predicted = 0
        predict_end = False
        while(predict_end == False):
            batch_data, predict_end, _, _, _ = read_batch_data(
                data_obj, CONFIGS.dev_batch_size, 
                CONFIGS.seq_len, CONFIGS.chars_len)
            p = sess.run(
                logits, 
                {
                    prem_x: batch_data["sentence1_word"], 
                    hyp_x: batch_data["sentence2_word"],
                    is_train: False
                })
            accuracy = sl.metrics.accuracy_score(
                batch_data["label"], np.argmax(p, -1))
            # moving average
            accuracy_total = (accuracy*CONFIGS.dev_batch_size+accuracy_total*num_predicted)/(num_predicted+CONFIGS.dev_batch_size)
            num_predicted = num_predicted+CONFIGS.dev_batch_size

        print("Dev accuracy: ", accuracy_total)

if __name__ == "__main__":
    test_data(CONFIGS.train_encoded_path)
    test_data(CONFIGS.dev_encoded_path)