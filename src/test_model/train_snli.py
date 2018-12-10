import sys
import os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_PATH))
import tensorflow as tf
import parameters as params
from data_processing import *
from models.DIIN import DIIN

import tqdm
import gzip
import pickle
import importlib
import sklearn as sl

FIXED_PARAMETERS, config = params.load_parameters()

######################## LOAD DATASET ################################
training_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
indices_to_words, word_indices, char_indices, indices_to_chars = \
sentences_to_padded_index_sequences([training_snli])

shared_content = load_mnli_shared_content()

config.char_vocab_size = len(char_indices.keys())

embedding_dir = os.path.join(config.datapath, "embeddings")
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)


embedding_path = os.path.join(embedding_dir, "mnli_emb_snli_embedding.pkl.gz")

print("embedding path exist")
print(os.path.exists(embedding_path))
if os.path.exists(embedding_path):
    f = gzip.open(embedding_path, 'rb')
    loaded_embeddings = pickle.load(f)
    f.close()
else:
    loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)
    f = gzip.open(embedding_path, 'wb')
    pickle.dump(loaded_embeddings, f)
    f.close()

def get_minibatch(dataset, start_index, end_index):

    indices = range(start_index, end_index)

    genres = [dataset[i]['genre'] for i in indices]
    labels = [dataset[i]['label'] for i in indices]
    pairIDs = np.array([dataset[i]['pairID'] for i in indices])


    premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(indices)


    premise_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_index_sequence'][:] for i in indices], premise_pad_crop_pair, 1)
    hypothesis_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_index_sequence'][:] for i in indices], hypothesis_pad_crop_pair, 1)
    premise_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_char_index'][:] for i in indices], premise_pad_crop_pair, 2, column_size=config.char_in_word_size)
    hypothesis_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_char_index'][:] for i in indices], hypothesis_pad_crop_pair, 2, column_size=config.char_in_word_size)

    premise_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence1_parse'][:] for i in indices], premise_pad_crop_pair)
    hypothesis_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence2_parse'][:] for i in indices], hypothesis_pad_crop_pair)

    premise_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_token_exact_match_with_s2"][:] for i in range(len(indices))], premise_pad_crop_pair, 1)
    hypothesis_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_token_exact_match_with_s1"][:] for i in range(len(indices))], hypothesis_pad_crop_pair, 1)

    premise_exact_match = np.expand_dims(premise_exact_match, 2)
    hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)


    return premise_vectors, hypothesis_vectors, labels, genres, premise_pos_vectors, \
            hypothesis_pos_vectors, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match
    

####################### CREATE MODEL #################################
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

seq_len=FIXED_PARAMETERS["seq_length"]
chars_len=config.char_in_word_size

snli_log = open(CURRENT_PATH+"/train_snli.log", 'w')

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

    current_epoch = 0
    current_batch = 0
    total_batch = int(len(training_snli)/batch_size)
    total_losses = 0
    while(current_epoch < epoch):
        # read batch data
        minibatch_premise_vectors, minibatch_hypothesis_vectors, \
        minibatch_labels, minibatch_genres, minibatch_pre_pos, minibatch_hyp_pos, \
        pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match  = get_minibatch(
            training_snli, batch_size * current_batch, batch_size * (current_batch + 1))
        current_batch += 1
        if current_batch >= total_batch:
            current_epoch += 1
            current_batch = 0

        losses, predict, global_step, debug = diin.update(
            sess, minibatch_labels,
            minibatch_premise_vectors,
            minibatch_hypothesis_vectors,
            premise_char_vectors,
            hypothesis_char_vectors,
            minibatch_pre_pos, 
            minibatch_hyp_pos,
            premise_exact_match, 
            hypothesis_exact_match,
            is_train, dropout_p)
        

        if current_batch % report_interval == 0:
            # save model
            saver.save(sess, ckpt_path, global_step=global_step)

            predict_label = np.argmax(predict, axis=-1)
            accuracy = sl.metrics.accuracy_score(minibatch_labels, predict_label)
            average_accuracy = np.mean(accuracy)
            precise = sl.metrics.precision_score(minibatch_labels, predict_label, average="micro")
            average_precise = np.mean(precise)
            recall = sl.metrics.recall_score(minibatch_labels, predict_label, average='micro')
            average_recall = np.mean(recall)
            f1 = sl.metrics.f1_score(minibatch_labels,predict_label, average='micro')
            average_f1 = np.mean(f1)
            mean_losses = total_losses / report_interval
            
            print("Epoch_idx: {0}, Batch_idx: {1}, Mean_losses: {2}".format(
                current_epoch, current_batch, mean_losses))
            print("Accuracy: {:.2f}, Precise: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(
                average_accuracy, average_precise, average_recall, average_f1))
            # print("Predict:", predict)
            # print("Debug:", debug)
            print(
                "Epoch_idx: {0}, Batch_idx: {1}, Mean_losses: {2}".format(
                    current_epoch, current_batch, mean_losses),
                file=snli_log)
            print(
                "Accuracy: {:.2f}, Precise: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(
                    average_accuracy, average_precise, average_recall, average_f1),
                file=snli_log)

            total_losses = 0
        else:
            total_losses += losses
