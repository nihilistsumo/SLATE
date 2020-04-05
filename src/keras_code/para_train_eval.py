import sys
sys.path.append('..')
sys.path.append('.')
from time import time
import matplotlib
import argparse
matplotlib.use('Agg')
from keras.models import load_model
import numpy as np
import json
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# File paths
TEST_TSV = '/home/sumanta/Documents/SiameseLSTM_data/by1test-tiny.tsv'
TEST_EMB_PIDS = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1test/bert-base-passage-wiki-sec-mean-sentwise/paraids_sents.npy'
TEST_EMB_VECS_DIR = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1test/bert-base-passage-wiki-sec-mean-sentwise'
EMB_FILE_PREFIX = 'bert-base-wikipedia-sections-mean-tokens-passage'
EMB_BATCH = 10000

class SentbertParaEmbedding():
    def __init__(self, paraid_file, embed_dir, embed_file_prefix, batch_size):
        self.para_info = None
        self.paraids = list(np.load(paraid_file))
        if('\t' in self.paraids[0]):
            self.para_info = self.paraids
            self.paraids = [p.split('\t')[0] for p in self.paraids]
        self.emb_dir = embed_dir
        self.prefix = embed_file_prefix
        self.batch_size = batch_size
        self.curr_para_emb = None
        self.part = -1

    def get_embeddings_as_dict(self, paraid_list):
        print('Going to retrieve embeddings for ' + str(len(paraid_list)) + ' paras')
        emb_dict = dict()
        #part = -1
        for p in paraid_list:
            emb_dict[p] = self.get_single_embedding(p)
        return emb_dict

    def get_single_embedding(self, paraid):
        #part = -1
        if paraid in self.paraids:
            p_index = self.paraids.index(paraid)
        else:
            print(paraid + ' not found in embedding dir')
            return None
        curr_part = p_index // self.batch_size + 1
        offset = p_index % self.batch_size
        if curr_part == self.part:
            emb_vec = self.curr_para_emb[offset]
        else:
            self.curr_para_emb = np.load(self.emb_dir + '/' + self.prefix + '-part' + str(curr_part) + '.npy')
            emb_vec = self.curr_para_emb[offset]
            self.part = curr_part
        return emb_vec

    def get_single_sent_embedding(self, paraid):
        if paraid in self.paraids:
            p_index = self.paraids.index(paraid)
            p_part = int(self.para_info[p_index].split('\t')[1])
            p_offset = int(self.para_info[p_index].split('\t')[2])
            p_len = int(self.para_info[p_index].split('\t')[3])
        else:
            print(paraid + ' not found in embedding dir')
            return None
        curr_part = p_part
        if curr_part == self.part:
            emb_vec = []
            for i in range(p_len):
                emb_vec.append(self.curr_para_emb[p_offset + i])
        else:
            self.curr_para_emb = np.load(self.emb_dir + '/' + self.prefix + '-part' + str(curr_part) + '.npy')
            emb_vec = []
            for i in range(p_len):
                emb_vec.append(self.curr_para_emb[p_offset + i])
            self.part = curr_part
        return emb_vec

    def get_sent_embeddings_as_dict(self, paraid_list):
        emb_dict = dict()
        for pid in paraid_list:
            emb_dict[pid] = self.get_single_sent_embedding(pid)
        return emb_dict


def make_psg_pair_embeddings(dat, emb_pid_file, emb_vec_dir, emb_file_prefix, batch_size, max_seq_len):
    emb_pid_dict = {}
    emb_pid_list = np.load(emb_pid_file)
    for l in emb_pid_list:
        emb_pid_dict[l.split('\t')[0]] = (int(l.split('\t')[1]), int(l.split('\t')[2]), int(l.split('\t')[3]))
    sent_embed = SentbertParaEmbedding(emb_pid_file, emb_vec_dir, emb_file_prefix, batch_size)
    data_mat = []
    parapairs = []
    # emb_start_index = 0
    print('Going to embed '+str(len(dat))+' parapair samples')
    emblen = 768
    c = 0
    labels = []
    for t in dat:
        # we have to make the embeddings matrix on the fly ( 2nd parameter returned from this)
        p1 = t[1]
        # p1dat = emb_pid_dict[p1]
        p1vec = np.array(sent_embed.get_single_sent_embedding(p1))
        p1vec_len = p1vec.shape[0]
        if p1vec_len == 0:
            print('Empty vec returned for '+p1+', using zero vec')
            p1emb = np.zeros((max_seq_len, emblen))
        elif p1vec_len < max_seq_len:
            p1emb = np.vstack((np.zeros((max_seq_len - p1vec_len, emblen)), p1vec))
        else:
            p1emb = p1vec[:max_seq_len]

        p2 = t[2].strip()
        # p2dat = emb_pid_dict[p2]
        p2vec = np.array(sent_embed.get_single_sent_embedding(p2))
        p2vec_len = p2vec.shape[0]
        if p2vec_len == 0:
            print('Empty vec returned for '+p2+', using zero vec')
            p2emb = np.zeros((max_seq_len, emblen))
        elif p2vec_len < max_seq_len:
            p2emb = np.vstack((np.zeros((max_seq_len - p2vec_len, emblen)), p2vec))
        else:
            p2emb = p2vec[:max_seq_len]
        p1p2emb = np.hstack((p1emb, p2emb))
        labels.append(t[0])
        data_mat.append(p1p2emb)
        parapairs.append(p1+'_'+p2)
        c += 1
        if c % (len(dat) // 20) == 0:
            print(str(c)+' samples embedded')
    data_mat = np.array(data_mat)
    labels = np.array(labels)
    return labels, data_mat, parapairs

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def eval(TEST_TSV, TEST_EMB_PIDS, TEST_EMB_DIR, EMB_PREFIX, EMB_BATCH_SIZE, MODEL_PATH, parapair_score_path):
    # Model variables
    embedding_dim = 768
    max_seq_length = 20
    gpus = 2
    batch_size = 1024 * gpus
    n_hidden = 64

    # Define the shared model
    x = Sequential()

    x.add(LSTM(n_hidden))

    shared_model = x

    # The visible layer
    left_input = Input(shape=(max_seq_length, embedding_dim,), dtype='float32')
    right_input = Input(shape=(max_seq_length, embedding_dim,), dtype='float32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    # if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    # model = tf.keras_code.utils.multi_gpu_model(model, gpus=gpus)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()
    model.load_weights(MODEL_PATH)

    test_dat = []
    with open(TEST_TSV, 'r') as tt:
        first = True
        for l in tt:
            if first:
                first = False
                continue
            test_dat.append([int(l.split('\t')[0]), l.split('\t')[1], l.split('\t')[2]])

    # Make word2vec embeddings
    embedding_dim = 768
    max_seq_length = 20
    batch_size = 10000
    num_batch = len(test_dat) // batch_size
    Y_test_all = []
    yhat = []
    all_pairs = []
    for b in range(num_batch):
        Y_test, X_test, test_pairs = make_psg_pair_embeddings(test_dat[b*batch_size:b*batch_size+batch_size],
                                                              TEST_EMB_PIDS, TEST_EMB_DIR, EMB_PREFIX, EMB_BATCH_SIZE,
                                                              max_seq_length)
        model.evaluate([X_test[:, :, :embedding_dim], X_test[:, :, embedding_dim:]], Y_test)
        yhat += [n[0] for n in model.predict([X_test[:, :, :embedding_dim], X_test[:, :, embedding_dim:]])]
        Y_test_all += list(Y_test)
        all_pairs += test_pairs
    Y_test, X_test, test_pairs = make_psg_pair_embeddings(test_dat[num_batch * batch_size:],
                                                          TEST_EMB_PIDS, TEST_EMB_DIR, EMB_PREFIX, EMB_BATCH_SIZE,
                                                          max_seq_length)
    model.evaluate([X_test[:, :, :embedding_dim], X_test[:, :, embedding_dim:]], Y_test)
    yhat += [n[0] for n in model.predict([X_test[:, :, :embedding_dim], X_test[:, :, embedding_dim:]])]
    Y_test_all += list(Y_test)
    all_pairs += test_pairs
    test_pair_scores = {}
    for i in range(len(yhat)):
        test_pair_scores[all_pairs[i]] = float(yhat[i])
    with open(parapair_score_path, 'w') as pps:
        json.dump(test_pair_scores, pps)
    print('BY1test AUC: '+str(roc_auc_score(Y_test_all, yhat)))

def main():
    parser = argparse.ArgumentParser(description='Train Siamese LSTM model for passage similarity')
    parser.add_argument('-te', '--test_dat', help='Path to test data in bert seq format', default=TEST_TSV)
    parser.add_argument('-ti', '--test_emb_pid', help='Path to test emb paraids', default=TEST_EMB_PIDS)
    parser.add_argument('-tv', '--test_emb_dir', help='Path to test emb dir', default=TEST_EMB_VECS_DIR)
    parser.add_argument('-pre', '--emb_prefix', help='Embedding file prefix', default=EMB_FILE_PREFIX)
    parser.add_argument('-bn', '--batch_size', help='Batch size of each embedding file shard', default=EMB_BATCH)
    parser.add_argument('-os', '--out_score', help='Path to save the parapair score file for test', default='../data/test_parapair.json')
    parser.add_argument('-mp', '--model_path', help='Path to save the model', default='../data/SiameseLSTM.h5')
    args = vars(parser.parse_args())
    test_file = args['test_dat']
    test_emb_pid = args['test_emb_pid']
    test_emb_vec_dir = args['test_emb_dir']
    prefix = args['emb_prefix']
    batch = int(args['batch_size'])
    outscore = args['out_score']
    model_path = args['model_path']
    eval(test_file, test_emb_pid, test_emb_vec_dir, prefix, batch, model_path, outscore)

if __name__ == '__main__':
    main()