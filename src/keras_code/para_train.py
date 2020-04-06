import sys
sys.path.append('..')
sys.path.append('.')
from time import time
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Dot, Bidirectional
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score
import random
import json

# File paths
TRAIN_TSV = '/home/sumanta/Documents/SiameseLSTM_data/by1train-discrim-bal-tiny.tsv'
TEST_TSV = '/home/sumanta/Documents/SiameseLSTM_data/by1test-tiny.tsv'
TRAIN_EMB_PIDS = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean-sentwise/paraids_sents.npy'
TRAIN_EMB_VECS_DIR = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean-sentwise'
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

# def make_psg_pair_embeddings(dat, emb_pid_file, emb_vec_file):
#     emb_pid_dict = {}
#     emb_pid_list = np.load(emb_pid_file)
#     for l in emb_pid_list:
#         emb_pid_dict[l.split('\t')[0]] = (int(l.split('\t')[1]), int(l.split('\t')[2]), int(l.split('\t')[3]))
#     data_mat = []
#     for t in dat:
#         # we have to make the embeddings matrix on the fly ( 2nd parameter returned from this)
#         p1 = t[1]
#         p1dat = emb_pid_dict[p1]
#         p1emb = list(range(p1dat[1], p1dat[1] + p1dat[2]))
#         p2 = t[2].strip()
#         p2dat = emb_pid_dict[p2]
#         p2emb = list(range(p2dat[1], p2dat[1] + p2dat[2]))
#         data_mat.append([t[0], p1emb, p2emb])
#     data_mat = pd.DataFrame(data_mat, columns=['similar', 'p1', 'p2'])
#     return data_mat, np.load(emb_vec_file)

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

def split_and_zero_padding(df, max_seq_length):
    # Split to dicts
    X = {'left': df['p1'], 'right': df['p2']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


#  --

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

class CosineDist(Layer):
    """
        Keras Custom Layer that calculates Cosine Distance.
        """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(CosineDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(CosineDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        a = K.l2_normalize(x[0], axis=-1)
        b = K.l2_normalize(x[1], axis=-1)
        self.result = K.exp(-K.mean(a * b, axis=-1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def train(TRAIN_TSV, TRAIN_EMB_PIDS, TRAIN_EMB_DIR, EMB_PREFIX, EMB_BATCH_SIZE, epochs, model_out_path, plot_path, max_seq_length=20, n_hidden=50):
    # Load training set
    train_dat = []
    with open(TRAIN_TSV, 'r') as tr:
        first = True
        for l in tr:
            if first:
                first = False
                continue
            train_dat.append([int(l.split('\t')[0]), l.split('\t')[1], l.split('\t')[2]])
    test_dat = []

    # Make word2vec embeddings
    embedding_dim = 768
    use_w2v = True

    Y, X, train_pairs = make_psg_pair_embeddings(train_dat, TRAIN_EMB_PIDS, TRAIN_EMB_DIR, EMB_PREFIX, EMB_BATCH_SIZE, max_seq_length)

    # Split to train validation
    validation_size = int(len(X) * 0.1)
    training_size = len(X) - validation_size

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)


    # Make sure everything is ok

    # --

    # Model variables
    gpus = 2
    batch_size = 1024 * gpus

    # Define the shared model
    x = Sequential()

    #x.add(LSTM(n_hidden))
    x.add(Bidirectional(LSTM(n_hidden)))

    shared_model = x

    # The visible layer
    left_input = Input(shape=(max_seq_length, embedding_dim,), dtype='float32')
    right_input = Input(shape=(max_seq_length, embedding_dim,), dtype='float32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    # cos_distance = CosineDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    #if gpus >= 2:
        # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
        #model = tf.keras_code.utils.multi_gpu_model(model, gpus=gpus)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()

    # Start trainings
    training_start_time = time()
    malstm_trained = model.fit([X_train[:, :, :embedding_dim], X_train[:, :, embedding_dim:]], Y_train,
                               batch_size=batch_size, epochs=epochs,
                               validation_data=([X_validation[:, :, :embedding_dim], X_validation[:, :, embedding_dim:]], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (epochs,
                                                            training_end_time - training_start_time))

    model.save_weights(model_out_path)

    # Plot accuracy
    plt.subplot(211)
    if 'accuracy' in malstm_trained.history.keys():
        plt.plot(malstm_trained.history['accuracy'])
        plt.plot(malstm_trained.history['val_accuracy'])
    else:
        plt.plot(malstm_trained.history['acc'])
        plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig(plot_path)

    if 'accuracy' in malstm_trained.history.keys():
        print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
    else:
        print(str(malstm_trained.history['val_acc'][-1])[:6] +
              "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description='Train Siamese LSTM model for passage similarity')
    parser.add_argument('-tr', '--train_dat', help='Path to training data in bert seq format', default=TRAIN_TSV)
    parser.add_argument('-ri', '--train_emb_pid', help='Path to train emb paraids', default=TRAIN_EMB_PIDS)
    parser.add_argument('-rv', '--train_emb_dir', help='Path to train emb dir', default=TRAIN_EMB_VECS_DIR)
    parser.add_argument('-pre', '--emb_prefix', help='Embedding file prefix', default=EMB_FILE_PREFIX)
    parser.add_argument('-bn', '--batch_size', help='Batch size of each embedding file shard', default=EMB_BATCH)
    parser.add_argument('-ep', '--num_epochs', help='Number of epochs to train', default=10)
    parser.add_argument('-om', '--out_model', help='Path to save the model', default='../data/SiameseLSTM.h5')
    parser.add_argument('-op', '--out_plot', help='Path to save the history plot', default='../data/history-graph.png')
    parser.add_argument('-len', '--max_len', help='Maximum seq len to consider')
    parser.add_argument('-hn', '--hidden', help='Number of hiddent units in LSTM')
    args = vars(parser.parse_args())
    train_file = args['train_dat']
    train_emb_pid = args['train_emb_pid']
    train_emb_vec_dir = args['train_emb_dir']
    prefix = args['emb_prefix']
    batch = int(args['batch_size'])
    epochs = int(args['num_epochs'])
    outmodel = args['out_model']
    outplot = args['out_plot']
    max_len = int(args['max_len'])
    hidden = int(args['hidden'])
    train(train_file, train_emb_pid, train_emb_vec_dir, prefix, batch, epochs, outmodel, outplot, max_len, hidden)

if __name__ == '__main__':
    main()