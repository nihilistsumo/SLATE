import re

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import itertools

def make_psg_pair_embeddings(dat, emb_pid_file, emb_vec_file):
    emb_pid_dict = {}
    for l in np.load(emb_pid_file):
        emb_pid_dict[l.split('\t')[0]] = (int(l.split('\t')[1]), int(l.split('\t')[2]), int(l.split('\t')[3]))
    data_mat = []
    for t in dat:
        p1dat = emb_pid_dict[t[1]]
        p1emb = list(range(p1dat[1], p1dat[1] + p1dat[2]))
        p2dat = emb_pid_dict[t[2].strip()]
        p2emb = list(range(p2dat[1], p2dat[1] + p2dat[2]))
        data_mat.append([t[0], p1emb, p2emb])
    data_mat = pd.DataFrame(data_mat, columns=['similar', 'p1', 'p2'])
    return data_mat, np.load(emb_vec_file)

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