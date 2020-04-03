from time import time
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout
from ..data.siamese_sent_data import make_psg_pair_embeddings
from ..data.siamese_sent_data import split_and_zero_padding
from ..data.siamese_sent_data import ManDist

# File paths
TRAIN_TSV = '/home/sumanta/Documents/SiameseLSTM_data/by1train-discrim-bal.tsv'
TEST_TSV = '/home/sumanta/Documents/SiameseLSTM_data/by1test.tsv'
TRAIN_EMB_PIDS = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean-sentwise/paraids_sents.npy'
TRAIN_EMB_VECS = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean-sentwise/bert-base-wikipedia-sections-mean-tokens-passage-part1.npy'
TEST_EMB_PIDS = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1test/bert-base-passage-wiki-sec-mean-sentwise/paraids_sents.npy'
TEST_EMB_VECS = '/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1test/bert-base-passage-wiki-sec-mean-sentwise/bert-base-wikipedia-sections-mean-tokens-passage-part1.npy'

def train(TRAIN_TSV, TEST_TSV, TRAIN_EMB_PIDS, TRAIN_EMB_VECS, TEST_EMB_PIDS, TEST_EMB_VECS, model_out_path, plot_path):
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
    use_w2v = True

    train_df, train_embeddings = make_psg_pair_embeddings(train_dat, TRAIN_EMB_PIDS, TRAIN_EMB_VECS)
    test_df, test_embeddings = make_psg_pair_embeddings(test_dat, TEST_EMB_PIDS, TEST_EMB_VECS)

    # Split to train validation
    validation_size = int(len(train_df) * 0.1)
    training_size = len(train_df) - validation_size

    X = train_df[['p1', 'p2']]
    Y = train_df['similar']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    # --

    # Model variables
    gpus = 2
    batch_size = 1024 * gpus
    n_epoch = 50
    n_hidden = 50

    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(train_embeddings), embedding_dim,
                    weights=[train_embeddings], input_shape=(max_seq_length,), trainable=False))

    x.add(LSTM(n_hidden))

    shared_model = x

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    #if gpus >= 2:
        # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
        #model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()

    # Start trainings
    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                            training_end_time - training_start_time))

    # model.save('./data/SiameseLSTM.h5')
    model.save(model_out_path)

    # Plot accuracy
    plt.subplot(211)
    plt.plot(malstm_trained.history['accuracy'])
    plt.plot(malstm_trained.history['val_accuracy'])
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
    # plt.savefig('./data/history-graph.png')
    plt.savefig(plot_path)

    print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description='Train Siamese LSTM model for passage similarity')
    parser.add_argument('-tr', '--train_dat', help='Path to training data in bert seq format', default=TRAIN_TSV)
    parser.add_argument('-te', '--test_dat', help='Path to test data in bert seq format', default=TEST_TSV)
    parser.add_argument('-ri', '--train_emb_pid', help='Path to train emb paraids', default=TRAIN_EMB_PIDS)
    parser.add_argument('-ti', '--test_emb_pid', help='Path to test emb paraids', default=TEST_EMB_PIDS)
    parser.add_argument('-rv', '--train_emb_vec', help='Path to train emb vecs', default=TRAIN_EMB_VECS)
    parser.add_argument('-tv', '--test_emb_vec', help='Path to test emb vecs', default=TEST_EMB_VECS)
    parser.add_argument('-om', '--out_model', help='Path to save the model', default='./data/SiameseLSTM.h5')
    parser.add_argument('-op', '--out_plot', help='Path to save the history plot', default='./data/history-graph.png')
    args = vars(parser.parse_args())
    train_file = args['train_dat']
    test_file = args['test_dat']
    train_emb_pid = args['train_emb_pid']
    test_emb_pid = args['test_emb_pid']
    train_emb_vec = args['train_emb_vec']
    test_emb_vec = args['test_emb_vec']
    outmodel = args['out_model']
    outplot = args['out_plot']
    train(train_file, test_file, train_emb_pid, train_emb_vec, test_emb_pid, test_emb_vec, outmodel, outplot)

if __name__ == '__main__':
    main()