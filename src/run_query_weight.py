from nets.models import Query_Weight_Network, Query_Attn_LL_Network, Siamese_Network
from data import process_qry_attn_data as dat
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate query weighted network for paragraph similarity task')
    parser.add_argument('-nn', '--neural_model', help='1: Query weight, 2: Query attn LL, 3: Siamese')
    parser.add_argument('-pd', '--emb_paraids_file', help='Path to train embedding paraids file/ train embedding json dict')
    parser.add_argument('-pt', '--test_emb_paraids_file', help='Path to test embedding paraids file')
    parser.add_argument('-ed', '--emb_vec_file', help='Path to para embedding vec file for train split paras')
    parser.add_argument('-et', '--emb_vec_test_file', help='Path to para embedding vec file for test split paras')
    parser.add_argument('-ep', '--embedding_model', help='Embedding model name or actual path')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate')
    parser.add_argument('-it', '--num_iteration', help='No. of iteration')
    parser.add_argument('-td', '--train_data_file', help='Path to train data file')
    parser.add_argument('-tt', '--test_data_file', help='Path to test data file')
    parser.add_argument('-o', '--model_outfile', help='Path to save the trained model')
    args = vars(parser.parse_args())
    nn_option = int(args['neural_model'])
    emb_vec = args['emb_vec_file']
    emb_vec_test = args['emb_vec_test_file']
    lrate = float(args['learning_rate'])
    iter = int(args['num_iteration'])
    emb_model = args['embedding_model']
    emb_pids_file = args['emb_paraids_file']
    test_emb_pids_file = args['test_emb_paraids_file']
    train_filepath = args['train_data_file']
    test_filepath = args['test_data_file']
    model_out = args['model_outfile']
    if torch.cuda.is_available():
        device1 = torch.device('cuda:0')
        device2 = torch.device('cuda:1')
    else:
        device1 = torch.device('cpu')
        device2 = device1
    log_out = model_out + '.train.log'

    X, y = dat.get_data(emb_vec, emb_model, emb_pids_file, train_filepath)
    X_val = X[:100, :].cuda(device1)
    y_val = y[:100]
    X_train = X[100:, :].cuda(device1)
    y_train = y[100:].cuda(device1)
    X_test, y_test = dat.get_data(emb_vec_test, emb_model, test_emb_pids_file, test_filepath)
    X_test = X_test.cuda(device1)

    if(nn_option == 1):
        NN = Query_Weight_Network().to(device1)
    elif(nn_option == 2):
        NN = Query_Attn_LL_Network().to(device1)
    elif(nn_option == 3):
        NN = Siamese_Network().to(device1)
    else:
        print('Wrong model option')

    criterion = nn.MSELoss().cuda(device1)
    opt = optim.SGD(NN.parameters(), lr=lrate, weight_decay=0.01)
    print()
    with open(log_out, 'w') as lo:
        for i in range(iter):
            opt.zero_grad()
            output = NN(X_train)
            loss = criterion(output, y_train)
            y_val_pred = NN.predict(X_val).detach().cpu().numpy()
            val_auc_score = roc_auc_score(y_val, y_val_pred)
            sys.stdout.write(
                '\r' + 'Iteration: ' + str(i) + ', loss: ' + str(loss) + ', val AUC: ' + '{:.4f}'.format(val_auc_score))
            loss.backward()
            if i % 10 == 0:
                lo.write('Iteration: ' + str(i) + ', loss: ' + str(loss) + ', val AUC: ' + str(val_auc_score) + '\n')

            opt.step()
    # NN.saveWeights(NN)
    y_pred = NN.predict(X_test).detach().cpu().numpy()
    auc_score = roc_auc_score(y_test, y_pred)
    print()
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(X_test))
    print("Output: " + str(y_pred))
    print('AUC score: ' + str(auc_score))
    # print(NN.parameters())
    # print('True output: ' + str(y_test))
    # print('Features: ' + str(NN.num_flat_features(X_test)))
    print('Saving model at ' + model_out)
    torch.save(NN.state_dict(), model_out)

if __name__ == '__main__':
    main()