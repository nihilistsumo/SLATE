import torch
import torch.nn as nn

class Query_Weight_Network(nn.Module):
    def __init__(self, ):
        super(Query_Weight_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.l1_out_size = 128
        self.l2_out_size = 64
        self.l3_out_size = 32
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.l1_out_size).cuda()
        self.LL2 = nn.Linear(self.l1_out_size, self.l2_out_size).cuda()
        self.LL3 = nn.Linear(self.l2_out_size, self.l3_out_size).cuda()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.zq1 = torch.relu(self.LL1(self.dropout(self.Xq)))
        self.zq2 = torch.relu(self.LL2(self.dropout(self.zq1)))
        self.zq3 = torch.relu(self.LL3(self.dropout(self.zq2)))
        self.zq3Xp1 = torch.einsum('bi, bj -> bij', (self.zq3, self.Xp1))
        self.zq3Xp2 = torch.einsum('bi, bj -> bij', (self.zq3, self.Xp2))
        self.zq3Xp1 = self.zq3Xp1.reshape(-1, self.l3_out_size*self.emb_size)
        self.zq3Xp2 = self.zq3Xp2.reshape(-1, self.l3_out_size*self.emb_size)
        o = self.cosine_sim(self.zq3Xp1, self.zq3Xp2)  # final activation function
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred