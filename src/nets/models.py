import torch
import torch.nn as nn

class Query_Weight_Network(nn.Module):
    def __init__(self, ):
        super(Query_Weight_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size).cuda()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.zq1 = torch.relu(self.LL1(self.dropout(self.Xq)))
        self.zq2Xp1 = torch.mul(self.zq1, self.Xp1)
        self.zq2Xp2 = torch.mul(self.zq1, self.Xp2)
        o = self.cosine_sim(self.zq2Xp1, self.zq2Xp2)  # final activation function
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

class Siamese_Network(nn.Module):
    def __init__(self, ):
        super(Siamese_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.l1_out_size = 64
        self.LL1 = nn.Linear(self.emb_size, self.l1_out_size).cuda()
        self.LL2 = nn.Linear(self.l1_out_size, self.l1_out_size).cuda()
        self.LL3 = nn.Linear(2*self.l1_out_size, 1).cuda()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.zp1 = torch.relu(self.LL1(self.dropout(self.Xp1)))
        self.zpp1 = torch.relu(self.LL2(self.dropout(self.zp1)))
        self.zp2 = torch.relu(self.LL1(self.dropout(self.Xp2)))
        self.zpp2 = torch.relu(self.LL2(self.dropout(self.zp2)))
        self.zp = torch.cat((self.zpp1, self.zpp2), dim=1)
        o = self.LL3(self.zp)  # final activation function
        o = o.reshape(-1)
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

class Query_Attn_LL_Network(nn.Module):
    def __init__(self, ):
        super(Query_Attn_LL_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size).cuda()
        self.LL2 = nn.Linear(2*self.emb_size, 1).cuda()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.z = torch.relu(self.LL1(self.dropout(self.Xq)))
        self.sXp1 = torch.mul(self.Xp1, self.z)
        self.sXp2 = torch.mul(self.Xp2, self.z)
        self.sXp = torch.cat((self.sXp1, self.sXp2), dim=1)
        #o = self.cosine_sim(self.sXp1, self.sXp2)  # final activation function
        o = self.LL2(self.sXp)
        o = o.reshape(-1)
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