import torch
import torch.nn as nn
import math
from config import *
from torch.nn.parameter import Parameter


class FeatureMLP(nn.Module):
    def __init__(self, input_dim=MLP_IN_DIM, hidden_dim=MLP_HIDDEN_DIM, output_dim=MLP_OUTPUT_DIM): 
        super(FeatureMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(GCN_FEATURE_DIM, GCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.dropout1 = nn.Dropout(p=dropout_rate) 
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.dropout2 = nn.Dropout(p=dropout_rate) 
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, adj):

        x = self.gc1(x, adj)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  
        
        x = self.gc2(x, adj)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  
        
        return x


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.mlp = FeatureMLP()
        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM, NUM_CLASSES)
        self.criterion = nn.MSELoss()
        weight_p, bias_p = [],[]
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': WEIGHT_DECAY},
                                            {'params': bias_p, 'weight_decay': 0}
                                            ], lr=LEARNING_RATE)        
       
 
    def forward(self, x, adj):  			
        x = x.float()        
        ogt_feature = x[:,:, -1]
        manual_features = x[:,:, :TRADITION_SIZE]   
        protein_features = x[:,:, TRADITION_SIZE:-1]
        assert protein_features.size(2) == 1280, f"MLP input wrong: {protein_features.size()}"
        processed_protein = self.mlp(protein_features)        
        x = torch.cat([manual_features, processed_protein, ogt_feature.unsqueeze(-1)], dim=2)
        assert x.size(2) == TRADITION_SIZE+MLP_OUTPUT_DIM+1, f"Wrong: {x.size()}"
        x = self.gcn(x, adj)  												# x.shape = (seq_len, GAT_OUTPUT_DIM)
        l = len(x.shape)
        if(l<3):
            x = x.unsqueeze(0).float()  										# x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x 									# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        output = torch.sigmoid(self.fc_final(node_feature_embedding_avg))  	# output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)