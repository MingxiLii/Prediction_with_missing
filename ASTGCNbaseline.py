import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import wandb
import argparse
import torch.optim as optim
import torch.nn.functional as F
from attnGN import load_metr_LA,load_PEMSD8,load_seattle,load_PEMS
from GN.utils_tool import Evaluation
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention import ASTGCN
from torch_geometric_temporal.nn.recurrent import DCRNN,A3TGCN2
from torch_geometric.utils import dense_to_sparse
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)
        return h
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features,pre_len):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, pre_len)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
parser = argparse.ArgumentParser()
### 16*9 in total
###dataset "LA", 'seattle',"PEMS", "D8"
### for each dataset and missRate, record the best 16
parser.add_argument('--dataset',type=str,default='seattle')
parser.add_argument('--exp_id',type=int,default=1)
parser.add_argument('--use_model', type=str, default="DCRNN")
parser.add_argument('--directory',type=str, default='datasets/Seattle_loop-data-set/',help= "datasets/Seattle_loop-data-set/,datasets/PeMS-data-set/,datasets/PeMSD8/,datasets/Metr-LA-data-set/")
###missRate  0.2,0.4,0.6,0.8
parser.add_argument('--missRate', type=float, default=0.6)
parser.add_argument('--missMode', type=str, default='PM', help="CM or PM")
parser.add_argument('--datatype', type=str, default='flow', help="flow or speed")
parser.add_argument('--device', type= str, default="cuda:0")
parser.add_argument('--pre_len', type= int, default=9)
parser.add_argument('--epoch',type=int, default=15)
args = parser.parse_args()

wandb.init(project="multi_baseline", config=args)
DEVICE =args.device
if args.dataset=="LA":
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask,edge_index,edge_attr, scaler \
        = load_metr_LA("/home/mingxi/deep_learning_implementation/CDE/datasets/Metr-LA-data-set/", args.missRate, args.missMode, True, args.pre_len)
    adj = np.load(args.directory + 'Metr_ADJ.npy')
elif args.dataset == 'seattle':
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler\
        = load_seattle(args.directory,args.missRate, args.missMode,  True, args.pre_len)
    adj = np.load(args.directory + 'Loop_Seattle_2015_A.npy')
elif args.dataset =="PEMS":
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler \
        =load_PEMS(args.directory, args.missRate, args.missMode, True, args.pre_len)
    adj_matrix = pd.read_csv(args.directory + 'weighted_adj.csv', header=None)
    adj = np.array(adj_matrix.values)
else:
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler\
        = load_PEMSD8(args.directory, args.missRate, args.missMode, True, args.datatype, args.pre_len)
    adj = np.load(args.directory + 'Adj.npy')
rows, cols = np.where(adj > 0.5)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=3)
plt.show()
rows, cols = np.where(adj > 0.5)
edges = zip(rows.tolist(), cols.tolist())
edge_index_data = torch.LongTensor(np.array([rows, cols])).to(DEVICE)
train_x = train_x[:,:,np.newaxis,:]
test_x = test_x[:,:,np.newaxis,:]
# ------- train_loader -------
train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
train_target_tensor = torch.from_numpy(train_y).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# ------- test_loader -------
test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
test_target_tensor = torch.from_numpy(test_y).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# print
print('train:', train_x_tensor.size(), train_target_tensor.size())
print('test:', test_x_tensor.size(), test_target_tensor.size())

nb_block = 2
in_channels = 1
K = 3
nb_chev_filter = 64
nb_time_filter = 64
time_strides = 1
num_for_predict = args.pre_len
len_input = 12
num_of_vertices= train_x.shape[1]
#L_tilde = scaled_Laplacian(adj_mx)
#cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
if args.use_model == "ASTGCN":
    net = ASTGCN( nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices).to(DEVICE)
elif args.use_model == "DCRNN":
    net = RecurrentGCN(node_features = len_input, pre_len=args.pre_len).to(DEVICE)
    _, edge_attrs = dense_to_sparse(torch.from_numpy(adj))
    edge_attrs.to(DEVICE)
else:
    net = TemporalGNN(node_features=1, periods=args.pre_len, batch_size=64).to(DEVICE)
print(net)

#------------------------------------------------------
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

print('Net\'s state_dict:')
total_param = 0
for param_tensor in net.state_dict():
    print(param_tensor, '\t', net.state_dict()[param_tensor].size(), '\t', net.state_dict()[param_tensor].device)
    total_param += np.prod(net.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)
#--------------------------------------------------
print('Optimizer\'s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])
# train model
global_step = 0
start_time= time()
criterion = nn.L1Loss().to(DEVICE)
for epoch in range(args.epoch):

    net.train()  # ensure dropout layers are in train mode

    for batch_index, batch_data in enumerate(train_loader):
        encoder_inputs, labels = batch_data   # encoder_inputs torch.Size([32, 307, 1, 12])  label torch.Size([32, 307, 12])
        optimizer.zero_grad()
        if args.use_model == "DCRNN":
            outputs = []
            for i in range(encoder_inputs.shape[0]):
                input = torch.squeeze(encoder_inputs.cpu()[i,:,:,:])
                output = net(input.to(dtype= torch.float32).to(DEVICE), edge_index_data.to(dtype= torch.int64).to(DEVICE), edge_attrs.to(dtype= torch.float32).to(DEVICE))
                outputs.append(output.unsqueeze(0))
            outputs = torch.cat(outputs,dim=0)
        else:
            outputs = net(encoder_inputs, edge_index_data) # torch.Size([32, 307, 12])

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        global_step += 1
        print(training_loss)

        if global_step % 200 == 0:
            print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))
net.train(False)  # ensure dropout layers are in evaluation mode
with torch.no_grad():
    MAE, MAPE, RMSE = [], [], []
    test_loader_length = len(test_loader)  # nb of batch
    tmp = []  # batch loss
    for batch_index, batch_data in enumerate(test_loader):
        encoder_inputs, labels = batch_data
        print(labels.shape)
        if args.use_model == "DCRNN":
            outputs = []
            for i in range(encoder_inputs.shape[0]):
                input = torch.squeeze(encoder_inputs.cpu()[i,:,:,:])
                output = net(input.to(dtype= torch.float32).to(DEVICE), edge_index_data.to(dtype= torch.int64).to(DEVICE), edge_attrs.to(dtype= torch.float32).to(DEVICE))
                outputs.append(output.unsqueeze(0))
            outputs = torch.cat(outputs,dim=0)
            print(outputs.shape)
        else:
            outputs = net(encoder_inputs, edge_index_data)
        loss = criterion(outputs, labels)
        tmp.append(loss.item())
        if batch_index % 100 == 0:
            print('test_loss batch %s / %s, loss: %.2f' % (batch_index + 1, test_loader_length, loss.item()))
        mae, mape, rmse = Evaluation.total(scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)),
                                           scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1)))

        MAE.append(mae)
        MAPE.append(mape)
        RMSE.append(rmse)

    wandb.log({'MAE': np.sum(MAE) / len(MAE)})
    wandb.log({'MAPE': np.sum(MAPE) / len(MAPE)})
    wandb.log({'RMSE': np.sum(RMSE) / len(RMSE)})
    test_loss = sum(tmp) / len(tmp)
    print(test_loss)
