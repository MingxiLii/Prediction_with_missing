
from STGCN import STGCN_ChebConv
import numpy as np
from attnGN import load_metr_LA,load_PEMSD8,load_seattle,load_PEMS
import torch
from GN.utils_tool import Evaluation
import random
import torch.optim as optim
import wandb
import argparse
import pandas as pd
from selfAttentionImpute.utils import masked_mae_cal, masked_rmse_cal,masked_mre_cal
parser = argparse.ArgumentParser()
### 16*9 in total
###dataset "LA", 'seattle',"PEMS", "D8"
### for each dataset and missRate, record the best 16
parser.add_argument('--dataset',type=str,default='LA')
parser.add_argument('--exp_id',type=int,default=1)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--pre_len', type=int, default=9)
### lr 0.01 0.001 0.0001
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.98)
### decay 0.00001 0.000001 0.0000001
parser.add_argument("--decay",type=float, default=0.00001)
parser.add_argument('--use_model', type=str, default="STGCN")
parser.add_argument('--directory',type=str, default='datasets/Metr-LA-data-set/',help= "datasets/Seattle_loop-data-set/,datasets/PeMS-data-set/,datasets/PeMSD8/,datasets/Metr-LA-data-set/")
###missRate  0.2,0.4,0.6,0.8
parser.add_argument('--missRate', type=float, default=0.6)
parser.add_argument('--missMode', type=str, default='PM', help="CM or PM")
parser.add_argument('--datatype', type=str, default='flow', help="flow or speed")
parser.add_argument('--lossweight', type=float, default=0.1)
parser.add_argument('--stblock_num', type=int, default=2)
parser.add_argument('--Ko', type=int, default=4)
parser.add_argument('--device', type= str, default="cuda:1")
args = parser.parse_args()

wandb.init(project="STGCN_baseline", config=args)
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

### load parking data
print(train_x.shape)
tr_len = train_x.shape[0]
test_len = ori_te_X.shape[0]

device = args.device
num_processing_steps= 15

stblock_num = args.stblock_num
blocks = []
blocks.append([1])
for l in range(stblock_num):
    blocks.append([64, 16, 64])
if args.Ko == 0:
    blocks.append([128])
elif args.Ko > 0:
    blocks.append([128, 128])
blocks.append([args.pre_len])
stgcn_chebconv= STGCN_ChebConv(2, 2, blocks, 12, adj.shape[0], 'glu', 'chebconv',torch.from_numpy(adj).float().to(device), int(0)).to(device)
model = stgcn_chebconv
model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= args.momentum)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

for epoch in range(args.epoch):
    model.train()
    random_selected_idxs = random.sample(range(0, tr_len), num_processing_steps)
    node_attr = np.nan_to_num(train_x)
    input_x = torch.tensor(node_attr[random_selected_idxs, :, :], dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2)
    output_tensors = model(input_x.to(device)).squeeze(2).transpose(2,1)
    loss_seq = [torch.sum((output - torch.tensor(train_y[random_selected_idxs[step_t], :,:],dtype=torch.float32, device=device)) ** 2)
                    for step_t, output in enumerate(output_tensors)]
    loss = sum(loss_seq) / len(loss_seq)
    optimizer.zero_grad()
    print(loss.item())
    loss.backward()
    optimizer.step()


model.eval()
with torch.no_grad():
    MAE, MAPE, RMSE = [], [], []
    imputedMAE, imputedMRE, imputedRMSE = [], [], []
    for tt in range(0, test_len - num_processing_steps):
        node_attr = np.nan_to_num(test_x)
        input_x = torch.tensor(node_attr[tt:tt+num_processing_steps, :, :], dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2)
        output_tensors = model(input_x.to(device)).squeeze(2).transpose(2, 1)
        te_loss_seq = [torch.sum((output - torch.tensor(test_y[tt + step_t, :, :],
                                                         dtype=torch.float32, device=device)) ** 2)
                        for step_t, output in enumerate(output_tensors)]
        te_loss = sum(te_loss_seq) / len(te_loss_seq)
        print(te_loss.item())
        mae, mape, rmse = Evaluation.total(scaler.inverse_transform(output_tensors[0].cpu().numpy().reshape(-1, 1)),
                                           scaler.inverse_transform(test_y[tt].reshape(-1, 1)))

        MAE.append(mae)
        MAPE.append(mape)
        RMSE.append(rmse)

    wandb.log({'MAE': np.sum(MAE) / len(MAE)})
    wandb.log({'MAPE': np.sum(MAPE) / len(MAPE)})
    wandb.log({'RMSE': np.sum(RMSE) / len(RMSE)})




