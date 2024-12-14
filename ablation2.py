import numpy as np
from attnGN import selfAttenGN, load_metr_LA,load_PEMSD8,load_seattle,load_PEMS,ablation_2
import torch
from GN.utils_tool import Evaluation
from torch_geometric.data import Data
import random
import torch.optim as optim
import wandb
import argparse
import pandas as pd
from selfAttentionImpute.utils import masked_mae_cal, masked_rmse_cal,masked_mre_cal
from sklearn.impute import MissingIndicator
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='LA')
parser.add_argument('--pre_len', type=int, default= 12)
parser.add_argument('--exp_id',type=int,default=1)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.98)
parser.add_argument('--use_model', type=str, default="ablation_2")
parser.add_argument('--directory',type=str, default='datasets/Metr-LA-data-set/',help= "datasets/Seattle_loop-data-set/,datasets/PeMS-data-set/,datasets/PeMSD8/,datasets/Metr-LA-data-set/")
parser.add_argument('--missRate', type=float, default=0.6)
parser.add_argument('--missMode', type=str, default='PM', help="CM or PM")
parser.add_argument('--datatype', type=str, default='flow', help="flow or speed")
parser.add_argument('--lossweight', type=float, default=0.1)
parser.add_argument('--device', type= str, default="cuda:0")

args = parser.parse_args()

wandb.init(project="traffic_attention_GN_ablation", config=args)
if args.dataset=="LA":
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask,edge_index,edge_attr, scaler \
        = load_metr_LA("/home/mingxi/deep_learning_implementation/CDE/datasets/Metr-LA-data-set/", args.missRate, args.missMode,False, args.pre_len)
    adj = np.load(args.directory + 'Metr_ADJ.npy')

elif args.dataset == 'seattle':
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler\
        = load_seattle(args.directory,args.missRate, args.missMode,False, args.pre_len)
elif args.dataset =="PEMS":
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler \
        =load_PEMS(args.directory, args.missRate, args.missMode,False, args.pre_len)
else:
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler\
        = load_PEMSD8(args.directory, args.missRate, args.missMode,False,args.datatype, args.pre_len)
    print('check')
    print(ori_X.shape)
    print(train_y.shape)
### load parking data

tr_len = ori_X.shape[0]
test_len = ori_te_X.shape[0]

device = args.device

print(ori_X.shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features


epochNum = 3500
num_processing_steps = 10
# model = selfAttenGN(input_channels=1, hidden_channels=64, hidden_channels_2=32, node_attr_size=12, out_size=args.pre_len,  device=device, edge_hidden_size=64 , node_hidden_size=64 ,
#                      global_hidden_size=64)
model = ablation_2(input_channels=1, hidden_channels=1, hidden_channels_2=1, node_attr_size=12, out_size=args.pre_len,  device=device, edge_hidden_size=1 , node_hidden_size=12 ,
                     global_hidden_size=1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00000001)
# optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.95)
maes, mapes, rmses =[], [], []
for epoch in range(epochNum):
    model.train()
    random_selected_idxs = random.sample(range(0, tr_len), num_processing_steps)
    node_attr = train_x * (1 - train_indicates_mask)
    # print(node_attr.shape)
    # print(train_X_mask.shape)
    x_masks = [torch.as_tensor(train_X_mask[step_t, :, :], dtype=torch.float32, device=device).to(device) for step_t in
               random_selected_idxs]
    x_holdouts = [torch.as_tensor(ori_X[step_t, :, :], dtype=torch.float32, device=device).to(device) for step_t in
                  random_selected_idxs]
    indicates_masks = [
        torch.as_tensor(train_indicates_mask[step_t, :, :], dtype=torch.float32, device=device).to(device) for step_t in
        random_selected_idxs]
    input_graphs = [
        Data(x=torch.as_tensor(node_attr[step_t, :, :], dtype=torch.float32, device=device),
             edge_index=torch.as_tensor(edge_index, device=device).to(device),
             edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32, device=device).to(device),
             y=torch.as_tensor(float(step_t) / (tr_len + test_len), dtype=torch.float32, device=device).unsqueeze(0).to(
                 device)
             ) for step_t in random_selected_idxs]
    output_tensors, _,_,_ = model(input_graphs, num_processing_steps,x_masks,x_holdouts,indicates_masks,"train")
    ##prediction loss
    loss_seq = [torch.sum((torch.nan_to_num(output) - torch.nan_to_num(torch.as_tensor(train_y[random_selected_idxs[step_t], :, :], dtype=torch.float32, device=device))) ** 2)
                for step_t, output in enumerate(output_tensors)]

    loss = args.lossweight* sum(loss_seq) / len(loss_seq)
    optimizer.zero_grad()
    # print(imputeLoss.item())
    # print(reconsLoss.item())
    # print("check loss")
    print(loss.item())
    loss.backward()
    optimizer.step()

model.eval()
input_plot = []
impute_plot = []
true_impute_plot = []
predict_plot = []
true_predict_plot = []
with torch.no_grad():
    losses_te = []
    MAE, MAPE, RMSE = [], [], []
    imputedMAE, imputedMRE, imputedRMSE = [], [], []
    for tt in range(0, test_len - num_processing_steps):
        node_attr = test_x * (1 - test_indicates_mask)
        x_masks = [torch.as_tensor(test_X_mask[tt + step_t, :, :], dtype=torch.float32, device=device).to(device) for
                   step_t in range(num_processing_steps)]
        x_holdouts = [torch.as_tensor(ori_te_X[tt + step_t, :, :], dtype=torch.float32, device=device).to(device) for
                      step_t in range(num_processing_steps)]
        indicates_masks = [
            torch.as_tensor(test_indicates_mask[tt + step_t, :, :], dtype=torch.float32, device=device).to(
                device) for step_t in range(num_processing_steps)]

        input_graphs = [
            Data(x=torch.as_tensor(node_attr[tt + step_t, :, :], dtype=torch.float32, device=device),
                 edge_index=torch.as_tensor(edge_index, device=device).to(device),
                 edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32, device=device).to(device),
                 y=torch.as_tensor(float(tt + step_t) / (tr_len + test_len), dtype=torch.float32,
                                   device=device).unsqueeze(0).to(device))
            for step_t in range(num_processing_steps)]

        output_tensors, _, _, _= model(input_graphs, num_processing_steps,x_masks,x_holdouts,indicates_masks,"test")
        te_loss_seq = [torch.sum((output - torch.as_tensor(test_y[tt + step_t, :, :],dtype=torch.float32, device=device)) ** 2) for step_t, output in enumerate(output_tensors)]
        te_loss = sum(te_loss_seq) / len(te_loss_seq)
        # print(te_loss)
        mae, mape, rmse = Evaluation.total(scaler.inverse_transform(output_tensors[0].cpu().numpy().reshape(-1,1)),
                                           scaler.inverse_transform(test_y[tt].reshape(-1,1)))
        # # print(indicates_masks[0])
        # imputed_mae= masked_mae_cal(torch.as_tensor(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1))).reshape(node_attr.shape[1],-1),torch.as_tensor(scaler.inverse_transform(np.nan_to_num(node_attr[tt, :, :]).reshape(-1, 1))).reshape(node_attr.shape[1],-1),indicates_masks[0].cpu())
        # imputed_rmse = masked_rmse_cal(torch.as_tensor(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1))).reshape(node_attr.shape[1],-1),torch.as_tensor(scaler.inverse_transform(np.nan_to_num(node_attr[tt, :, :]).reshape(-1, 1))).reshape(node_attr.shape[1],-1),indicates_masks[0].cpu())
        # imputed_mre = masked_mre_cal(torch.as_tensor(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1))).reshape(node_attr.shape[1],-1),torch.as_tensor(scaler.inverse_transform(np.nan_to_num(node_attr[tt, :, :]).reshape(-1, 1))).reshape(node_attr.shape[1],-1), indicates_masks[0].cpu())
        #
        # temp_impute_plot = np.reshape(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1)),(node_attr.shape[1],-1))
        # impute_plot.append(temp_impute_plot)
        # temp_true_impute_plot =np.reshape(scaler.inverse_transform(np.nan_to_num(ori_te_X[tt, :, :]).reshape(-1, 1)),(ori_te_X.shape[1],-1))
        # input_plot.append(temp_true_impute_plot*(test_X_mask[tt,:,:]))
        # true_impute_plot.append(temp_true_impute_plot)
        # predict_plot.append(np.squeeze(scaler.inverse_transform(output_tensors[0].cpu().numpy().reshape(-1,1)), axis=1).reshape(-1,12))
        # true_predict_plot.append(np.squeeze(scaler.inverse_transform(test_y[tt].reshape(-1,1)),axis=1).reshape(-1,12))

        MAE.append(mae)
        MAPE.append(mape)
        MAPE.append(mape)
        RMSE.append(rmse)
        # imputedMAE.append(imputed_mae)
        # imputedMRE.append(imputed_mre)
        # imputedRMSE.append(imputed_rmse)


    wandb.log({'MAE': np.sum(MAE) / len(MAE)})
    wandb.log({'MAPE': np.sum(MAPE) / len(MAPE)})
    wandb.log({'RMSE': np.sum(RMSE) / len(RMSE)})
    # wandb.log({'imputedMAE': np.sum(imputedMAE) / len(imputedMAE)})
    # wandb.log({'imputedMRE': np.sum(imputedMRE) / len(imputedMRE)})
    # wandb.log({'imputedRMSE': np.sum(imputedRMSE) / len(imputedRMSE)})
    # for i in range(15):
    #     df_input_plot = pd.DataFrame(np.array(input_plot[i]))
    #     df_input_plot.to_csv('plotData_/input/input_plot' + str(i) + '.csv', index=False)
    #     df_impute_plot = pd.DataFrame(np.array(impute_plot[i]))
    #     df_impute_plot.to_csv('plotData_/impute/impute_plot'+ str(i)+'.csv', index=False)
    #     df_true_impute_plot = pd.DataFrame(np.array(true_impute_plot[i]))
    #     df_true_impute_plot.to_csv('plotData_/true_impute/true_impute_plot'+ str(i)+'.csv', index=False)
    # for j in range(15):
    #     df_predict_plot = pd.DataFrame(predict_plot[j])
    #     df_predict_plot.to_csv('plotData_/predict/predict'+ str(j)+'.csv', index=False)
    #     df_true_predict_plot = pd.DataFrame(true_predict_plot[j])
    #     df_true_predict_plot.to_csv('plotData_/true_pre/true_predict_plot'+ str(j)+'.csv', index=False)
    # df_edge_index= pd.DataFrame(edge_index.cpu().numpy())
    # df_edge_index.to_csv('plotData/edge_index.csv', index=False)
