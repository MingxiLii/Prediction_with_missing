

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
from attnGN import selfAttenGN, load_metr_LA,load_PEMSD8,load_seattle,load_PEMS
import wandb
import time
import argparse

class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)

        return Hidden_State, Cell_State

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        outputs = None
        for i in range(time_step):
            Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((Hidden_State.unsqueeze(1), outputs), 1)
        return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


class BiLSTM(nn.Module):

    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(BiLSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.il_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.fl_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.il_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl_b = nn.Linear(input_size + hidden_size, hidden_size)

    def step(self, input_f, input_b, Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b):
        batch_size = input_f.size(0)

        combined_f = torch.cat((input_f, Hidden_State_f), 1)

        f_f = F.sigmoid(self.fl_f(combined_f))
        i_f = F.sigmoid(self.il_f(combined_f))
        o_f = F.sigmoid(self.ol_f(combined_f))
        C_f = F.tanh(self.Cl_f(combined_f))
        Cell_State_f = f_f * Cell_State_f + i_f * C_f
        Hidden_State_f = o_f * F.tanh(Cell_State_f)

        combined_b = torch.cat((input_b, Hidden_State_b), 1)

        f_b = F.sigmoid(self.fl_b(combined_b))
        i_b = F.sigmoid(self.il_b(combined_b))
        o_b = F.sigmoid(self.ol_b(combined_b))
        C_b = F.tanh(self.Cl_b(combined_b))
        Cell_State_b = f_b * Cell_State_b + i_b * C_b
        Hidden_State_b = o_b * F.tanh(Cell_State_b)

        return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b

    def forward(self, inputs):
        outputs_f = None
        outputs_b = None

        batch_size = inputs.size(0)
        steps = inputs.size(1)

        Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b = self.initHidden(batch_size)

        for i in range(steps):
            Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b = \
                self.step(torch.squeeze(inputs[:, i:i + 1, :]), torch.squeeze(inputs[:, steps - i - 1:steps - i, :]) \
                          , Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b)

            if outputs_f is None:
                outputs_f = Hidden_State_f.unsqueeze(1)
            else:
                outputs_f = torch.cat((outputs_f, Hidden_State_f.unsqueeze(1)), 1)
            if outputs_b is None:
                outputs_b = Hidden_State_b.unsqueeze(1)
            else:
                outputs_b = torch.cat((Hidden_State_b.unsqueeze(1), outputs_b), 1)
        outputs = (outputs_f + outputs_b) / 2
        return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State_f = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State_f = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Hidden_State_b = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State_b = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
        else:
            Hidden_State_f = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State_f = Variable(torch.zeros(batch_size, self.hidden_size))
            Hidden_State_b = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State_b = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b


def Train_Multi_Bi_LSTM(train_dataloader, valid_dataloader, num_epochs=3):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    #     multiBiLSTM = Multi_Bi_LSTM(input_dim, hidden_dim, output_dim)

    multiBiLSTM = nn.Sequential(BiLSTM(input_dim, hidden_dim, output_dim), LSTM(input_dim, hidden_dim, output_dim))

    multiBiLSTM.cuda()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(multiBiLSTM.parameters(), lr=learning_rate)
    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            multiBiLSTM.zero_grad()

            outputs = multiBiLSTM(inputs)

            full_labels = torch.cat((inputs[:, 1:, :], labels), dim=1)

            loss_train = loss_MSE(outputs, full_labels)

            losses_train.append(loss_train.data)

            optimizer.zero_grad()

            loss_train.backward()

            optimizer.step()

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            multiBiLSTM.zero_grad()

            full_labels_val = torch.cat((inputs_val[:, 1:, :], labels_val), dim=1)

            outputs_val = multiBiLSTM(inputs_val)

            #             Hidden_State, Cell_State = bilstm.loop(inputs_val)

            loss_valid = loss_MSE(outputs_val, full_labels_val)
            #             loss_valid = loss_MSE(Hidden_State, labels_val)

            losses_valid.append(loss_valid.data)

            # output
            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0] / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0] / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format( \
                    trained_number * batch_size, \
                    loss_interval_train, \
                    loss_interval_valid, \
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

    return multiBiLSTM, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]


def PrepareDataset(speed_matrix, BATCH_SIZE=40, seq_len=10, pred_len=1, train_propotion=0.7, valid_propotion=0.2):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]

    speed_matrix = speed_matrix.clip(0, 100)

    max_speed = speed_matrix.max().max()
    speed_matrix = speed_matrix / max_speed

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LA')
    parser.add_argument('--pre_len', type=int, default=12)
    parser.add_argument('--exp_id', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.98)
    parser.add_argument('--use_model', type=str, default="attenGN")
    parser.add_argument('--directory', type=str, default='datasets/Metr-LA-data-set/',
                        help="datasets/Seattle_loop-data-set/,datasets/PeMS-data-set/,datasets/PeMSD8/,datasets/Metr-LA-data-set/")
    parser.add_argument('--missRate', type=float, default=0.8)
    parser.add_argument('--missMode', type=str, default='PM', help="CM or PM")
    parser.add_argument('--datatype', type=str, default='flow', help="flow or speed")
    parser.add_argument('--lossweight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default="cuda:1")

    args = parser.parse_args()

    wandb.init(project="traffic_attention_GN", config=args)
    if args.dataset == "LA":
        ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler \
            = load_metr_LA("/home/mingxi/deep_learning_implementation/CDE/datasets/Metr-LA-data-set/", args.missRate,
                           args.missMode, False, args.pre_len)
        adj = np.load(args.directory + 'Metr_ADJ.npy')

    elif args.dataset == 'seattle':
        ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler \
            = load_seattle(args.directory, args.missRate, args.missMode, False, args.pre_len)
    elif args.dataset == "PEMS":
        ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler \
            = load_PEMS(args.directory, args.missRate, args.missMode, False, args.pre_len)
    else:
        ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler \
            = load_PEMSD8(args.directory, args.missRate, args.missMode, False, args.datatype, args.pre_len)
        print('check')
        print(ori_X.shape)
        print(train_y.shape)

    tr_len = ori_X.shape[0]
    test_len = ori_te_X.shape[0]

    device = args.device

    print(ori_X.shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features

    epochNum = 3500
    num_processing_steps = 10
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    model = nn.Sequential(BiLSTM(input_dim, hidden_dim, output_dim), LSTM(input_dim, hidden_dim, output_dim))
    # model = ablation_1(input_channels=1, hidden_channels=64, hidden_channels_2=32, node_attr_size=12, out_size=args.pre_len,  device=device, edge_hidden_size=64 , node_hidden_size=64 ,
    #                      global_hidden_size=64)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00000001)
    # optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.95)
    maes, mapes, rmses = [], [], []
    for epoch in range(epochNum):
        model.train()
        random_selected_idxs = random.sample(range(0, tr_len), num_processing_steps)
        node_attr = train_x * (1 - train_indicates_mask)
        # print(node_attr.shape)
        # print(train_X_mask.shape)
        x_masks = [torch.as_tensor(train_X_mask[step_t, :, :], dtype=torch.float32, device=device).to(device) for step_t
                   in random_selected_idxs]
        x_holdouts = [torch.as_tensor(ori_X[step_t, :, :], dtype=torch.float32, device=device).to(device) for step_t in
                      random_selected_idxs]
        indicates_masks = [
            torch.as_tensor(train_indicates_mask[step_t, :, :], dtype=torch.float32, device=device).to(device) for
            step_t in random_selected_idxs]
        input_graphs = [
            Data(x=torch.as_tensor(node_attr[step_t, :, :], dtype=torch.float32, device=device),
                 edge_index=torch.as_tensor(edge_index, device=device).to(device),
                 edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32, device=device).to(device),
                 y=torch.as_tensor(float(step_t) / (tr_len + test_len), dtype=torch.float32, device=device).unsqueeze(
                     0).to(device)
                 ) for step_t in random_selected_idxs]
        output_tensors, reconsLoss, imputeLoss, _ = model(input_graphs, num_processing_steps, x_masks, x_holdouts,
                                                          indicates_masks, "train")
        ##prediction loss
        loss_seq = [torch.sum((torch.nan_to_num(output) - torch.nan_to_num(
            torch.as_tensor(train_y[random_selected_idxs[step_t], :, :], dtype=torch.float32, device=device))) ** 2)
                    for step_t, output in enumerate(output_tensors)]

        loss = args.lossweight * sum(loss_seq) / len(loss_seq) + imputeLoss + reconsLoss
        optimizer.zero_grad()
        print(imputeLoss.item())
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
            x_masks = [torch.as_tensor(test_X_mask[tt + step_t, :, :], dtype=torch.float32, device=device).to(device)
                       for step_t in range(num_processing_steps)]
            x_holdouts = [torch.as_tensor(ori_te_X[tt + step_t, :, :], dtype=torch.float32, device=device).to(device)
                          for step_t in range(num_processing_steps)]
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

            output_tensors, _, _, imputed = model(input_graphs, num_processing_steps, x_masks, x_holdouts,
                                                  indicates_masks, "test")
            te_loss_seq = [torch.sum(
                (output - torch.as_tensor(test_y[tt + step_t, :, :], dtype=torch.float32, device=device)) ** 2) for
                           step_t, output in enumerate(output_tensors)]
            te_loss = sum(te_loss_seq) / len(te_loss_seq)
            # print(te_loss)
            mae, mape, rmse = Evaluation.total(scaler.inverse_transform(output_tensors[0].cpu().numpy().reshape(-1, 1)),
                                               scaler.inverse_transform(test_y[tt].reshape(-1, 1)))
            # print(indicates_masks[0])
            imputed_mae = masked_mae_cal(
                torch.as_tensor(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1))).reshape(
                    ori_te_X.shape[1], -1),
                torch.as_tensor(scaler.inverse_transform(np.nan_to_num(ori_te_X[tt, :, :]).reshape(-1, 1))).reshape(
                    ori_te_X.shape[1], -1), indicates_masks[0].cpu())
            imputed_rmse = masked_rmse_cal(
                torch.as_tensor(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1))).reshape(
                    ori_te_X.shape[1], -1),
                torch.as_tensor(scaler.inverse_transform(np.nan_to_num(ori_te_X[tt, :, :]).reshape(-1, 1))).reshape(
                    ori_te_X.shape[1], -1), indicates_masks[0].cpu())
            imputed_mre = masked_mre_cal(
                torch.as_tensor(scaler.inverse_transform(imputed[0].squeeze(2).cpu().numpy().reshape(-1, 1))).reshape(
                    ori_te_X.shape[1], -1),
                torch.as_tensor(scaler.inverse_transform(np.nan_to_num(ori_te_X[tt, :, :]).reshape(-1, 1))).reshape(
                    ori_te_X.shape[1], -1), indicates_masks[0].cpu())

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
            imputedMAE.append(imputed_mae)
            imputedMRE.append(imputed_mre)
            imputedRMSE.append(imputed_rmse)

        wandb.log({'MAE': np.sum(MAE) / len(MAE)})
        wandb.log({'MAPE': np.sum(MAPE) / len(MAPE)})
        wandb.log({'RMSE': np.sum(RMSE) / len(RMSE)})
        wandb.log({'imputedMAE': np.sum(imputedMAE) / len(imputedMAE)})
        wandb.log({'imputedMRE': np.sum(imputedMRE) / len(imputedMRE)})
        wandb.log({'imputedRMSE': np.sum(imputedRMSE) / len(imputedRMSE)})
