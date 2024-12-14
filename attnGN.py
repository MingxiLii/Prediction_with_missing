import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from utils import generate_edge_index_, generate_edge_index_pems
from selfAttentionImpute.saits import SAITS
from GN.GN_module import P_GN
from GN.utils_tool import decompose_graph,generate_flow_dataset_adj, get_adjacency_matrix
from GN.blocks import EdgeBlock, NodeBlock, GlobalBlock
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data
from gcimpute.gaussian_copula import GaussianCopula
from sklearn.impute import SimpleImputer
class selfAttenGN(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_channels_2,
                 node_attr_size, out_size,  device, edge_hidden_size , node_hidden_size ,
                 global_hidden_size):
        super(selfAttenGN, self).__init__()
        self.in_channel = input_channels
        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim) / 2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.device = device
        ##n_groups, n_group_inner_layers,d_time,d_feature, d_model, d_inner,n_head,d_k,d_v.dropout,
        self.saits = SAITS( n_groups=1,n_group_inner_layers=1, d_time= node_attr_size,d_feature=1, d_model=hidden_channels, d_inner=hidden_channels, n_head=4, d_k=hidden_channels_2, d_v=hidden_channels_2, dropout=0.0, device= device)
        # Encoder
        self.edge_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())
        self.global_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())

        self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                            )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                            )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                            )
        self.eb_module = EdgeBlock((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True,
                                   use_sender_nodes=True,
                                   use_receiver_nodes=True,
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=True,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add,
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        self.gn = P_GN(self.eb_module,
                       self.nb_module,
                       self.gb_module,
                       use_edge_block=True,
                       use_node_block=True,
                       use_global_block=True)
        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, out_size)
                                      )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size))

    def forward(self,data,num_processing_steps,x_masks, x_holdouts,indicates,stage):
        input_graphs = []
        imputed= []
        recons_loss = 0
        imputed_loss = 0
        for step_t in range(num_processing_steps):
            node_attr, edge_index, edge_attr,global_attr = decompose_graph(data[step_t])
            x_holdout = x_holdouts[step_t]
            x_mask =x_masks[step_t]
            indicating_mask = indicates[step_t]
            X, masks = node_attr.unsqueeze(2), x_mask.unsqueeze(2)
            x_holdout, indicating_mask = x_holdout.unsqueeze(2),  indicating_mask.unsqueeze(2)
            inputs= {'X':X, "missing_mask":masks, 'X_holdouts':x_holdout, 'indicating_mask':indicating_mask}

            imputed_out = self.saits(inputs,stage)
            imputed_data = imputed_out['imputed_data']
            imputed.append(imputed_data)
            recons_loss += imputed_out['reconstruction_loss']
            imputed_loss += imputed_out['imputation_loss']
            #### Input for GN
            encoded_node = self.node_enc(imputed_data.squeeze(2))
            encoded_edge = self.edge_enc(edge_attr)
            encoded_global = self.global_enc(global_attr.unsqueeze(1))

            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            input_graph.y = encoded_global
            input_graphs.append(input_graph)
        init_graph = input_graphs[0]
        # h_init is zero tensor
        h_init = Data(x=init_graph.x,
                      edge_index=init_graph.edge_index,
                      edge_attr=init_graph.edge_attr)
        h_init.y = init_graph.y

        ### GN
        output_graphs = self.gn(input_graphs, h_init)

        output_nodes, pred_inputs = [], []
        for output_graph in output_graphs:
            output_nodes.append(self.node_dec(output_graph.x))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))
        return output_nodes, recons_loss/num_processing_steps, imputed_loss/num_processing_steps,imputed



class ablation_1(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_channels_2,
                 node_attr_size, out_size,  device, edge_hidden_size , node_hidden_size ,
                 global_hidden_size):
        super(ablation_1, self).__init__()
        self.in_channel = input_channels
        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim) / 2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.device = device
        ##n_groups, n_group_inner_layers,d_time,d_feature, d_model, d_inner,n_head,d_k,d_v.dropout,
        self.saits = SAITS( n_groups=1,n_group_inner_layers=1, d_time= node_attr_size,d_feature=1, d_model=hidden_channels, d_inner=hidden_channels, n_head=4, d_k=hidden_channels_2, d_v=hidden_channels_2, dropout=0.0, device= device)
        # Encoder
        self.edge_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())
        self.global_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())

        self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                            )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                            )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                            )
        self.eb_module = EdgeBlock((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True,
                                   use_sender_nodes=True,
                                   use_receiver_nodes=True,
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=True,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add,
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        self.gn = P_GN(self.eb_module,
                       self.nb_module,
                       self.gb_module,
                       use_edge_block=True,
                       use_node_block=True,
                       use_global_block=True)
        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, out_size)
                                      )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size))

    def forward(self,data,num_processing_steps,x_masks, x_holdouts,indicates,stage):
        input_graphs = []
        imputed= []
        recons_loss = 0
        imputed_loss = 0
        for step_t in range(num_processing_steps):
            node_attr, edge_index, edge_attr,global_attr = decompose_graph(data[step_t])
            # x_holdout = x_holdouts[step_t]
            # x_mask =x_masks[step_t]
            # indicating_mask = indicates[step_t]
            # X, masks = node_attr.unsqueeze(2), x_mask.unsqueeze(2)
            # x_holdout, indicating_mask = x_holdout.unsqueeze(2),  indicating_mask.unsqueeze(2)
            # inputs= {'X':X, "missing_mask":masks, 'X_holdouts':X, 'indicating_mask':indicating_mask}
            #
            # imputed_out = self.saits(inputs,stage)
            # imputed_data = imputed_out['imputed_data']
            # imputed.append(imputed_data)
            # recons_loss += imputed_out['reconstruction_loss']
            # imputed_loss += imputed_out['imputation_loss']
            #### Input for GN
            # encoded_node = self.node_enc(imputed_data.squeeze(2))
            encoded_node = self.node_enc(node_attr)
            encoded_edge = self.edge_enc(edge_attr)
            encoded_global = self.global_enc(global_attr.unsqueeze(1))

            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            input_graph.y = encoded_global
            input_graphs.append(input_graph)
        init_graph = input_graphs[0]
        # h_init is zero tensor
        h_init = Data(x=init_graph.x,
                      edge_index=init_graph.edge_index,
                      edge_attr=init_graph.edge_attr)
        h_init.y = init_graph.y

        ### GN
        output_graphs = self.gn(input_graphs, h_init)

        output_nodes, pred_inputs = [], []
        for output_graph in output_graphs:
            output_nodes.append(self.node_dec(output_graph.x))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))
        return output_nodes, 0.0, 0.0, 0.0


class ablation_2(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_channels_2,
                 node_attr_size, out_size,  device, edge_hidden_size , node_hidden_size ,
                 global_hidden_size):
        super(ablation_2, self).__init__()
        self.in_channel = input_channels
        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim) / 2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.device = device
        ##n_groups, n_group_inner_layers,d_time,d_feature, d_model, d_inner,n_head,d_k,d_v.dropout,
        self.saits = SAITS( n_groups=1,n_group_inner_layers=1, d_time= node_attr_size,d_feature=1, d_model=hidden_channels, d_inner=hidden_channels, n_head=4, d_k=hidden_channels_2, d_v=hidden_channels_2, dropout=0.0, device= device)
        # Encoder
        self.edge_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())
        self.global_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())

        self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                            )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                            )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                            )
        self.eb_module = EdgeBlock((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True,
                                   use_sender_nodes=True,
                                   use_receiver_nodes=True,
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=True,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add,
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        self.gn = P_GN(self.eb_module,
                       self.nb_module,
                       self.gb_module,
                       use_edge_block=True,
                       use_node_block=True,
                       use_global_block=True)
        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, out_size)
                                      )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size))

    def forward(self,data,num_processing_steps,x_masks, x_holdouts,indicates,stage):
        input_graphs = []
        imputed= []
        recons_loss = 0
        imputed_loss = 0
        for step_t in range(num_processing_steps):
            node_attr, edge_index, edge_attr,global_attr = decompose_graph(data[step_t])
            encoded_node = self.node_enc(node_attr)
            encoded_edge = self.edge_enc(edge_attr)
            encoded_global = self.global_enc(global_attr.unsqueeze(1))

            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            input_graph.y = encoded_global
            input_graphs.append(input_graph)
        init_graph = input_graphs[0]
        # h_init is zero tensor
        h_init = Data(x=init_graph.x,
                      edge_index=init_graph.edge_index,
                      edge_attr=init_graph.edge_attr)
        h_init.y = init_graph.y

        ### GN
        output_graphs = self.gn(input_graphs, h_init)

        output_nodes, pred_inputs = [], []
        for output_graph in output_graphs:
            output_nodes.append(self.node_dec(output_graph.x))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))
        return output_nodes, 0.0, 0.0, 0.0



class ablation_3(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_channels_2,
                 node_attr_size, out_size,  device, edge_hidden_size , node_hidden_size ,
                 global_hidden_size):
        super(ablation_3, self).__init__()
        self.in_channel = input_channels
        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim) / 2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.device = device
        ##n_groups, n_group_inner_layers,d_time,d_feature, d_model, d_inner,n_head,d_k,d_v.dropout,
        self.saits = SAITS( n_groups=1,n_group_inner_layers=1, d_time= node_attr_size,d_feature=1, d_model=hidden_channels, d_inner=hidden_channels, n_head=4, d_k=hidden_channels_2, d_v=hidden_channels_2, dropout=0.0, device= device)
        # Encoder
        self.edge_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())
        self.global_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())

        self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                            )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                            )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                            )
        self.eb_module = EdgeBlock((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True,
                                   use_sender_nodes=True,
                                   use_receiver_nodes=True,
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=True,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add,
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        self.gn = P_GN(self.eb_module,
                       self.nb_module,
                       self.gb_module,
                       use_edge_block=True,
                       use_node_block=True,
                       use_global_block=True)
        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, out_size)
                                      )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size))

    def forward(self,data,num_processing_steps,x_masks, x_holdouts,indicates,stage):
        input_graphs = []
        imputed= []
        recons_loss = 0
        imputed_loss = 0
        for step_t in range(num_processing_steps):
            node_attr, edge_index, edge_attr,global_attr = decompose_graph(data[step_t])
            encoded_node = self.node_enc(node_attr)
            encoded_edge = self.edge_enc(edge_attr)
            encoded_global = self.global_enc(global_attr.unsqueeze(1))

            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            input_graph.y = encoded_global
            input_graphs.append(input_graph)
        # init_graph = input_graphs[0]
        # # h_init is zero tensor
        # h_init = Data(x=init_graph.x,
        #               edge_index=init_graph.edge_index,
        #               edge_attr=init_graph.edge_attr)
        # h_init.y = init_graph.y
        #
        # ### GN
        # output_graphs = self.gn(input_graphs, h_init)

        output_nodes, pred_inputs = [], []
        for output_graph in input_graphs:
            output_nodes.append(self.node_dec(output_graph.x))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))
        return output_nodes, 0.0, 0.0, 0.0

def divide_mask_randomly(mask, ratio):
    """
    Divides a binary mask into two parts randomly, based on a specified ratio.

    Parameters:
    - mask: A numpy array representing the binary mask to be divided.
    - ratio: A float indicating the proportion of object pixels to go into the first mask.

    Returns:
    - mask1, mask2: Two numpy arrays, where mask1 contains a portion of the object pixels based on the specified ratio,
                    and mask2 contains the remaining pixels.
    """
    # Ensure the input is a binary mask and ratio is within valid range
    mask = (mask > 0).astype(np.uint8)
    ratio = max(0, min(ratio, 1))  # Clamping ratio to [0, 1]

    # Find indices of all object pixels
    object_indices = np.where(mask == 1)

    # Pair up the indices and shuffle them
    paired_indices = list(zip(object_indices[0], object_indices[1]))
    np.random.shuffle(paired_indices)

    # Calculate the split point based on the specified ratio
    split_point = int(len(paired_indices) * ratio)
    group1_indices, group2_indices = paired_indices[:split_point], paired_indices[split_point:]

    # Create two new masks and assign the split object pixels
    mask1 = np.zeros_like(mask)
    mask2 = np.zeros_like(mask)

    for i, j in group1_indices:
        mask1[i, j] = 1

    for i, j in group2_indices:
        mask2[i, j] = 1

    return mask1, mask2

def get_chunk_IO(array3d_true,array3d_missing, binary_mask, look_back_len,pre_len,ratio):
    _, indicating_mask = divide_mask_randomly(binary_mask, ratio)
    missing_masks = binary_mask-indicating_mask
    missing_rate = float(np.sum(binary_mask) / np.sum(np.ones_like(binary_mask)))
    print(missing_rate)
    print(binary_mask.shape)
    t_in = look_back_len
    t_out = pre_len
    dT = t_in + t_out
    T = array3d_true.shape[1]
    M = T - dT + 1
    chunk = []
    chunk_truth = []
    indicates = []
    mask = []
    for mind in range(M):
        chunk.append(array3d_missing[:, range(mind, mind + dT), :])
        chunk_truth.append(array3d_true[:,  range(mind, mind + dT),:])
        indicates.append(indicating_mask[:, range(mind, mind + dT), :])
        mask.append(missing_masks[:, range(mind, mind + dT), :])
    mask = np.array(mask)
    indicates = np.array(indicates)
    chunk = np.array(chunk)
    chunk_truth = np.array(chunk_truth)
    print("chunk is created. shape (N,M,D,T_IN+T_OUT)={}".format(chunk.shape))

    full_x = chunk_truth[:, :, :t_in, :]
    miss_x = chunk[:, :, :t_in, :]
    indicates = indicates[:, :, :t_in, :]
    mask = mask[:, :, :t_in, :]
    print(array3d_missing.shape)

    y = chunk_truth[:, :, t_in:, :]
    print(y.max())

    return full_x.squeeze(3),miss_x.squeeze(3), y.squeeze(3), mask.squeeze(3), indicates.squeeze(3)

def load_metr_LA(directory,missing_rate, mode, impute,pre_len):
    ### load data
    # directory = '/home/mingxi/deep_learning_implementation/CDE/datasets/Metr-LA-data-set/'
    #
    # missing_rate = 0.4
    # mode = 'PM'

    ADJ = np.load(directory + 'Metr_ADJ.npy')
    dense_mat = np.load( directory + 'Metr-LA.npy')
    scaler = MinMaxScaler()
    scaler.fit(dense_mat.reshape(-1,1))
    dense_mat = scaler.transform(dense_mat.reshape(-1,1)).reshape(dense_mat.shape)
    print('Dataset shape:')
    print(dense_mat.shape)

    # =============================================================================
    ### Random missing (RM) scenario
    ### Set the RM scenario by:

    rm_random_mat = np.load(directory + 'rm_random_mat.npy')
    binary_mat = np.round(rm_random_mat + 0.5 - missing_rate)
    sparse_mat = np.multiply(dense_mat, binary_mat)
    print(sparse_mat.shape)

    # =============================================================================
    ### Random missing (RM) scenario
    ### Set the RM scenario by:
    if mode == 'PM':
        rm_random_mat = np.load(directory + 'rm_random_mat.npy')
        binary_mat = np.round(rm_random_mat + 0.5 - missing_rate)
    # =============================================================================
    # =============================================================================
    ### Non-random missing (NM) scenario
    ### Set the NM scenario by:
    if mode == 'CM':
        nm_random_mat = np.load(directory + 'nm_random_mat.npy')
        binary_tensor = np.zeros((dense_mat.shape[0], 61, 288))
        for i1 in range(binary_tensor.shape[0]):
            for i2 in range(binary_tensor.shape[1]):
                binary_tensor[i1, i2, :] = np.round(nm_random_mat[i1, i2] + 0.5 - missing_rate)
        binary_mat = binary_tensor.reshape([binary_tensor.shape[0], binary_tensor.shape[1] * binary_tensor.shape[2]])
    # =============================================================================

    sparse_mat = np.multiply(dense_mat, binary_mat)
    print(sparse_mat.shape)
    if impute==True:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mask = (binary_mat == 0)
        sparse_mat[imputer_mask] = np.nan
        imputer.fit(np.transpose(sparse_mat,(1,0)))
        imputed_mat = imputer.transform(np.transpose(sparse_mat,(1,0)))
        sparse_mat= np.transpose(imputed_mat,(1,0))
    test_len = 6048 #sparse_mat.shape[1] - train_len
    train_len = sparse_mat.shape[1] - test_len
    training_set = sparse_mat[:, :train_len]
    train_mask =  binary_mat[:,:train_len]
    test_set = sparse_mat[:, train_len:]
    test_mask =  binary_mat[:,train_len:]
    print('The size of training set is:')
    print(training_set.shape)
    print()
    print('The size of test set is:')
    print(test_set.shape)

    training_ground_truth = dense_mat[:, :train_len]
    test_ground_truth = dense_mat[:, train_len:]
    print('The size of training set ground truth is:')
    print(training_ground_truth.shape)
    print()
    print('The size of test set ground truth is:')
    print(test_ground_truth.shape)

    ###
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask = get_chunk_IO(training_ground_truth[:,:,np.newaxis],training_set[:,:,np.newaxis], train_mask[:,:,np.newaxis], 12,pre_len, 0.90)
    ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask = get_chunk_IO(test_ground_truth[:,:,np.newaxis],test_set[:,:,np.newaxis], test_mask[:,:,np.newaxis], 12,pre_len, 0.90)
    edge_index,edge_attr = generate_edge_index_(directory + 'Metr_ADJ.npy')
    print(train_y.shape)
    return ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask,edge_index,edge_attr,scaler

def graph_weight_cal(dist, epsilon = 0.5):
    dim = dist.shape[0]
    distances = dist[np.nonzero(dist)].flatten()
    std = distances.std()
    print(std)
    std_square = 5 * std ** 2
    A = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if dist[i][j] > 0:
                weight = np.exp(- dist[i][j] ** 2 / std_square)
                if i != j and weight >= epsilon:
                    A[i][j] = weight
    return A
def load_PEMSD8(directory,missing_rate, mode, impute, datatype,pre_len):
    Dist = np.load(directory + 'Dist.npy')
    A = np.load(directory + 'Adj.npy')
    dense_mat = np.load(directory + 'PeMS08.npy')
    print('check_dense_mat')
    print(dense_mat.shape)
    if datatype == 'speed':
        dense_mat = dense_mat[:,2,:]
    else:
        dense_mat = dense_mat[:, 0, :]
    scaler = MinMaxScaler()
    scaler.fit(dense_mat.reshape(-1, 1))
    dense_mat = scaler.transform(dense_mat.reshape(-1, 1)).reshape(dense_mat.shape)
    # dense_mat = dense_tensor.reshape(dim1 * dim2, dim3)
    # =============================================================================
    ### Point-wise missing (PM) scenario
    ### Set the PM scenario by:
    if mode == 'PM':
        pm_random_mat = np.load(directory + 'pm_random_mat.npy')
        binary_mat = np.round(pm_random_mat + 0.5 - missing_rate)
        sparse_mat= dense_mat* binary_mat
    # =============================================================================
    # =============================================================================
    ### Continuous-random missing (CM) scenario
    ### Set the CM scenario by:
    if mode == 'CM':
        nm_random_mat = np.load(directory + 'nm_random_mat.npy')
        binary_tensor = np.zeros((dense_mat.shape[0], 61, 288))
        for i1 in range(binary_tensor.shape[0]):
            for i2 in range(binary_tensor.shape[1]):
                binary_tensor[i1, i2, :] = np.round(nm_random_mat[i1, i2] + 0.5 - missing_rate)
        binary_mat = binary_tensor.reshape([binary_tensor.shape[0], binary_tensor.shape[1] * binary_tensor.shape[2]])
    # =============================================================================
    print('Missing rate = %s %.1f' % (mode, missing_rate))
    A = graph_weight_cal(Dist, epsilon=0.5)

    if impute==True:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mask = (binary_mat == 0)
        sparse_mat[imputer_mask] = np.nan
        imputer.fit(np.transpose(sparse_mat,(1,0)))
        imputed_mat = imputer.transform(np.transpose(sparse_mat,(1,0)))
        sparse_mat= np.transpose(imputed_mat,(1,0))
    sparse_mat = sparse_mat[:,  8928:]
    dense_tensor = dense_mat[:,  8928:]
    test_len = 2880
    train_len = sparse_mat.shape[1] - test_len
    training_set = sparse_mat[:, :train_len]
    test_set = sparse_mat[:, train_len:]
    print('The size of training set is:')
    print(training_set.shape)
    print()
    print('The size of test set is:')
    print(test_set.shape)
    training_ground_truth = dense_tensor[:,  :train_len]
    test_ground_truth = dense_tensor[:,  train_len:]
    print('The size of training set ground truth is:')
    print(training_ground_truth.shape)
    print()
    print('The size of test set ground truth is:')
    print(test_ground_truth.shape)
    training_set = sparse_mat[:, :train_len]
    train_mask = binary_mat[:, :train_len]
    test_set = sparse_mat[:, train_len:]
    test_mask =binary_mat[:, train_len:]
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask = get_chunk_IO(training_ground_truth[:, :, np.newaxis],
                                                                               training_set[:, :, np.newaxis],
                                                                               train_mask[:, :, np.newaxis], 12, pre_len,
                                                                               0.75)
    ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask = get_chunk_IO(test_ground_truth[:, :, np.newaxis],
                                                                              test_set[:, :, np.newaxis],
                                                                              test_mask[:, :, np.newaxis], 12, pre_len, 0.75)

    edge_index, edge_attr = generate_edge_index_(directory + 'Adj.npy')
    print(train_y.shape)
    return ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask,edge_index,edge_attr,scaler
def load_seattle(directory,missing_rate, mode, impute,pre_len):
    ADJ = np.load(directory + 'Loop_Seattle_2015_A.npy')
    dense_mat = np.load(directory + 'dense_mat.npy')
    scaler = MinMaxScaler()
    scaler.fit(dense_mat.reshape(-1, 1))
    dense_mat = scaler.transform(dense_mat.reshape(-1, 1)).reshape(dense_mat.shape)
    print('Dataset shape:')
    print(dense_mat.shape)
    print('Adjacency matrix shape:')
    print(ADJ.shape)
    # =============================================================================
    ### Random missing (PM) scenario
    ### Set the PM scenario by:
    if mode =='PM':
        rm_random_mat = np.load(directory + 'rm_random_mat.npy')
        binary_mat = np.round(rm_random_mat + 0.5 - missing_rate)
    else:
        ### Set the CM scenario by:
        nm_random_mat = np.load(directory + 'nm_random_mat.npy')
        binary_tensor = np.zeros((dense_mat.shape[0], 61, 288))
        for i1 in range(binary_tensor.shape[0]):
            for i2 in range(binary_tensor.shape[1]):
                binary_tensor[i1, i2, :] = np.round(nm_random_mat[i1, i2] + 0.5 - missing_rate)
        binary_mat = binary_tensor.reshape([binary_tensor.shape[0], binary_tensor.shape[1] * binary_tensor.shape[2]])
    # =============================================================================

    sparse_mat = np.multiply(dense_mat, binary_mat)
    test_rate = 0.15
    if impute==True:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mask = (binary_mat == 0)
        sparse_mat[imputer_mask] = np.nan
        imputer.fit(np.transpose(sparse_mat,(1,0)))
        imputed_mat = imputer.transform(np.transpose(sparse_mat,(1,0)))
        sparse_mat= np.transpose(imputed_mat,(1,0))

    train_len = int((1 - test_rate) * sparse_mat.shape[1])
    test_len = sparse_mat.shape[1] - train_len
    training_set = sparse_mat[:, :train_len]
    train_mask = binary_mat[:, :train_len]
    test_set = sparse_mat[:, train_len:]
    test_mask = binary_mat[:, train_len:]
    print('The size of training set is:')
    print(training_set.shape)
    print()
    print('The size of test set is:')
    print(test_set.shape)

    training_ground_truth = dense_mat[:, :train_len]
    test_ground_truth = dense_mat[:, train_len:]
    print('The size of training set ground truth is:')
    print(training_ground_truth.shape)
    print()
    print('The size of test set ground truth is:')
    print(test_ground_truth.shape)
    ###
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask = get_chunk_IO(training_ground_truth[:, :, np.newaxis],
                                                                               training_set[:, :, np.newaxis],
                                                                               train_mask[:, :, np.newaxis], 12, pre_len,
                                                                               0.75)
    ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask = get_chunk_IO(test_ground_truth[:, :, np.newaxis],
                                                                              test_set[:, :, np.newaxis],
                                                                              test_mask[:, :, np.newaxis], 12, pre_len, 0.75)
    edge_index, edge_attr = generate_edge_index_(directory + 'Loop_Seattle_2015_A.npy')
    print(train_y.shape)
    return ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler

def load_PEMS(directory,missing_rate, mode, impute,pre_len):
    dense_mat = np.load(directory+'pems.npy')

    scaler = MinMaxScaler()
    scaler.fit(dense_mat.reshape(-1, 1))
    dense_mat = scaler.transform(dense_mat.reshape(-1, 1)).reshape(dense_mat.shape)
    print('Dataset shape:')
    print(dense_mat.shape)

    # =============================================================================
    ### Random missing (PM) scenario
    ### Set the PM scenario by:
    if mode == 'PM':
        rm_random_mat =  np.random.rand(dense_mat.shape[0], dense_mat.shape[1])
        binary_mat = np.round(rm_random_mat + 0.5 - missing_rate)
    else:
        ### Set the CM scenario by:
        nm_random_mat = np.load(directory + 'random_matrix.npy')
        binary_tensor = np.zeros((dense_mat.shape[0], 61, 288))
        for i1 in range(binary_tensor.shape[0]):
            for i2 in range(binary_tensor.shape[1]):
                binary_tensor[i1, i2, :] = np.round(nm_random_mat[i1, i2] + 0.5 - missing_rate)
        binary_mat = binary_tensor.reshape([binary_tensor.shape[0], binary_tensor.shape[1] * binary_tensor.shape[2]])
    # =============================================================================

    sparse_mat = np.multiply(dense_mat, binary_mat)
    test_rate = 0.15
    if impute==True:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mask = (binary_mat == 0)
        sparse_mat[imputer_mask] = np.nan
        imputer.fit(np.transpose(sparse_mat,(1,0)))
        imputed_mat = imputer.transform(np.transpose(sparse_mat,(1,0)))
        sparse_mat= np.transpose(imputed_mat,(1,0))

    train_len = int((1 - test_rate) * sparse_mat.shape[1])
    test_len = sparse_mat.shape[1] - train_len
    training_set = sparse_mat[:, :train_len]
    train_mask = binary_mat[:, :train_len]
    test_set = sparse_mat[:, train_len:]
    test_mask = binary_mat[:, train_len:]
    print('The size of training set is:')
    print(training_set.shape)
    print()
    print('The size of test set is:')
    print(test_set.shape)

    training_ground_truth = dense_mat[:, :train_len]
    test_ground_truth = dense_mat[:, train_len:]
    print('The size of training set ground truth is:')
    print(training_ground_truth.shape)
    print()
    print('The size of test set ground truth is:')
    print(test_ground_truth.shape)
    ###
    ori_X, train_x, train_y, train_X_mask, train_indicates_mask = get_chunk_IO(training_ground_truth[:, :, np.newaxis],
                                                                               training_set[:, :, np.newaxis],
                                                                               train_mask[:, :, np.newaxis], 12, pre_len,
                                                                               0.75)
    ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask = get_chunk_IO(test_ground_truth[:, :, np.newaxis],
                                                                              test_set[:, :, np.newaxis],
                                                                              test_mask[:, :, np.newaxis], 12, pre_len, 0.75)
    edge_index, edge_attr = generate_edge_index_pems(directory + 'weighted_adj.csv')
    print(edge_attr.shape)
    return ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler

# ori_X, train_x, train_y, train_X_mask, train_indicates_mask, ori_te_X, test_x, test_y, test_X_mask, test_indicates_mask, edge_index, edge_attr, scaler = load_PEMS('datasets/PeMS-data-set/',0.4, 'PM')