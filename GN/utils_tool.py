import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import pandas as pd
def calculate_nx_path(batch_edge_cost_data, batch_arrive_t,edge_tuple_list, batch_source_node, batch_target_node):
    # batch_edge_cost_array = batch_edge_cost_data.cpu().detach().numpy()
    # batch_arrive_t_array = batch_arrive_t.cpu().detach().numpy()
    # batch_source_node_array = batch_source_node.cpu().detach().numpy()
    # batch_target_node_array = batch_target_node.cpu().detach().numpy()

    edge_path_list = []
    sp_time_list = []
    for edge_cost_array, arrive_t_array, source_node_array, target_node_array in zip(batch_edge_cost_data,batch_arrive_t, batch_source_node, batch_target_node):
        edge_cost_list = np.squeeze(edge_cost_array).tolist()
        arrive_t = np.squeeze(arrive_t_array).tolist()
        source_node = np.squeeze(source_node_array).tolist()
        target_node = np.squeeze(target_node_array).tolist()

        edge_path, sp_time = calculate_sp_time(edge_cost_list,arrive_t,edge_tuple_list,int(source_node),int(target_node))
        edge_path_list.append(edge_path)
        sp_time_list.append(sp_time)
    return torch.tensor(edge_path_list).squeeze(1), torch.tensor(sp_time_list)

def calculate_sp_time(edge_cost_list,arrive_t,edge_tuple_list, source_node, target_node):
    G = nx.DiGraph()
    # G = nx.Graph()
    for (s_node,t_node), cost in zip(edge_tuple_list,edge_cost_list):
        G.add_edge(s_node,t_node,weight= cost)
    for u, v in G.edges():
        if u == v:
            G.remove_edge(u, v)
    # sp_time,path = nx.single_source_dijkstra(G,source=source_node,target=target_node,weight='weight')
    path_dict = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    time_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    sp_time = time_dict[source_node][target_node]
    path =path_dict[source_node][target_node]
    edge_path = np.zeros(len(edge_tuple_list))
    edge_index_list = []
    for i in range(len(path)-1):
        edge_index = edge_tuple_list.index((path[i],path[i+1]))
        edge_index_list.append(edge_index)
    edge_path[edge_index_list]=1.0
    # sp_time = sp_time + arrive_t[edge_tuple_list.index((path[len(path)-2],path[len(path)-1]))]
    return edge_path,sp_time
def batch_cvx_path_to_real_path(batch_path, edge_tuple_list, source_node, target_node):
    path_list = []
    count = 0
    for  single_path,s_n,t_n in zip(batch_path,source_node,target_node):

        single_path_weight = torch.abs(torch.log(torch.abs(single_path)))
        single_path_weight = torch.nan_to_num(single_path_weight,nan=1.0)
        # single_path_weight[single_path_weight == 0.0]=1.0
        single_path_weight[single_path_weight <0.0] = 1.0
        G = nx.DiGraph()

        for (s_node, t_node), cost in zip(edge_tuple_list, single_path_weight):
            # print(cost)
            G.add_edge(s_node, t_node, weight= abs(cost))
        for u, v in G.edges():
            if u == v:
                G.remove_edge(u, v)
        # print(nx.is_negatively_weighted(G, weight='weight'))
        # path_dict = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
        _, path = nx.single_source_dijkstra(G, source=int(s_n), target=int(t_n), weight='weight')
        # path = path_dict[int(s_n.item())][int(t_n.item())]
        edge_path = np.zeros(len(edge_tuple_list))
        edge_index_list = []
        for i in range(len(path) - 1):
            edge_index = edge_tuple_list.index((path[i], path[i + 1]))
            edge_index_list.append(edge_index)
        edge_path[edge_index_list] = 1.0

        path_list.append(edge_path)
        path_tensor = np.array(path_list)
        count = count + 1
        # print(count)

    return torch.Tensor(path_tensor)
def path_to_time(start_times, single_paths, time_matrix):

    time_horizon = time_matrix.shape[0]
    time_list= []
    for start_time, single_path in zip(start_times,single_paths):
        cur_ts = start_time

        for idx,edge_mask in enumerate(single_path):
            if edge_mask.item()>0.5:
                if cur_ts < time_horizon:
                    cur_ts = round(cur_ts)
                    edge_cost = time_matrix[cur_ts][idx]
                    cur_ts = cur_ts + edge_cost
                else:
                    cur_ts = time_horizon - 1
            else:
                continue

        totoal_time = cur_ts - start_time
        time_list.append(totoal_time)

    return time_list


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs((target - output) / (target+0.00001)))*100

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)

        return mae, mape, rmse


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
def generate_edge_adj(node_to_link):
    edge_tuple_list = np.array(node_to_link)
    edge_adj = np.eye(len(edge_tuple_list))
    for index, link_lable in enumerate(edge_tuple_list):
        pre_index = index
        pre_link_lable = link_lable
        for cur_index, cur_link_lable in enumerate(edge_tuple_list):
            common_node_exsited = any(x in set(pre_link_lable) for x in set(cur_link_lable))
            if common_node_exsited:
                edge_adj[pre_index, cur_index] = 1.0
    return edge_adj

def load_gn_data(X_path,Y_path,embedding_path):

    X_dic = torch.load(X_path)
    Edge_attrs = X_dic['X_time']
    X_source = X_dic['X_s_node']
    X_target = X_dic['X_t_node']
    Y_dic = torch.load(Y_path)
    Y_departure_t = Y_dic["departure_t"]
    Y_sp_tree = Y_dic["Y_sp_tree"]
    Y_label_time = Y_dic["Y_label_time"]
    Y_ins_time = Y_dic["Y_ins_time"]
    Y_path_ = Y_dic['Y_path']
    embedding_dic = torch.load(embedding_path)
    source_node_embedding = embedding_dic['s_embedding']
    target_node_embedding = embedding_dic['t_embedding']
    Y_ins_path = embedding_dic["Y_ins_path"]
    sample_num = Edge_attrs.shape[0]
    return Y_departure_t,Edge_attrs,X_source,X_target,Y_sp_tree,Y_path_, source_node_embedding,Y_ins_path,sample_num
def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        elif key=="y":
            global_attr = graph.y
        else:
            pass
    return (x,edge_index, edge_attr, global_attr)
def graph_concat(graph1, graph2,
                 node_cat=True, edge_cat=True, global_cat=False):
    """
    Args:
        graph1: torch_geometric.data.data.Data
        graph2: torch_geometric.data.data.Data
        node_cat: True if concat node_attr
        edge_cat: True if concat edge_attr
        global_cat: True if concat global_attr
    Return:
        new graph: concat(graph1, graph2)
    """
    # graph2 attr is used for attr that is not concated.
    _x = graph2.x
    _edge_attr = graph2.edge_attr
    _global_attr = graph2.y
    _edge_index = graph2.edge_index

    if node_cat:
        try:
            _x = torch.cat([graph1.x, graph2.x], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'x' key.")

    if edge_cat:
        try:
            _edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'edge_attr' key.")

    if global_cat:
        try:
            _global_attr = torch.cat([graph1.y, graph2.y], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'y' key.")

    ret = Data(x=_x, edge_attr=_edge_attr, edge_index=_edge_index)
    ret.y = _global_attr

    return ret


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)

    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.y = global_attr

    return ret

def create_nx_graph(edge_list,node_list):

    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    return G
def get_edge_index_from_nxG(G):
    """return edge_index for torch_geometric.data.data.Data
    G is networkx Graph.
    """
    A = nx.adj_matrix(G)  # A: sparse.csr_matrix
    r, c = A.nonzero()
    r = torch.tensor(r, dtype=torch.long)
    c = torch.tensor(c, dtype=torch.long)

    return torch.stack([r, c])

def get_adj(edge_index, weight=None):
    """return adjacency matrix"""
    if not weight:
        weight = torch.ones(edge_index.shape[1])

    row, col = edge_index
    return torch.sparse.FloatTensor(edge_index, weight)
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA
def generate_flow_dataset_adj(dataset_name,device, data_used_subset= False, subset_percent = 0.2):
    file_path = "/home/mingxi/flow_data/data/"+ dataset_name +"/" + dataset_name
    if dataset_name == 'PEMSD7':
        # PEMSD7: Time(5/1/2012 - 6/30/2012, weekdays), Nodes(228)
        speed_matrix = pd.read_csv("~/flow_data/PEMSD7/V_228.csv")
        speed_matrix = speed_matrix.values
        adj_matrix = pd.read_csv("~/flow_data/PEMSD7/weighted_adj.csv", index_col=None,header=None)
        adj_matrix = adj_matrix.values
    if dataset_name != 'PEMSD7':
        ##dataset_name
        # PEMS04: Time(1/1/2018 - 2/28/2018), Nodes(307)
        # PEMS07: Time(5/1/2017 - 8/31/2017), Nodes(883)
        # PEMS08: Time(7/1/2016 - 8/31/2016), Nodes(170)
        speed_matrix = np.load( file_path + ".npz")['data']
        speed_matrix = speed_matrix[:, :, 0]
        num_nodes = speed_matrix.shape[1]
        _ , adj_matrix = get_adjacency_matrix(file_path + ".csv", num_nodes)
    if data_used_subset == True:
        start_index = np.random.randint(0, (1 - subset_percent) * speed_matrix.shape[0])
        end_index = int(start_index + subset_percent * speed_matrix.shape[0])
        speed_matrix = speed_matrix.iloc[start_index, end_index]

    max = np.max(speed_matrix)
    speed_matrix = torch.tensor(speed_matrix)
    X = torch.unsqueeze(speed_matrix, 2)

    num_nodes = X.shape[1]

    adj = nx.convert_matrix.from_numpy_matrix(adj_matrix,parallel_edges=True,create_using=nx.MultiGraph)
    sp_L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(adj)
    rows, cols = sp_L .nonzero()
    data = sp_L[rows, cols]
    indicies = []
    indicies.append(rows)
    indicies.append(cols)
    indicies= torch.tensor(indicies)
    data = torch.tensor(data,  dtype=torch.float32).squeeze(0)
    sp_L = torch.sparse_coo_tensor(indicies, data).to(device)

    edgelist = [(u, v) for (u, v) in adj.edges()]
    edge_index = torch.tensor(edgelist)
    edge_index = edge_index.transpose(0,1)


    edge_attr = []
    for edge in edgelist:
        edge_attr.append(adj_matrix[edge[0]][edge[1]])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(1)

    return X, adj_matrix, edgelist, edge_index, edge_attr, num_nodes, max