#!/usr/bin/env python
# coding:utf-8
import json
import math
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from addict import Dict
from sklearn import datasets, manifold
from visdom import Visdom


def get_config(config_id=None):
    '''
    Method to read the config file and return as a dict
    '''
    path =  os.path.dirname(os.path.realpath(__file__)).split('helper')[0]

    if (config_id):
        config_file = "{}.json".format(config_id)
    with open(os.path.join(path, 'config', config_file), 'r') as fin:
        config = json.load(fin)
        
    return config

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, model_file):
    """
    :param state: Dict, e.g. {'state_dict': state,
                              'optimizer': optimizer,
                              'best_performance': [Float, Float],
                              'epoch': int}
    :param model_file: Str, file path
    :return:
    """
    torch.save(state, model_file)

def load_checkpoint(model_file, model, config, optimizer=None):
    """
    Load models
    :param model_file: Str, file_path
    :param model: Computational Graph
    :param config: Dict, config information
    :param optimizer: optimizer, torch.xxx
    :return: best_performance and config
    """
    checkpoint_model = torch.load(model_file)
    # re_define the start epoch value
    config.train.start_epoch = checkpoint_model['epoch'] + 1
    best_performance = checkpoint_model['best_performance']
    model.load_state_dict(checkpoint_model['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_model['optimizer'])

    return best_performance, config

def gen_A(node_num, taxonomy):
    """
    Params-->node_num: the number of nodes, i.e. n_classes, int
    Params-->taxonomy: the node relation, Dict
    """
    edges = []
    for key, values in taxonomy.items():
        for value in values:
            edges.append([key, value])
    edges = np.asarray(edges, dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_num, node_num),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    adj = normalize(adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def normalize(adj):
    """
    Params-->adj: adjacent matrix with self-connection
    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def t_SNE(output, dimension):
    """
    output: 待降维的数据
    dimension: 降维到的维度
    """
    tsne = manifold.TSNE(n_components=dimension, init='pca', random_state=0)
    result = tsne.fit_transform(output)

    return result

def visualization(result, labels):

    vis = Visdom()
    vis.scatter(
        X=result,
        Y=labels+1, # 将label的最小值从0变1，显示时label不可为0
        opts=dict(markersize=8, 
                  legend=['0','1','2','3','4','5','6','7','8','9','10','11'], 
                  title='Dimension reduction to %dD' %(result.shape[1]))
    )

def get_parent_node_number(label_tree):
    """
    Function: get the class number of each parent node of label tree for multi-task learning
    Params: label_tree: Dict, {...}
    """
    n_classes = {}
    for key, value in label_tree['taxonomy'].items():
        if key == 0:
            n_classes[key] = len(value)
        else:
            n_classes[key] = len(value) + 1

    return n_classes

def gen_A_parent_node(taxonomy):
    """
    Params-->node_num: the number of parent nodes, int
    Params-->taxonomy: the node relation, Dict
    """
    node_num = len(taxonomy)
    edges = []
    for key, values in taxonomy.items():
        if key == 0 or key == 2:
            for value in values:
                if value == 6:
                    value = 3
                if value == 7:
                    value = 4
                edges.append([key, value])
    edges = np.asarray(edges, dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_num, node_num),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    adj = adj.todense().A

    adj[adj == 0] = -np.inf
    adj[adj == 1] = 0

    return adj

def build_edge_index(taxonomy, add_self_edges=True):

    num_of_nodes = len(taxonomy)

    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()
    
    for src_node, neighboring_nodes in taxonomy.items():
        if src_node == 0 or src_node == 2:
            for trg_node in neighboring_nodes:
                if trg_node == 6:
                    trg_node = 3
                if trg_node == 7:
                    trg_node = 4
                if (src_node, trg_node) not in seen_edges:
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)

                    seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index

