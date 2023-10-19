#!/usr/bin/env python
# coding:utf-8

import math
import os
import pickle
import re
import time

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestCentroid
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import helper.logger as logger
from helper.data import DataPreprocess
from helper.data_loader import data_loaders
from helper.plot import annotate_heatmap, heatmap, plot_cm, plot_prc, plot_roc
from helper.utils import (build_edge_index, gen_A, gen_A_parent_node,
                          get_parent_node_number, load_checkpoint,
                          save_checkpoint)
from models.attention import PreSpatialAttn
from models.gat import GATlayer_IMP8
from models.grud_layer import GRUD_cell, GRUD_cells
from models.loss import MultitaskLoss, MultitaskWeightedLoss
from models.mlp import MLP
from train_modules.early_stopping import EarlyStopping
from train_modules.evaluation_metrics import evaluate4test, prc, roc
from train_modules.trainer import FlatTrainer, GlobalTrainer, LocalTrainer


class PreAttnMMs_GAT_IMP8_GC(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes, adj):
        super(PreAttnMMs_GAT_IMP8_GC, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes
        self.adj = torch.tensor(adj, dtype=torch.float).to(self.device)

        # static encoder
        self.static_encoder = MLP(config, n_features)

        # temporal encoder
        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)

            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features

        self.transformation_0 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_1 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_2 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_3 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_4 = nn.Linear(self.output_dim, self.output_dim)

        self.batchnorm_0 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_1 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_2 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_3 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_4 = nn.BatchNorm1d(self.output_dim)

        self.gatlayer = GATlayer_IMP8(config, self.output_dim)

        if config.model.gat_layer.is_concat:
            self.gat_out_features = config.model.gat_layer.num_out_features * config.model.gat_layer.num_of_heads
        else:
            self.gat_out_features = config.model.gat_layer.num_out_features

        # head 0
        self.classifier_0 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[0])
        # head 1
        self.classifier_1 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[1])
        # head 2
        self.classifier_2 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[2])
        # head 3
        self.classifier_3 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[6])
        # head 4
        self.classifier_4 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[7])

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        # (B, FIN)
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # gat
        # (B, FIN)
        x_0 = self.batchnorm_0(F.relu(self.transformation_0(embedding)))
        x_1 = self.batchnorm_1(F.relu(self.transformation_1(embedding)))
        x_2 = self.batchnorm_2(F.relu(self.transformation_2(embedding)))
        x_3 = self.batchnorm_3(F.relu(self.transformation_3(embedding)))
        x_4 = self.batchnorm_4(F.relu(self.transformation_4(embedding)))
        in_node_features = torch.concat([x_0.unsqueeze(1), x_1.unsqueeze(1), x_2.unsqueeze(1), x_3.unsqueeze(1), x_4.unsqueeze(1)], dim=1)

        # (B, N, FIN)-->(B, N, NH * FOUT) OR (B, N, FOUT)
        gat_output, adj = self.gatlayer(in_node_features, self.adj)

        # concatenation
        output_0 = torch.concat([x_0.detach(), gat_output[:, 0, :]], dim=-1)
        output_1 = torch.concat([x_1.detach(), gat_output[:, 1, :]], dim=-1)
        output_2 = torch.concat([x_2.detach(), gat_output[:, 2, :]], dim=-1)
        output_3 = torch.concat([x_3.detach(), gat_output[:, 3, :]], dim=-1)
        output_4 = torch.concat([x_4.detach(), gat_output[:, 4, :]], dim=-1)

        head0 = self.classifier_0(output_0)
        head1 = self.classifier_1(output_1)
        head2 = self.classifier_2(output_2)
        head3 = self.classifier_3(output_3)
        head4 = self.classifier_4(output_4)

        return [head0, head1, head2, head3, head4], gat_output

class PreAttnMMs_MTL_IMP3(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_MTL_IMP3, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size

        self.static_encoder = MLP(config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)
            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features
            
        # head 0
        self.fc_0 = nn.Linear(self.output_dim, config.model.mtl_head_block.head0.hidden_dim)
        self.dropout_0 = nn.Dropout(config.model.mtl_head_block.head0.dropout)
        self.classifier_0 = nn.Linear(config.model.mtl_head_block.head0.hidden_dim, 
                                n_classes[0])
        # head 1
        self.fc_1 = nn.Linear(self.output_dim, config.model.mtl_head_block.head1.hidden_dim)
        self.dropout_1 = nn.Dropout(config.model.mtl_head_block.head1.dropout)
        self.classifier_1 = nn.Linear(config.model.mtl_head_block.head1.hidden_dim, 
                                n_classes[1])
        # head 2
        self.fc_2 = nn.Linear(self.output_dim, config.model.mtl_head_block.head2.hidden_dim)
        self.dropout_2 = nn.Dropout(config.model.mtl_head_block.head2.dropout)
        self.classifier_2 = nn.Linear(config.model.mtl_head_block.head2.hidden_dim, 
                                n_classes[2])
        # head 3
        self.fc_3 = nn.Linear(self.output_dim, config.model.mtl_head_block.head3.hidden_dim)
        self.dropout_3 = nn.Dropout(config.model.mtl_head_block.head3.dropout)
        self.classifier_3 = nn.Linear(config.model.mtl_head_block.head3.hidden_dim, 
                                n_classes[6])
        # head 4
        self.fc_4 = nn.Linear(self.output_dim, config.model.mtl_head_block.head4.hidden_dim)
        self.dropout_4 = nn.Dropout(config.model.mtl_head_block.head4.dropout)
        self.classifier_4 = nn.Linear(config.model.mtl_head_block.head4.hidden_dim, 
                                n_classes[7])
    
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # multi-task block
        x0 = F.relu(self.fc_0(embedding))
        x0 = self.dropout_0(x0)
        head0 = self.classifier_0(x0)

        x1 = F.relu(self.fc_1(embedding))
        x1 = self.dropout_1(x1)
        head1 = self.classifier_1(x1)

        x2 = F.relu(self.fc_2(embedding))
        x2 = self.dropout_2(x2)
        head2 = self.classifier_2(x2)

        x3 = F.relu(self.fc_3(embedding))
        x3 = self.dropout_3(x3)
        head3 = self.classifier_3(x3)

        x4 = F.relu(self.fc_4(embedding))
        x4 = self.dropout_4(x4)
        head4 = self.classifier_4(x4)

        # concat the embeddings of each head
        head_embeddings = torch.concat([x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)], dim=1)
        
        return [head0, head1, head2, head3, head4], head_embeddings

def single_distance(clusters, cluster_num):
    """
    Function: calculate the single linkage between different clusters
    clusters: List, [[[[node_dimension], ...]],[],[],[],[]], len()=5
    cluster_num: 5
    """
    distances = {}
    # for every cluster
    for cluster_id, cluster in enumerate(clusters):
        # for each cluster after the current one
        for cluster2_id, cluster2 in enumerate(clusters[(cluster_id+1):]):
            closest_distance = math.inf
            # for each point in each cluster
            for point_id, point in enumerate(cluster):
                # go through every point in this cluster as well
                for point2_id, point2 in enumerate(cluster2):
                    if distance.euclidean(point, point2) < closest_distance:
                        # Only used for comparing 
                        closest_distance = distance.euclidean(point,point2)
        
            distances["({}, {})".format(cluster_id, cluster_id+cluster2_id+1)] = closest_distance

    return distances

def complete_distance(clusters, cluster_num):
    """
    Function: calculate the complete linkage between different clusters
    clusters: List, [[[[node_dimension], ...]],[],[],[],[]], len()=5
    cluster_num: 5
    """
    distances = {}
    # for every cluster
    for cluster_id, cluster in enumerate(clusters):
        # for each cluster after the current one
        for cluster2_id, cluster2 in enumerate(clusters[(cluster_id+1):]):
            furthest_distance = -math.inf
            # for each point in each cluster
            for point_id, point in enumerate(cluster):
                # go through every point in this cluster as well
                for point2_id, point2 in enumerate(cluster2):
                    if distance.euclidean(point, point2) > furthest_distance:
                        # Only used for comparing 
                        furthest_distance = distance.euclidean(point, point2)

            distances["({}, {})".format(cluster_id, cluster_id+cluster2_id+1)] = furthest_distance

    return distances

def average_distance(clusters, cluster_num):
    """
    Function: calculate the average linkage between different clusters
    clusters: List, [[[[node_dimension], ...]],[],[],[],[]], len()=5
    cluster_num: 5
    """
    distances = {}
    # for every cluster
    for cluster_id, cluster in enumerate(clusters):
        # for each cluster after the current one
        for cluster2_id, cluster2 in enumerate(clusters[(cluster_id+1):]):
            all_distances = []
            # for each point in each cluster
            for point_id, point in enumerate(cluster):
                # go through every point in this cluster as well
                for point2_id, point2 in enumerate(cluster2):
                    all_distances.append(distance.euclidean(point, point2))

            distances["({}, {})".format(cluster_id, cluster_id+cluster2_id+1)] = np.asarray(all_distances).mean()

    return distances

def centroid_distance(nodes_embeddings, nodes_labels, cluster_num):
    
    clf = NearestCentroid()
    clf.fit(nodes_embeddings, nodes_labels)
    centroids = clf.centroids_

    distances = {}
    # for every cluster
    for cluster_id in range(cluster_num):
        # for each cluster after the current one
        for cluster2_id in range(cluster_id+1, cluster_num):

            centroid_distance = distance.euclidean(centroids[cluster_id], centroids[cluster2_id])

            distances["({}, {})".format(cluster_id, cluster2_id)] = centroid_distance

    return distances

def run_visualization(config, _run):
    """
    Function: to visualize the node embedding and attention
    :param config: Object
    :param _run: the run object of this experiment
    """
    # Load the preprocessed data
    dp = DataPreprocess(config)
    data, label, indices = dp.load()

    # define DataLoader
    train_loader, validation_loader, test_loader = data_loaders(config, data, label, indices)

    # get the number of classes (!!only for model with our task decomposition strategy (i.e., one classifier per parent node))
    n_classes = get_parent_node_number(label)

    # define the adjacent matrix
    adj = gen_A_parent_node(label['taxonomy'])
    
    # define the model
    if config.model.type in ['PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
        model = PreAttnMMs_GAT_IMP8_GC(config, 
                                        data['X_t_steps'], 
                                        data['X_t_features'],
                                        data['X_features'],
                                        n_classes,
                                        adj)
    elif config.model.type == "PreAttnMMs_MTL_IMP3":
        model = PreAttnMMs_MTL_IMP3(config,
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes)
    model.to(config.train.device_setting.device)

    # define the loss function and optimizer
    if config.model.type == 'PreAttnMMs_MTL_IMP3':
        criterion = MultitaskLoss(n_classes)
    elif config.model.type == 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss':
        criterion = MultitaskWeightedLoss(n_classes)

    if config.train.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr = config.train.optimizer.learning_rate,
            weight_decay=config.train.optimizer.weight_decay
        )
    elif config.train.optimizer.type == "AdamW":
        optimizer = torch.optim.AdamW(
            params = model.parameters(),
            lr = config.train.optimizer.learning_rate,
            weight_decay=config.train.optimizer.weight_decay
        )
    elif config.train.optimizer.type == "Adagrad":
        optimizer = torch.optim.Adagrad(
            params = model.parameters(),
            lr = config.train.optimizer.learning_rate,
            weight_decay=config.train.optimizer.weight_decay
        )
    elif config.train.optimizer.type == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params = model.parameters(),
            lr = config.train.optimizer.learning_rate,
            weight_decay=config.train.optimizer.weight_decay
        )

    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='max', 
                                  factor=0.5, 
                                  patience=5, 
                                  verbose=True)
    
    if config.model.type == "PreAttnMMs_GAT_IMP8_GC_WeightedLoss":
        # checkpoint file path
        checkpoint_base = config.train.checkpoint.dir
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
        
        # get the best checkpoint .pt file
        checkpoints = []
        idx_checkpoint = np.array([])
        for i in os.listdir(checkpoint_dir):
            if "best_best_checkpoint" in i:
                checkpoints.append(i)
        for i in checkpoints:
            seachobj = re.search(r"\d+(?=\).pt)", i)
            idx_checkpoint = np.append(idx_checkpoint, int(seachobj.group()))
        target_model = checkpoints[np.argmax(idx_checkpoint)]

        logger.info("Loading the checkpoint --> {} for {}".format(target_model, config.model.type))

        # reload the checkpoint file and run on test Dataset
        best_epoch_model_file = os.path.join(checkpoint_dir, target_model)
        if os.path.isfile(best_epoch_model_file):
            best_performance, config = load_checkpoint(best_epoch_model_file, 
                                                        model=model,
                                                        config=config,
                                                        optimizer=optimizer)

        # evaluate the model
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": [],
                        "node_embeddings": [],
                        "attention_weights": []}

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                for key, value in batch.items():
                    batch[key] = value.to(config.train.device_setting.device)
                
                logits, batch_nodes_embeddings = model(batch)

                # (BATCH, NODE, DIM)->(64, 5, 240)
                batch_nodes_embeddings = batch_nodes_embeddings.cpu().numpy()

                loss = criterion(logits, batch['label'])
                
                # (BATCH, NH, NODE, NODE)-->(64, 5, 5, 5)
                all_attention_weights = model.gatlayer.attention_weights.cpu().numpy()

                for i in range(len(n_classes)):
                    prediction = F.softmax(logits[i], dim=1)
                    if idx == 0:
                        predict_info['predict_probas'].append([])
                        predict_info["predict_labels"].append([])
                        predict_info['target_labels'].append([])
                    predict_info['predict_probas'][i].extend(prediction.cpu().tolist())
                    predict_info["predict_labels"][i].extend(prediction.max(1)[-1].cpu().tolist())
                    predict_info["target_labels"][i].extend(batch['label'][i].cpu().tolist())

                predict_info['node_embeddings'].append(batch_nodes_embeddings)
                predict_info['attention_weights'].append(all_attention_weights)

        # set results dir
        result_base = config.test.results.dir
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
        
        """
        visualize the node embeddings and calculate different cluster distance
        """
        # visualize the node embeddings
        # node_labels = [0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4,...]
        node_labels = np.tile(np.array([int(i) for i in range(len(n_classes))]), len(predict_info['node_embeddings'])* config.train.batch_size)
        # (33, 64, 5, 320)-->(10560, 320)
        nodes_embeddings = np.stack(predict_info['node_embeddings'], axis=0).reshape(-1, predict_info['node_embeddings'][0].shape[-1])
        
        """
        more information about T-SNE can be seen in: https://distill.pub/2016/misread-tsne/
        """
        t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(nodes_embeddings)

        # Used whenever we need to visualzie points from different classes (t-SNE, CORA visualization)
        node_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "gray"}
        labels = {0: "Task Representations for Parent Node 0",
                  1: "Task Representations for Parent Node 1",
                  2: "Task Representations for Parent Node 2",
                  3: "Task Representations for Parent Node 6",
                  4: "Task Representations for Parent Node 7"}
        fig, ax = plt.subplots()
        for class_id in range(len(n_classes)):
            ax.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=10, color=node_label_to_color_map[class_id], edgecolors='black', linewidths=0.2, label=labels[class_id])
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "node_embedding_scatters_(nc={}, perp={}).png".format(2, 30)), dpi=400)
        np.save(os.path.join(result_dir, "t_sne_embeddings.npy"), t_sne_embeddings)

        # 2 - calculate different cluster distance based on low-dimensional features after T-SNE
        init_clusters_tsne = []
        for cluster_idx in range(len(n_classes)):
            init_clusters_tsne.append(t_sne_embeddings[node_labels == cluster_idx].tolist())

        # calculate and save all kinds of distances
        sd_tsne = single_distance(init_clusters_tsne, len(n_classes))
        cd_tsne = complete_distance(init_clusters_tsne, len(n_classes))
        ad_tsne = average_distance(init_clusters_tsne, len(n_classes))
        cent_d_tsne = centroid_distance(t_sne_embeddings, node_labels, len(n_classes))
        cluster_distances_tsne = {"single_distance": sd_tsne,
                             "complete_distance": cd_tsne,
                             "average_distance": ad_tsne,
                             "centroid_distance": cent_d_tsne}
        np.save(os.path.join(result_dir, "cluster_distances_tsne.npy"), cluster_distances_tsne)

        """
        visualize the attention weights
        https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        """
        # (sample_size, NH, 5, 5)
        attention_weights = np.stack(predict_info['attention_weights'], axis=0).reshape(-1, predict_info['attention_weights'][0].shape[1], 5, 5)

        # get the exact match sample index
        predict_labels = np.asarray(predict_info["predict_labels"])
        target_labels = np.asarray(predict_info["target_labels"])
        sample_size = predict_labels.shape[1]

        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))

        # transform label_target and label_pred into binarized label, i.e. [[1,0,1,0,0,0,0,0,0,0,0],...]
        for i in range(sample_size):
            for index, j in enumerate(target_labels[:, i]):
                if index == 0:
                    if j == 0:
                        label_target[i, 0] = 1
                        label_target[i, 1] = 0
                    elif j == 1:
                        label_target[i, 0] = 0
                        label_target[i, 1] = 1
                elif index == 1:
                    if j == 0:
                        label_target[i, 2] = 1
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                    elif j == 1:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 1
                        label_target[i, 4] = 0
                    elif j == 2:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 1
                    elif j == 3:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                elif index == 2:
                    if j == 0:
                        label_target[i, 5] = 1
                        label_target[i, 6] = 0
                    elif j == 1:
                        label_target[i, 5] = 0
                        label_target[i, 6] = 1
                    elif j == 2:
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                elif index == 3:
                    if j == 0:
                        label_target[i, 7] = 1
                        label_target[i, 8] = 0
                    elif j == 1:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 1
                    elif j == 2:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                elif index == 4:
                    if j == 0:
                        label_target[i, 9] = 1
                        label_target[i, 10] = 0
                    elif j == 1:
                        label_target[i, 9] = 0
                        label_target[i, 10] = 1
                    elif j == 2:
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0

            for index, j in enumerate(predict_labels[:, i]):
                if index == 0:
                    if j == 0:
                        label_pred[i, 0] = 1
                        label_pred[i, 1] = 0
                    elif j == 1:
                        label_pred[i, 0] = 0
                        label_pred[i, 1] = 1
                elif index == 1:
                    if j == 0:
                        label_pred[i, 2] = 1
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                    elif j == 1:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 1
                        label_pred[i, 4] = 0
                    elif j == 2:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 1
                    elif j == 3:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                elif index == 2:
                    if j == 0:
                        label_pred[i, 5] = 1
                        label_pred[i, 6] = 0
                    elif j == 1:
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 1
                    elif j == 2:
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                elif index == 3:
                    if j == 0:
                        label_pred[i, 7] = 1
                        label_pred[i, 8] = 0
                    elif j == 1:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 1
                    elif j == 2:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                elif index == 4:
                    if j == 0:
                        label_pred[i, 9] = 1
                        label_pred[i, 10] = 0
                    elif j == 1:
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 1
                    elif j == 2:
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
    
        # 获取完全预测正确和非完全预测正确的样本index
        all_correct_idx = []
        for index, value in enumerate(label_target):
            if (label_pred[index,:] == label_target[index,:]).all():
                all_correct_idx.append(index)

        # 求所有预测正确的样本的，所有head的注意力均值
        attention_head = attention_weights[all_correct_idx].mean(axis=0)
        attention_avg = attention_weights[all_correct_idx].mean(axis=0).mean(axis=0)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 6))
        y_axis = ["node 0", "node 1", "node 2", "node 6", "node 7"]
        x_axis = ["node 0", "node 1", "node 2", "node 6", "node 7"]

        # attention head 1
        im, _ = heatmap(attention_head[0], y_axis, x_axis, ax=ax1, vmin=0, vmax=1,
                        cmap="OrRd", cbarlabel="attention weights")
        annotate_heatmap(im, valfmt="{x:.4f}", size=8)

        # attention head 2
        im, _ = heatmap(attention_head[1], y_axis, x_axis, ax=ax2, vmin=0, vmax=1,
                cmap="YlGn", cbarlabel="attention weights")
        annotate_heatmap(im, valfmt="{x:.4f}", size=8)

        # attention head 3
        im, _ = heatmap(attention_head[2], y_axis, x_axis, ax=ax3, vmin=0, vmax=1,
                cmap="GnBu", cbarlabel="attention weights")
        annotate_heatmap(im, valfmt="{x:.4f}", size=8)
        
        # attention head 4
        im, _ = heatmap(attention_head[3], y_axis, x_axis, ax=ax4, vmin=0, vmax=1,
                cmap="GnBu", cbarlabel="attention weights")
        annotate_heatmap(im, valfmt="{x:.4f}", size=8)
        
        # attention head 5
        im, _ = heatmap(attention_head[4], y_axis, x_axis, ax=ax5, vmin=0, vmax=1,
                cmap="GnBu", cbarlabel="attention weights")
        annotate_heatmap(im, valfmt="{x:.4f}", size=8)

        # attention avg
        im, cbar = heatmap(attention_avg, y_axis, x_axis, ax=ax6, vmin=0, vmax=1,
                        cmap="magma_r", cbarlabel="attention weights")
        annotate_heatmap(im, valfmt="{x:.4f}", size=8)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "attention_weights(heads).png"), dpi=400)
        
        attention_weights = {"attention_weights": attention_head,
                             "attention_weights_avg": attention_avg}
        
        np.save(os.path.join(result_dir, "attention_weights.npy"), attention_weights)

    elif config.model.type == "PreAttnMMs_MTL_IMP3":

        # checkpoint file path
        checkpoint_base = config.train.checkpoint.dir
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    
        # get the best checkpoint .pt file
        checkpoints = []
        idx_checkpoint = np.array([])
        for i in os.listdir(checkpoint_dir):
            if "best_best_checkpoint" in i:
                checkpoints.append(i)
        for i in checkpoints:
            seachobj = re.search(r"\d+(?=\).pt)", i)
            idx_checkpoint = np.append(idx_checkpoint, int(seachobj.group()))
        target_model = checkpoints[np.argmax(idx_checkpoint)]

        logger.info("Loading the checkpoint --> {} for {}".format(target_model, config.model.type))
        
        # reload the checkpoint file and run on test Dataset
        best_epoch_model_file = os.path.join(checkpoint_dir, target_model)
        if os.path.isfile(best_epoch_model_file):
            best_performance, config = load_checkpoint(best_epoch_model_file, 
                                                       model=model,
                                                       config=config,
                                                       optimizer=optimizer)

        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": [],
                        "node_embeddings": [],
                        "attention_weights": []}

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                for key, value in batch.items():
                    batch[key] = value.to(config.train.device_setting.device)
                
                logits, batch_nodes_embeddings = model(batch)

                # (BATCH, NODE, DIM)->(64, 5, 240)
                batch_nodes_embeddings = batch_nodes_embeddings.cpu().numpy()

                loss = criterion(logits, batch['label'])

                for i in range(len(n_classes)):
                    prediction = F.softmax(logits[i], dim=1)
                    if idx == 0:
                        predict_info['predict_probas'].append([])
                        predict_info["predict_labels"].append([])
                        predict_info['target_labels'].append([])
                    predict_info['predict_probas'][i].extend(prediction.cpu().tolist())
                    predict_info["predict_labels"][i].extend(prediction.max(1)[-1].cpu().tolist())
                    predict_info["target_labels"][i].extend(batch['label'][i].cpu().tolist())

                predict_info['node_embeddings'].append(batch_nodes_embeddings)

        # set results dir
        result_base = config.test.results.dir
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
            
        # visualize the node embeddings
        node_labels = np.tile(np.array([int(i) for i in range(len(n_classes))]), len(predict_info['node_embeddings'])* config.train.batch_size)
        nodes_embeddings = np.stack(predict_info['node_embeddings'], axis=0).reshape(-1, predict_info['node_embeddings'][0].shape[-1])
        
        """
        more information about T-SNE can be seen in: https://distill.pub/2016/misread-tsne/
        """
        t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(nodes_embeddings)

        # Used whenever we need to visualzie points from different classes (t-SNE, CORA visualization)
        node_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "gray"}
        labels = {0: "Task Representations for Parent Node 0",
                  1: "Task Representations for Parent Node 1",
                  2: "Task Representations for Parent Node 2",
                  3: "Task Representations for Parent Node 6",
                  4: "Task Representations for Parent Node 7"}
        
        fig, ax = plt.subplots()
        for class_id in range(len(n_classes)):
            ax.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=10, color=node_label_to_color_map[class_id], edgecolors='black', linewidths=0.2, label=labels[class_id])
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "node_embedding_scatters_(nc={}, perp={}).png".format(2, 30)), dpi=400)
        np.save(os.path.join(result_dir, "t_sne_embeddings.npy"), t_sne_embeddings)
        
        # # 2 - calculate different cluster distance based on low-dimensional features after T-SNE
        init_clusters_tsne = []
        for cluster_idx in range(len(n_classes)):
            init_clusters_tsne.append(t_sne_embeddings[node_labels == cluster_idx].tolist())

        # calculate and save all kinds of distances
        sd_tsne = single_distance(init_clusters_tsne, len(n_classes))
        cd_tsne = complete_distance(init_clusters_tsne, len(n_classes))
        ad_tsne = average_distance(init_clusters_tsne, len(n_classes))
        cent_d_tsne = centroid_distance(t_sne_embeddings, node_labels, len(n_classes))
        cluster_distances_tsne = {"single_distance": sd_tsne,
                                  "complete_distance": cd_tsne,
                                  "average_distance": ad_tsne,
                                  "centroid_distance": cent_d_tsne}
        np.save(os.path.join(result_dir, "cluster_distances_tsne.npy"), cluster_distances_tsne)
