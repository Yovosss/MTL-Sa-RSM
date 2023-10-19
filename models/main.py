#!/usr/bin/env python
# coding:utf-8

import os
import pickle
import re
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler

import helper.logger as logger
from helper.data import DataPreprocess
from helper.data_loader import data_loaders

from helper.collator import FlatCollator, GlobalCollator, LocalCollator
from helper.dataset import FlatDataset, GlobalHcDataset, LocalHcDataset
from helper.plot import plot_cm, plot_prc, plot_roc
from helper.utils import (build_edge_index, gen_A, gen_A_parent_node,
                          get_parent_node_number, load_checkpoint,
                          save_checkpoint)
from models.loss import MultitaskLoss, MultitaskWeightedLoss
from models.model import (PreAttnMMs, PreAttnMMs_FCAN, PreAttnMMs_FCLN,
                          PreAttnMMs_GAT_IMP8_GC, PreAttnMMs_GCN_MAP, PreAttnMMs_HMCN,
                          PreAttnMMs_MTL_IMP2, PreAttnMMs_MTL_LCL)
from train_modules.early_stopping import EarlyStopping
from train_modules.evaluation_metrics import evaluate4test, prc, roc
from train_modules.trainer import FlatTrainer, GlobalTrainer, LocalTrainer


def run_experiment(config, _run):
    """
    :param config: Object
    :param _run: the run object of this experiment
    """
    # Load the preprocessed data
    dp = DataPreprocess(config)
    data, label, indices = dp.load()

    # define DataLoader
    train_loader, validation_loader, test_loader = data_loaders(config, data, label, indices)

    # global mode
    if config.model.type == 'PreAttnMMs_GCN_MAP_V1':
        n_classes = label['unique_label_number']
    elif config.model.type in ['PreAttnMMs_MTL_IMP2', 'PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
        n_classes = get_parent_node_number(label)
    elif config.model.type == 'PreAttnMMs_HMCN':
        local_output_size = [2, 5, 4]
        globbal_output_size = label['unique_label_number'] - 1
        n_classes = (local_output_size, globbal_output_size)
    elif config.model.type == 'PreAttnMMs_MTL_LCL':
        n_classes = [2, 5, 7]
    # flat mode
    elif config.model.type == 'PreAttnMMs_FCLN':
        n_classes = label['unique_label_number'] - len(label['taxonomy'].keys())
    elif config.model.type == 'PreAttnMMs_FCAN':
        n_classes = label['unique_label_number'] - 1
    else:
        # local mode
        n_classes = len(label['taxonomy'][config.experiment.local_task])

    # build up model
    if config.model.type == 'PreAttnMMs_LCPN':
        model = PreAttnMMs(config, 
                          data['X_t_steps'], 
                          data['X_t_features'],
                          data['X_features'],
                          n_classes)
    elif config.model.type == 'PreAttnMMs_FCLN':
        model = PreAttnMMs_FCLN(config, 
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_FCAN':
        model = PreAttnMMs_FCAN(config, 
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_HMCN':
        model = PreAttnMMs_HMCN(config,
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_MTL_LCL':
        model = PreAttnMMs_MTL_LCL(config,
                                   data['X_t_steps'], 
                                   data['X_t_features'],
                                   data['X_features'],
                                   n_classes)
    elif config.model.type =='PreAttnMMs_MTL_IMP2' :
        model = PreAttnMMs_MTL_IMP2(config,
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes)
    elif config.model.type == 'PreAttnMMs_GCN_MAP_V1':
        adj = gen_A(label['unique_label_number'], label['taxonomy'])
        model = PreAttnMMs_GCN_MAP(config, 
                                   data['X_t_steps'], 
                                   data['X_t_features'],
                                   data['X_features'],
                                   n_classes,
                                   adj)
    elif config.model.type in ['PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
            adj = gen_A_parent_node(label['taxonomy'])
            model = PreAttnMMs_GAT_IMP8_GC(config, 
                                        data['X_t_steps'], 
                                        data['X_t_features'],
                                        data['X_features'],
                                        n_classes,
                                        adj)
    model.to(config.train.device_setting.device)

    # define training objective & optimizer & lr_scheduler
    if config.model.type in ['PreAttnMMs_GCN_MAP_V1', 'PreAttnMMs_FCAN']:
        """
        For multi-label prediction
        """
        criterion = nn.MultiLabelSoftMarginLoss()
    elif config.model.type in ["PreAttnMMs_MTL_LCL", "PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC"]:
        criterion = MultitaskLoss(n_classes)
    elif config.model.type in ["PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        criterion = MultitaskWeightedLoss(n_classes)
    elif config.model.type == "PreAttnMMs_HMCN":
        criterion = nn.BCELoss()
    else:
        criterion = nn.NLLLoss()

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
    elif config.train.optimizer.type == "SGD":
        optimizer = torch.optim.SGD(
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

    # define trainer
    if config.experiment.type == 'flat':
        trainer = FlatTrainer(model,
                               criterion,
                               optimizer,
                               scheduler,
                               config,
                               data['X_t_steps'], 
                               data['X_t_features'],
                               n_classes)
    elif config.experiment.type == 'local':
        trainer = LocalTrainer(model,
                               criterion,
                               optimizer,
                               scheduler,
                               config,
                               data['X_t_steps'], 
                               data['X_t_features'],
                               n_classes)
    elif config.experiment.type == 'global':
        trainer = GlobalTrainer(model,
                                criterion,
                                optimizer,
                                scheduler,
                                config,
                                data['X_t_steps'], 
                                data['X_t_features'],
                                n_classes)

    # set checkpoint
    checkpoint_base = config.train.checkpoint.dir
    if config.experiment.type == "flat":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'no_hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    elif config.experiment.type == "local":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'no_hp_tuning', 
                                    config.model.type,
                                    'node-{}'.format(config.experiment.local_task),
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    elif config.experiment.type == "global":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'no_hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)

    if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.train.early_stopping.patience, 
                                    path=os.path.join(checkpoint_dir, 'best_checkpoint.pt'),
                                    verbose=True)

    # train
    best_epoch = -1
    best_performance = 0.0
    log = {"train": {"loss": [],
                     "metric": [],
                     "predict_info": []},
           "valid": {"loss": [],
                     "metric": [],
                     "predict_info": []},
            "test": {"loss": [],
                     "metric": [],
                     "predict_info": []},
           "best_epoch": None,
           "best_performance": None}

    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        start_time = time.time()
        train_loss, train_metrics, train_predicts = trainer.train(train_loader, epoch)
        valid_loss, valid_metrics, valid_predicts = trainer.eval(validation_loader, epoch, "VALIDATION")


        monitored_metric = config.experiment.monitored_metric

        if valid_metrics[monitored_metric] > best_performance:

            best_epoch = epoch
            best_performance = valid_metrics[monitored_metric]

        log["train"]["loss"].append(train_loss)
        log["train"]["metric"].append({"epoch {0}".format(epoch): train_metrics})
        log["train"]["predict_info"].append({"epoch {0}".format(epoch): train_predicts})
        log["valid"]["loss"].append(valid_loss)
        log["valid"]["metric"].append({"epoch {0}".format(epoch): valid_metrics})
        log["valid"]["predict_info"].append({"epoch {0}".format(epoch): valid_predicts})
        log["best_epoch"] = best_epoch
        log["best_performance"] = best_performance

        # early_stopping needs the validation AUC to check if it has incresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_metrics[monitored_metric], model, optimizer, best_epoch, best_performance, monitored_metric)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        logger.info('Epoch {}: --- Time Cost {} secs. \n'.format(epoch, time.time() - start_time))

    best_epoch_model_file = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
    if os.path.isfile(best_epoch_model_file):
        best_performance, config = load_checkpoint(best_epoch_model_file, 
                                                   model=model,
                                                   config=config,
                                                   optimizer=optimizer)
        test_loss, test_metrics, test_predicts = trainer.eval(test_loader, config.train.start_epoch, 'TEST')

        log["test"]["loss"].append(test_loss)
        log["test"]["metric"].append({"epoch {0}".format(best_epoch): test_metrics})
        log["test"]["predict_info"].append({"epoch {0}".format(best_epoch): test_predicts})

    else:
        logger.error("There is no checkpoint datafile can be loaded for TEST!!")

    # save the model training related information
    with open(os.path.join(checkpoint_dir, 'logfile.pkl'), 'wb') as f:
        pickle.dump(log, f)

    # set results dir
    result_base = config.test.results.dir
    if config.experiment.type == "flat":
        result_dir = os.path.join(result_base, 
                                  'no_hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
    elif config.experiment.type == "local":
        result_dir = os.path.join(result_base, 
                                  'no_hp_tuning', 
                                  config.model.type,
                                  'node-{}'.format(config.experiment.local_task),
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)

    elif config.experiment.type == "global":
        result_dir = os.path.join(result_base, 
                                  'no_hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
    if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

    # plot and save the results
    if config.model.type == "PreAttnMMs_LCPN":
        """
        code below is only for each local task, not for local hierarchical classification, 
        more info can be seen at 2_Test_LCPN.ipynb.
        """
        roc_params = roc(n_classes=n_classes,
                        target_labels=test_predicts['target_labels'],
                        predict_probs=np.asarray(test_predicts['predict_probas']))

        prc_params = prc(n_classes=n_classes,
                        target_labels=test_predicts['target_labels'],
                        predict_probs=np.asarray(test_predicts['predict_probas']),
                        predict_labels=test_predicts["predict_labels"])
        
        plot_roc(result_dir, roc_params, config.experiment.local_task)
        plot_prc(result_dir, prc_params, config.experiment.local_task)
        plot_cm(result_dir, test_predicts['target_labels'], test_predicts["predict_labels"], config.experiment.local_task)

        # save all results
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "roc_params": roc_params,
                   "prc_params": prc_params}
        # caution: the hierarchical test results can be found at `/data/wzx/HC4FUOV2/results/hp_tuning/PreAttnMMs_LCPN/test_results.npy`
    
    elif config.model.type == "PreAttnMMs_FCLN":
        target_labels_array = np.zeros((len(test_predicts['target_labels']), 11))
        predcit_labels_array = np.zeros((len(test_predicts['target_labels']), 11))
        
        # reconstruct the target labels into binary form
        for i in range(target_labels_array.shape[0]):
            if test_predicts['target_labels'][i] == 0:
                target_labels_array[i, 0] = 1
                target_labels_array[i, 2] = 1
            elif test_predicts['target_labels'][i] == 1:
                target_labels_array[i, 0] = 1
                target_labels_array[i, 3] = 1
            elif test_predicts['target_labels'][i] == 2:
                target_labels_array[i, 0] = 1
                target_labels_array[i, 4] = 1
            elif test_predicts['target_labels'][i] == 3:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 5] = 1
                target_labels_array[i, 7] = 1
            elif test_predicts['target_labels'][i] == 4:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 5] = 1
                target_labels_array[i, 8] = 1
            elif test_predicts['target_labels'][i] == 5:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 6] = 1
                target_labels_array[i, 9] = 1
            elif test_predicts['target_labels'][i] == 6:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 6] = 1
                target_labels_array[i, 10] = 1
            else:
                logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

        # reconstruct the predict labels into binary form
        for i in range(predcit_labels_array.shape[0]):
            if test_predicts['predict_labels'][i] == 0:
                predcit_labels_array[i, 0] = 1
                predcit_labels_array[i, 2] = 1
            elif test_predicts['predict_labels'][i] == 1:
                predcit_labels_array[i, 0] = 1
                predcit_labels_array[i, 3] = 1
            elif test_predicts['predict_labels'][i] == 2:
                predcit_labels_array[i, 0] = 1
                predcit_labels_array[i, 4] = 1
            elif test_predicts['predict_labels'][i] == 3:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 5] = 1
                predcit_labels_array[i, 7] = 1
            elif test_predicts['predict_labels'][i] == 4:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 5] = 1
                predcit_labels_array[i, 8] = 1
            elif test_predicts['predict_labels'][i] == 5:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 6] = 1
                predcit_labels_array[i, 9] = 1
            elif test_predicts['predict_labels'][i] == 6:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 6] = 1
                predcit_labels_array[i, 10] = 1
            else:
                logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

        # cal evaluate4test
        hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    elif config.model.type == "PreAttnMMs_GCN_MAP_V1":
        target_labels_array = np.asarray(test_predicts['target_labels'])[:,1:]
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])[:,1:]
        hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        
        target_labels_array = np.asarray(test_predicts['target_labels'])
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])

        sample_size = target_labels_array.shape[1]
        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))
        # transform label_target and label_pred into binarized label, i.e. [[1,0,1,0,0,0,0,0,0,0,0],...]
        for i in range(sample_size):
            for index, j in enumerate(target_labels_array[:, i]):
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

            for index, j in enumerate(predcit_labels_array[:, i]):
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

        hier_metrics = evaluate4test(config, label_target, label_pred)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    elif config.model.type == "PreAttnMMs_MTL_LCL":
        target_labels_array = np.asarray(test_predicts['target_labels'])
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])

        sample_size = target_labels_array.shape[1]
        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))
        for i in range(sample_size):
            for index, j in enumerate(target_labels_array[:, i]):
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
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 1:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 1
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 2:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 1
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 3:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 1
                        label_target[i, 6] = 0
                    elif j == 4:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 1
                elif index == 2:
                    if j == 0:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 1:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 2:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 3:
                        label_target[i, 7] = 1
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 4:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 1
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 5:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 1
                        label_target[i, 10] = 0
                    elif j == 6:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 1

            for index, j in enumerate(predcit_labels_array[:, i]):
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
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 1:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 1
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 2:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 1
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 3:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 1
                        label_pred[i, 6] = 0
                    elif j == 4:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 1
                elif index == 2:
                    if j == 0:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 1:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 2:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 3:
                        label_pred[i, 7] = 1
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 4:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 1
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 5:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 1
                        label_pred[i, 10] = 0
                    elif j == 6:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 1

        hier_metrics = evaluate4test(config, label_target, label_pred)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}

    else:
        target_labels_array = np.asarray(test_predicts['target_labels'])
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])
        hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    np.save(os.path.join(result_dir, "test_results.npy"), results)

    return test_metrics[monitored_metric]

def run_experiment_test(config, _run):
    """
    :param config: Object
    :param _run: the run object of this experiment
    """
    # Load the preprocessed data
    dp = DataPreprocess(config)
    data, label, indices = dp.load()

    # define DataLoader
    train_loader, validation_loader, test_loader = data_loaders(config, data, label, indices)

    # build up model
    """
    TODO: here need to justify the model structure based on the parameters of corresponding best checkpoints
    """
    # global mode
    if config.model.type == 'PreAttnMMs_GCN_MAP_V1':
        n_classes = label['unique_label_number']
    elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        n_classes = get_parent_node_number(label)
    elif config.model.type == 'PreAttnMMs_HMCN':
        local_output_size = [2, 5, 4]
        globbal_output_size = label['unique_label_number'] - 1
        n_classes = (local_output_size, globbal_output_size)
    elif config.model.type == 'PreAttnMMs_MTL_LCL':
        n_classes = [2, 5, 7]
    # flat mode
    elif config.model.type == 'PreAttnMMs_FCLN':
        n_classes = label['unique_label_number'] - len(label['taxonomy'].keys())
    elif config.model.type == 'PreAttnMMs_FCAN':
        n_classes = label['unique_label_number'] - 1
    else:
        # local mode
        n_classes = len(label['taxonomy'][config.experiment.local_task])

    # build up model
    if config.model.type == 'PreAttnMMs_LCPN':
        model = PreAttnMMs(config, 
                          data['X_t_steps'], 
                          data['X_t_features'],
                          data['X_features'],
                          n_classes)
    elif config.model.type == 'PreAttnMMs_FCLN':
        model = PreAttnMMs_FCLN(config, 
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_FCAN':
        model = PreAttnMMs_FCAN(config, 
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_HMCN':
        model = PreAttnMMs_HMCN(config,
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_MTL_IMP2':
            model = PreAttnMMs_MTL_IMP2(config, 
                                       data['X_t_steps'], 
                                       data['X_t_features'],
                                       data['X_features'],
                                       n_classes)
    elif config.model.type == 'PreAttnMMs_MTL_LCL':
        model = PreAttnMMs_MTL_LCL(config, 
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes)
    elif config.model.type == 'PreAttnMMs_GCN_MAP_V1':
        adj = gen_A(label['unique_label_number'], label['taxonomy'])
        model = PreAttnMMs_GCN_MAP(config, 
                                   data['X_t_steps'], 
                                   data['X_t_features'],
                                   data['X_features'],
                                   n_classes,
                                   adj)
    elif config.model.type in ['PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
        adj = gen_A_parent_node(label['taxonomy'])
        model = PreAttnMMs_GAT_IMP8_GC(config, 
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes,
                                    adj)
    
    model.to(config.train.device_setting.device)

    # define training objective & optimizer & lr_scheduler
    if config.model.type in ['PreAttnMMs_GCN_MAP_V1', 'PreAttnMMs_FCAN']:
        """
        For multi-label prediction
        """
        criterion = nn.MultiLabelSoftMarginLoss()
    elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC"]:
        criterion = MultitaskLoss(n_classes)
    elif config.model.type in ["PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        criterion = MultitaskWeightedLoss(n_classes)
    elif config.model.type == "PreAttnMMs_HMCN":
        criterion = nn.BCELoss()
    else:
        criterion = nn.NLLLoss()

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
    elif config.train.optimizer.type == "SGD":
        optimizer = torch.optim.SGD(
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

    # define trainer
    if config.experiment.type == 'flat':
        trainer = FlatTrainer(model,
                               criterion,
                               optimizer,
                               scheduler,
                               config,
                               data['X_t_steps'], 
                               data['X_t_features'],
                               n_classes)
    elif config.experiment.type == 'local':
        trainer = LocalTrainer(model,
                               criterion,
                               optimizer,
                               scheduler,
                               config,
                               data['X_t_steps'], 
                               data['X_t_features'],
                               n_classes)
    elif config.experiment.type == 'global':
        trainer = GlobalTrainer(model,
                                criterion,
                                optimizer,
                                scheduler,
                                config,
                                data['X_t_steps'], 
                                data['X_t_features'],
                                n_classes)

    # set checkpoint
    checkpoint_base = config.train.checkpoint.dir
    if config.experiment.type == "flat":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    elif config.experiment.type == "local":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'hp_tuning', 
                                    config.model.type,
                                    'node-{}'.format(config.experiment.local_task),
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    elif config.experiment.type == "global":
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
        test_loss, test_metrics, test_predicts = trainer.eval(test_loader, config.train.start_epoch, 'TEST')

    # set results dir
    result_base = config.test.results.dir
    if config.experiment.type == "flat":
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
    elif config.experiment.type == "local":
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  'node-{}'.format(config.experiment.local_task),
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)

    elif config.experiment.type == "global":
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
    if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

     # plot and save the results
    if config.model.type == "PreAttnMMs_LCPN":
        """
        code below is only for each local task, not for local hierarchical classification, 
        more info can be seen at 2_Test_LCPN.ipynb.
        """
        roc_params = roc(n_classes=n_classes,
                        target_labels=test_predicts['target_labels'],
                        predict_probs=np.asarray(test_predicts['predict_probas']))

        prc_params = prc(n_classes=n_classes,
                        target_labels=test_predicts['target_labels'],
                        predict_probs=np.asarray(test_predicts['predict_probas']),
                        predict_labels=test_predicts["predict_labels"])
        
        plot_roc(result_dir, roc_params, config.experiment.local_task)
        plot_prc(result_dir, prc_params, config.experiment.local_task)
        plot_cm(result_dir, test_predicts['target_labels'], test_predicts["predict_labels"], config.experiment.local_task)

        # save all results
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "roc_params": roc_params,
                   "prc_params": prc_params}
        # caution: the hierarchical test results can be found at `/data/wzx/HC4FUOV2/results/hp_tuning/PreAttnMMs_LCPN/test_results.npy`
    
    elif config.model.type == "PreAttnMMs_FCLN":
        target_labels_array = np.zeros((len(test_predicts['target_labels']), 11))
        predcit_labels_array = np.zeros((len(test_predicts['target_labels']), 11))
        
        # reconstruct the target labels into binary form
        for i in range(target_labels_array.shape[0]):
            if test_predicts['target_labels'][i] == 0:
                target_labels_array[i, 0] = 1
                target_labels_array[i, 2] = 1
            elif test_predicts['target_labels'][i] == 1:
                target_labels_array[i, 0] = 1
                target_labels_array[i, 3] = 1
            elif test_predicts['target_labels'][i] == 2:
                target_labels_array[i, 0] = 1
                target_labels_array[i, 4] = 1
            elif test_predicts['target_labels'][i] == 3:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 5] = 1
                target_labels_array[i, 7] = 1
            elif test_predicts['target_labels'][i] == 4:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 5] = 1
                target_labels_array[i, 8] = 1
            elif test_predicts['target_labels'][i] == 5:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 6] = 1
                target_labels_array[i, 9] = 1
            elif test_predicts['target_labels'][i] == 6:
                target_labels_array[i, 1] = 1
                target_labels_array[i, 6] = 1
                target_labels_array[i, 10] = 1
            else:
                logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

        # reconstruct the predict labels into binary form
        for i in range(predcit_labels_array.shape[0]):
            if test_predicts['predict_labels'][i] == 0:
                predcit_labels_array[i, 0] = 1
                predcit_labels_array[i, 2] = 1
            elif test_predicts['predict_labels'][i] == 1:
                predcit_labels_array[i, 0] = 1
                predcit_labels_array[i, 3] = 1
            elif test_predicts['predict_labels'][i] == 2:
                predcit_labels_array[i, 0] = 1
                predcit_labels_array[i, 4] = 1
            elif test_predicts['predict_labels'][i] == 3:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 5] = 1
                predcit_labels_array[i, 7] = 1
            elif test_predicts['predict_labels'][i] == 4:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 5] = 1
                predcit_labels_array[i, 8] = 1
            elif test_predicts['predict_labels'][i] == 5:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 6] = 1
                predcit_labels_array[i, 9] = 1
            elif test_predicts['predict_labels'][i] == 6:
                predcit_labels_array[i, 1] = 1
                predcit_labels_array[i, 6] = 1
                predcit_labels_array[i, 10] = 1
            else:
                logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

        # cal evaluate4test
        hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    elif config.model.type == "PreAttnMMs_GCN_MAP_V1":
        target_labels_array = np.asarray(test_predicts['target_labels'])[:,1:]
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])[:,1:]
        hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        
        target_labels_array = np.asarray(test_predicts['target_labels'])
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])

        sample_size = target_labels_array.shape[1]
        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))
        # transform label_target and label_pred into binarized label, i.e. [[1,0,1,0,0,0,0,0,0,0,0],...]
        for i in range(sample_size):
            for index, j in enumerate(target_labels_array[:, i]):
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

            for index, j in enumerate(predcit_labels_array[:, i]):
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

        hier_metrics = evaluate4test(config, label_target, label_pred)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": label_target,
                                "pred_labels": label_pred,
                                "metrics": hier_metrics}}
    elif config.model.type == "PreAttnMMs_MTL_LCL":
        target_labels_array = np.asarray(test_predicts['target_labels'])
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])

        sample_size = target_labels_array.shape[1]
        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))
        for i in range(sample_size):
            for index, j in enumerate(target_labels_array[:, i]):
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
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 1:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 1
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 2:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 1
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 3:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 1
                        label_target[i, 6] = 0
                    elif j == 4:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 1
                elif index == 2:
                    if j == 0:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 1:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 2:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 3:
                        label_target[i, 7] = 1
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 4:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 1
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 5:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 1
                        label_target[i, 10] = 0
                    elif j == 6:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 1

            for index, j in enumerate(predcit_labels_array[:, i]):
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
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 1:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 1
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 2:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 1
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 3:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 1
                        label_pred[i, 6] = 0
                    elif j == 4:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 1
                elif index == 2:
                    if j == 0:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 1:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 2:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 3:
                        label_pred[i, 7] = 1
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 4:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 1
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 5:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 1
                        label_pred[i, 10] = 0
                    elif j == 6:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 1
        
        hier_metrics = evaluate4test(config, label_target, label_pred)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": label_target,
                                "pred_labels": label_pred,
                                "metrics": hier_metrics}}

    else:
        target_labels_array = np.asarray(test_predicts['target_labels'])
        predcit_labels_array = np.asarray(test_predicts['predict_labels'])
        hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
        results = {"test": {"loss": test_loss,
                            "metric": test_metrics,
                            "predict_info": test_predicts},
                   "hier_test":{"target_labels": target_labels_array,
                                "pred_labels": predcit_labels_array,
                                "metrics": hier_metrics}}
    
    np.save(os.path.join(result_dir, "test_results.npy"), results)

    return test_metrics[config.experiment.monitored_metric]

def run_experiment_btsp(config, _run):
    """
    :param config: Object
    :param _run: the run object of this experiment
    """
    # Load the preprocessed data
    dp = DataPreprocess(config)
    data, label, indices = dp.load()
    
    # global mode
    if config.model.type in ['PreAttnMMs_GCN_MAP_V1']:
        n_classes = label['unique_label_number']
    elif config.model.type in ['PreAttnMMs_MTL_IMP2', "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        n_classes = get_parent_node_number(label)
    elif config.model.type == 'PreAttnMMs_HMCN':
        local_output_size = [2, 5, 4]
        globbal_output_size = label['unique_label_number'] - 1
        n_classes = (local_output_size, globbal_output_size)
    elif config.model.type == 'PreAttnMMs_MTL_LCL':
        n_classes = [2, 5, 7]
    # flat mode
    elif config.model.type == 'PreAttnMMs_FCLN':
        n_classes = label['unique_label_number'] - len(label['taxonomy'].keys())
    elif config.model.type == 'PreAttnMMs_FCAN':
        n_classes = label['unique_label_number'] - 1
    else:
        # local mode
        n_classes = len(label['taxonomy'][config.experiment.local_task])
        
        
    # build up model
    if config.model.type == 'PreAttnMMs_LCPN':
        model = PreAttnMMs(config, 
                          data['X_t_steps'], 
                          data['X_t_features'],
                          data['X_features'],
                          n_classes)
    elif config.model.type == 'PreAttnMMs_FCLN':
        model = PreAttnMMs_FCLN(config, 
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_FCAN':
        model = PreAttnMMs_FCAN(config, 
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_HMCN':
        model = PreAttnMMs_HMCN(config,
                                data['X_t_steps'], 
                                data['X_t_features'],
                                data['X_features'],
                                n_classes)
    elif config.model.type == 'PreAttnMMs_MTL_IMP2':
        model = PreAttnMMs_MTL_IMP2(config, 
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes)
    elif config.model.type == 'PreAttnMMs_MTL_LCL':
        model = PreAttnMMs_MTL_LCL(config, 
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes)
    elif config.model.type == 'PreAttnMMs_GCN_MAP_V1':
        adj = gen_A(label['unique_label_number'], label['taxonomy'])
        model = PreAttnMMs_GCN_MAP(config, 
                                   data['X_t_steps'], 
                                   data['X_t_features'],
                                   data['X_features'],
                                   n_classes,
                                   adj)
    elif config.model.type in ['PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
        adj = gen_A_parent_node(label['taxonomy'])
        model = PreAttnMMs_GAT_IMP8_GC(config, 
                                    data['X_t_steps'], 
                                    data['X_t_features'],
                                    data['X_features'],
                                    n_classes,
                                    adj)
    model.to(config.train.device_setting.device)
    
    # define training objective & optimizer & lr_scheduler
    if config.model.type in ['PreAttnMMs_GCN_MAP_V1', 'PreAttnMMs_FCAN']:
        """
        For multi-label prediction
        """
        criterion = nn.MultiLabelSoftMarginLoss()
    elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC"]:
        criterion = MultitaskLoss(n_classes)
    elif config.model.type in ["PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
        criterion = MultitaskWeightedLoss(n_classes)
    elif config.model.type == "PreAttnMMs_HMCN":
        criterion = nn.BCELoss()
    else:
        criterion = nn.NLLLoss()
    
    # define optimizer
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
    elif config.train.optimizer.type == "SGD":
        optimizer = torch.optim.SGD(
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
    
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='max', 
                                  factor=0.5, 
                                  patience=5, 
                                  verbose=True)

    # define trainer
    if config.experiment.type == 'flat':
        trainer = FlatTrainer(model,
                               criterion,
                               optimizer,
                               scheduler,
                               config,
                               data['X_t_steps'], 
                               data['X_t_features'],
                               n_classes)
    elif config.experiment.type == 'local':
        trainer = LocalTrainer(model,
                               criterion,
                               optimizer,
                               scheduler,
                               config,
                               data['X_t_steps'], 
                               data['X_t_features'],
                               n_classes)
    elif config.experiment.type == 'global':
        trainer = GlobalTrainer(model,
                                criterion,
                                optimizer,
                                scheduler,
                                config,
                                data['X_t_steps'], 
                                data['X_t_features'],
                                n_classes)
    
    # set checkpoint
    checkpoint_base = config.train.checkpoint.dir
    if config.experiment.type == "flat":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    elif config.experiment.type == "local":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'hp_tuning', 
                                    config.model.type,
                                    'node-{}'.format(config.experiment.local_task),
                                    config.data.norm_type,
                                    config.experiment.monitored_metric)
    elif config.experiment.type == "global":
        checkpoint_dir = os.path.join(checkpoint_base, 
                                    'no_hp_tuning', 
                                    config.model.type,
                                    config.data.norm_type,
                                    config.experiment.monitored_metric,
                                    config.experiment.head)
    
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

    # load the DataSet
    if config.experiment.type == "flat":
        """
        Generate data loader for flat experiment
        """
        collate_fn = FlatCollator(config, label)
        test_dataset = FlatDataset(config, data, label, indices, stage="TEST")

    elif config.experiment.type == "local":
        """
        Generate data loader for local experiment
        """
        collate_fn = LocalCollator(config, label)
        test_dataset = LocalHcDataset(config, data, label, indices, stage="TEST")

    elif config.experiment.type == "global":
        """
        Generate data loader for global experiment
        """
        collate_fn = GlobalCollator(config, label)
        test_dataset = GlobalHcDataset(config, data, label, indices, stage="TEST")
    
    else:
        logger.error("The loaded dataset is not flat, local dataset or global dataset!")
    
    # set results dir
    result_base = config.test.results.dir
    if config.experiment.type == "flat":
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)
    elif config.experiment.type == "local":
        result_dir = os.path.join(result_base, 
                                  'hp_tuning', 
                                  config.model.type,
                                  'node-{}'.format(config.experiment.local_task),
                                  config.data.norm_type,
                                  config.experiment.monitored_metric)

    elif config.experiment.type == "global":
        result_dir = os.path.join(result_base, 
                                  'no_hp_tuning', 
                                  config.model.type,
                                  config.data.norm_type,
                                  config.experiment.monitored_metric,
                                  config.experiment.head)
    if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        
    # Bootstrap Test
    results_btsp = {}
    n_bootstrap = 1000
    
    for runtime in range(n_bootstrap):
        logger.info("The {}-th iteration is processing !!!".format(runtime))
        # Create a random index list with replacement
        indices = np.random.choice(len(test_dataset), len(test_dataset), replace=True)
        
        # Create a SubsetRandomSampler from the random index list
        sampler = SubsetRandomSampler(indices)
        
        # Create a new DataLoader for the test dataset with the SubsetRandomSampler
        test_loader = DataLoader(test_dataset,
                        batch_size=config.train.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=config.train.device_setting.num_workers,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=True)
        
        # run test
        test_loss, test_metrics, test_predicts = trainer.eval(test_loader, config.train.start_epoch, 'TEST')
        
        # plot and save the results
        if config.model.type == "PreAttnMMs_LCPN":
            """
            code below is only for each local task, not for local hierarchical classification, 
            more info can be seen at 2_Test_LCPN.ipynb.
            """
            roc_params = roc(n_classes=n_classes,
                            target_labels=test_predicts['target_labels'],
                            predict_probs=np.asarray(test_predicts['predict_probas']))

            prc_params = prc(n_classes=n_classes,
                            target_labels=test_predicts['target_labels'],
                            predict_probs=np.asarray(test_predicts['predict_probas']),
                            predict_labels=test_predicts["predict_labels"])
            
            plot_roc(result_dir, roc_params, config.experiment.local_task)
            plot_prc(result_dir, prc_params, config.experiment.local_task)
            plot_cm(result_dir, test_predicts['target_labels'], test_predicts["predict_labels"], config.experiment.local_task)

            # save all results
            results = {"test": {"loss": test_loss,
                                "metric": test_metrics,
                                "predict_info": test_predicts},
                    "roc_params": roc_params,
                    "prc_params": prc_params}
            # caution: the hierarchical test results can be found at `/data/wzx/HC4FUOV2/results/hp_tuning/PreAttnMMs_LCPN/test_results.npy`
        
        elif config.model.type == "PreAttnMMs_FCLN":
            target_labels_array = np.zeros((len(test_predicts['target_labels']), 11))
            predcit_labels_array = np.zeros((len(test_predicts['target_labels']), 11))
            
            # reconstruct the target labels into binary form
            for i in range(target_labels_array.shape[0]):
                if test_predicts['target_labels'][i] == 0:
                    target_labels_array[i, 0] = 1
                    target_labels_array[i, 2] = 1
                elif test_predicts['target_labels'][i] == 1:
                    target_labels_array[i, 0] = 1
                    target_labels_array[i, 3] = 1
                elif test_predicts['target_labels'][i] == 2:
                    target_labels_array[i, 0] = 1
                    target_labels_array[i, 4] = 1
                elif test_predicts['target_labels'][i] == 3:
                    target_labels_array[i, 1] = 1
                    target_labels_array[i, 5] = 1
                    target_labels_array[i, 7] = 1
                elif test_predicts['target_labels'][i] == 4:
                    target_labels_array[i, 1] = 1
                    target_labels_array[i, 5] = 1
                    target_labels_array[i, 8] = 1
                elif test_predicts['target_labels'][i] == 5:
                    target_labels_array[i, 1] = 1
                    target_labels_array[i, 6] = 1
                    target_labels_array[i, 9] = 1
                elif test_predicts['target_labels'][i] == 6:
                    target_labels_array[i, 1] = 1
                    target_labels_array[i, 6] = 1
                    target_labels_array[i, 10] = 1
                else:
                    logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

            # reconstruct the predict labels into binary form
            for i in range(predcit_labels_array.shape[0]):
                if test_predicts['predict_labels'][i] == 0:
                    predcit_labels_array[i, 0] = 1
                    predcit_labels_array[i, 2] = 1
                elif test_predicts['predict_labels'][i] == 1:
                    predcit_labels_array[i, 0] = 1
                    predcit_labels_array[i, 3] = 1
                elif test_predicts['predict_labels'][i] == 2:
                    predcit_labels_array[i, 0] = 1
                    predcit_labels_array[i, 4] = 1
                elif test_predicts['predict_labels'][i] == 3:
                    predcit_labels_array[i, 1] = 1
                    predcit_labels_array[i, 5] = 1
                    predcit_labels_array[i, 7] = 1
                elif test_predicts['predict_labels'][i] == 4:
                    predcit_labels_array[i, 1] = 1
                    predcit_labels_array[i, 5] = 1
                    predcit_labels_array[i, 8] = 1
                elif test_predicts['predict_labels'][i] == 5:
                    predcit_labels_array[i, 1] = 1
                    predcit_labels_array[i, 6] = 1
                    predcit_labels_array[i, 9] = 1
                elif test_predicts['predict_labels'][i] == 6:
                    predcit_labels_array[i, 1] = 1
                    predcit_labels_array[i, 6] = 1
                    predcit_labels_array[i, 10] = 1
                else:
                    logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

            # cal evaluate4test
            hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
            results = {"test": {"loss": test_loss,
                                "metric": test_metrics,
                                "predict_info": test_predicts},
                    "hier_test":{"target_labels": target_labels_array,
                                 "pred_labels": predcit_labels_array,
                                 "metrics": hier_metrics}}
        
        elif config.model.type == "PreAttnMMs_GCN_MAP_V1":
            target_labels_array = np.asarray(test_predicts['target_labels'])[:,1:]
            predcit_labels_array = np.asarray(test_predicts['predict_labels'])[:,1:]
            hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
            results = {"test": {"loss": test_loss,
                                "metric": test_metrics,
                                "predict_info": test_predicts},
                    "hier_test":{"target_labels": target_labels_array,
                                    "pred_labels": predcit_labels_array,
                                    "metrics": hier_metrics}}
        
        elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
            
            target_labels_array = np.asarray(test_predicts['target_labels'])
            predcit_labels_array = np.asarray(test_predicts['predict_labels'])

            sample_size = target_labels_array.shape[1]
            label_pred = np.zeros((sample_size, 11))
            label_target = np.zeros((sample_size, 11))
            # transform label_target and label_pred into binarized label, i.e. [[1,0,1,0,0,0,0,0,0,0,0],...]
            for i in range(sample_size):
                for index, j in enumerate(target_labels_array[:, i]):
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

                for index, j in enumerate(predcit_labels_array[:, i]):
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

            hier_metrics = evaluate4test(config, label_target, label_pred)
            results = {"test": {"loss": test_loss,
                                "metric": test_metrics,
                                "predict_info": test_predicts},
                       "hier_test":{"target_labels": label_target,
                                    "pred_labels": label_pred,
                                    "metrics": hier_metrics}}
            
        elif config.model.type == "PreAttnMMs_MTL_LCL":
            target_labels_array = np.asarray(test_predicts['target_labels'])
            predcit_labels_array = np.asarray(test_predicts['predict_labels'])

            sample_size = target_labels_array.shape[1]
            label_pred = np.zeros((sample_size, 11))
            label_target = np.zeros((sample_size, 11))
            for i in range(sample_size):
                for index, j in enumerate(target_labels_array[:, i]):
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
                            label_target[i, 5] = 0
                            label_target[i, 6] = 0
                        elif j == 1:
                            label_target[i, 2] = 0
                            label_target[i, 3] = 1
                            label_target[i, 4] = 0
                            label_target[i, 5] = 0
                            label_target[i, 6] = 0
                        elif j == 2:
                            label_target[i, 2] = 0
                            label_target[i, 3] = 0
                            label_target[i, 4] = 1
                            label_target[i, 5] = 0
                            label_target[i, 6] = 0
                        elif j == 3:
                            label_target[i, 2] = 0
                            label_target[i, 3] = 0
                            label_target[i, 4] = 0
                            label_target[i, 5] = 1
                            label_target[i, 6] = 0
                        elif j == 4:
                            label_target[i, 2] = 0
                            label_target[i, 3] = 0
                            label_target[i, 4] = 0
                            label_target[i, 5] = 0
                            label_target[i, 6] = 1
                    elif index == 2:
                        if j == 0:
                            label_target[i, 7] = 0
                            label_target[i, 8] = 0
                            label_target[i, 9] = 0
                            label_target[i, 10] = 0
                        elif j == 1:
                            label_target[i, 7] = 0
                            label_target[i, 8] = 0
                            label_target[i, 9] = 0
                            label_target[i, 10] = 0
                        elif j == 2:
                            label_target[i, 7] = 0
                            label_target[i, 8] = 0
                            label_target[i, 9] = 0
                            label_target[i, 10] = 0
                        elif j == 3:
                            label_target[i, 7] = 1
                            label_target[i, 8] = 0
                            label_target[i, 9] = 0
                            label_target[i, 10] = 0
                        elif j == 4:
                            label_target[i, 7] = 0
                            label_target[i, 8] = 1
                            label_target[i, 9] = 0
                            label_target[i, 10] = 0
                        elif j == 5:
                            label_target[i, 7] = 0
                            label_target[i, 8] = 0
                            label_target[i, 9] = 1
                            label_target[i, 10] = 0
                        elif j == 6:
                            label_target[i, 7] = 0
                            label_target[i, 8] = 0
                            label_target[i, 9] = 0
                            label_target[i, 10] = 1

                for index, j in enumerate(predcit_labels_array[:, i]):
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
                            label_pred[i, 5] = 0
                            label_pred[i, 6] = 0
                        elif j == 1:
                            label_pred[i, 2] = 0
                            label_pred[i, 3] = 1
                            label_pred[i, 4] = 0
                            label_pred[i, 5] = 0
                            label_pred[i, 6] = 0
                        elif j == 2:
                            label_pred[i, 2] = 0
                            label_pred[i, 3] = 0
                            label_pred[i, 4] = 1
                            label_pred[i, 5] = 0
                            label_pred[i, 6] = 0
                        elif j == 3:
                            label_pred[i, 2] = 0
                            label_pred[i, 3] = 0
                            label_pred[i, 4] = 0
                            label_pred[i, 5] = 1
                            label_pred[i, 6] = 0
                        elif j == 4:
                            label_pred[i, 2] = 0
                            label_pred[i, 3] = 0
                            label_pred[i, 4] = 0
                            label_pred[i, 5] = 0
                            label_pred[i, 6] = 1
                    elif index == 2:
                        if j == 0:
                            label_pred[i, 7] = 0
                            label_pred[i, 8] = 0
                            label_pred[i, 9] = 0
                            label_pred[i, 10] = 0
                        elif j == 1:
                            label_pred[i, 7] = 0
                            label_pred[i, 8] = 0
                            label_pred[i, 9] = 0
                            label_pred[i, 10] = 0
                        elif j == 2:
                            label_pred[i, 7] = 0
                            label_pred[i, 8] = 0
                            label_pred[i, 9] = 0
                            label_pred[i, 10] = 0
                        elif j == 3:
                            label_pred[i, 7] = 1
                            label_pred[i, 8] = 0
                            label_pred[i, 9] = 0
                            label_pred[i, 10] = 0
                        elif j == 4:
                            label_pred[i, 7] = 0
                            label_pred[i, 8] = 1
                            label_pred[i, 9] = 0
                            label_pred[i, 10] = 0
                        elif j == 5:
                            label_pred[i, 7] = 0
                            label_pred[i, 8] = 0
                            label_pred[i, 9] = 1
                            label_pred[i, 10] = 0
                        elif j == 6:
                            label_pred[i, 7] = 0
                            label_pred[i, 8] = 0
                            label_pred[i, 9] = 0
                            label_pred[i, 10] = 1
            
            hier_metrics = evaluate4test(config, label_target, label_pred)
            results = {"test": {"loss": test_loss,
                                "metric": test_metrics,
                                "predict_info": test_predicts},
                    "hier_test":{"target_labels": label_target,
                                 "pred_labels": label_pred,
                                 "metrics": hier_metrics}}

        else:
            target_labels_array = np.asarray(test_predicts['target_labels'])
            predcit_labels_array = np.asarray(test_predicts['predict_labels'])
            hier_metrics = evaluate4test(config, target_labels_array, predcit_labels_array)
            results = {"test": {"loss": test_loss,
                                "metric": test_metrics,
                                "predict_info": test_predicts},
                    "hier_test":{"target_labels": target_labels_array,
                                    "pred_labels": predcit_labels_array,
                                    "metrics": hier_metrics}}
        
        results_btsp[runtime] = results
    
    np.save(os.path.join(result_dir, "btsp_results.npy"), results_btsp)
    print("The process finished!")
    
    return test_metrics[config.experiment.monitored_metric]

        