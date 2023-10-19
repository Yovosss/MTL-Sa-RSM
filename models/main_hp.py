#!/usr/bin/env python
# coding:utf-8

import os
import pickle
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import RandomSampler
from optuna.trial import TrialState
from optuna.visualization import (plot_contour, plot_intermediate_values,
                                  plot_optimization_history,
                                  plot_parallel_coordinate,
                                  plot_param_importances, plot_slice)
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import helper.logger as logger
from helper.data import DataPreprocess
from helper.data_loader import data_loaders
from helper.utils import (build_edge_index, gen_A, gen_A_parent_node,
                          get_parent_node_number, load_checkpoint,
                          save_checkpoint)
from models.loss import (MultitaskLoss, MultitaskWeightedLoss)
from models.model import (PreAttnMMs_FCAN_Hp, PreAttnMMs_FCLN_Hp,
                          PreAttnMMs_GAT_IMP8_GC_Hp, PreAttnMMs_GCN_MAP_Hp,
                          PreAttnMMs_HMCN_Hp, PreAttnMMs_MTL_IMP2_Hp, PreAttnMMs_MTL_LCL_Hp,
                          PreAttnMMsHp)
from train_modules.early_stopping import EarlyStopping
from train_modules.trainer import (FlatTrainerHp, GlobalTrainerHp,
                                   LocalTrainerHp)


class Objective():
    def __init__(self, config, data, label, indices):
        """
        Reference: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
        """
        self.config = config
        self.data = data
        self.label = label
        self.indices = indices

        self._best_model = None
        self._best_optimizer = None
        self._best_trial_log = None

    def __call__(self, trial):
        
        # global mode
        if self.config.model.type == 'PreAttnMMs_GCN_MAP_V1':
            n_classes = self.label['unique_label_number']
        elif self.config.model.type in ['PreAttnMMs_MTL_IMP2', 'PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
            n_classes = get_parent_node_number(self.label)
        elif self.config.model.type == 'PreAttnMMs_HMCN':
            local_output_size = [2, 5, 4]
            globbal_output_size = self.label['unique_label_number'] - 1
            n_classes = (local_output_size, globbal_output_size)
        elif self.config.model.type == 'PreAttnMMs_MTL_LCL':
            n_classes = [2, 5, 7]
        # flat mode
        elif self.config.model.type == 'PreAttnMMs_FCLN':
            n_classes = self.label['unique_label_number'] - len(self.label['taxonomy'].keys())
        elif self.config.model.type == 'PreAttnMMs_FCAN':
            n_classes = self.label['unique_label_number'] - 1
        else:
            # local mode
            n_classes = len(self.label['taxonomy'][self.config.experiment.local_task])

        # build up model
        if self.config.model.type == 'PreAttnMMs_LCPN':
            model = PreAttnMMsHp(trial,
                                self.config, 
                                self.data['X_t_steps'], 
                                self.data['X_t_features'],
                                self.data['X_features'],
                                n_classes)
        elif self.config.model.type == 'PreAttnMMs_FCLN':
            model = PreAttnMMs_FCLN_Hp(trial,
                                       self.config, 
                                       self.data['X_t_steps'], 
                                       self.data['X_t_features'],
                                       self.data['X_features'],
                                       n_classes)
        elif self.config.model.type == 'PreAttnMMs_FCAN':
            model = PreAttnMMs_FCAN_Hp(trial,
                                       self.config, 
                                       self.data['X_t_steps'], 
                                       self.data['X_t_features'],
                                       self.data['X_features'],
                                       n_classes)
        elif self.config.model.type == 'PreAttnMMs_HMCN':
            model = PreAttnMMs_HMCN_Hp(trial,
                                       self.config, 
                                       self.data['X_t_steps'], 
                                       self.data['X_t_features'],
                                       self.data['X_features'],
                                       n_classes)
        elif self.config.model.type == 'PreAttnMMs_MTL_LCL':
            model = PreAttnMMs_MTL_LCL_Hp(trial,
                                       self.config, 
                                       self.data['X_t_steps'], 
                                       self.data['X_t_features'],
                                       self.data['X_features'],
                                       n_classes)
        elif self.config.model.type == 'PreAttnMMs_MTL_IMP2':
            model = PreAttnMMs_MTL_IMP2_Hp(trial,
                                       self.config, 
                                       self.data['X_t_steps'], 
                                       self.data['X_t_features'],
                                       self.data['X_features'],
                                       n_classes)
        elif self.config.model.type == 'PreAttnMMs_GCN_MAP_V1':
            adj = gen_A(self.label['unique_label_number'], self.label['taxonomy'])
            model = PreAttnMMs_GCN_MAP_Hp(trial,
                                          self.config, 
                                          self.data['X_t_steps'], 
                                          self.data['X_t_features'],
                                          self.data['X_features'],
                                          n_classes,
                                          adj)
        elif self.config.model.type in ['PreAttnMMs_GAT_IMP8_GC', 'PreAttnMMs_GAT_IMP8_GC_WeightedLoss']:
            adj = gen_A_parent_node(self.label['taxonomy'])
            model = PreAttnMMs_GAT_IMP8_GC_Hp(trial,
                                      self.config, 
                                      self.data['X_t_steps'], 
                                      self.data['X_t_features'],
                                      self.data['X_features'],
                                      n_classes,
                                      adj)
        model.to(self.config.train.device_setting.device)

        # define DataLoader
        train_loader, validation_loader, test_loader = data_loaders(self.config, self.data, self.label, self.indices)

        # define training objective & optimizer & lr_scheduler
        if self.config.model.type in ['PreAttnMMs_GCN_MAP_V1', 'PreAttnMMs_FCAN']:
            """
            For multi-label prediction
            """
            criterion = nn.MultiLabelSoftMarginLoss()
        elif self.config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC"]:
            criterion = MultitaskLoss(n_classes)
        elif self.config.model.type in ["PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
            criterion = MultitaskWeightedLoss(n_classes)
        elif self.config.model.type == "PreAttnMMs_HMCN":
            criterion = nn.BCELoss()
        else:
            criterion = nn.NLLLoss()

        # define the optimizers
        kwargs = {}
        optimizer_selected = trial.suggest_categorical("optimizer", ["RMSprop", "Adam", "AdamW", "SGD"])
        if optimizer_selected == "Adam":
            kwargs["lr"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-2, log=True)
            kwargs["weight_decay"] = trial.suggest_float("adam_weight_decay", 1e-12, 1e-1, log=True)
        elif optimizer_selected == "RMSprop":
            kwargs["lr"] = trial.suggest_float("rmsprop_learning_rate", 1e-5, 1e-2, log=True)
            kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 1e-12, 1e-1, log=True)
        elif optimizer_selected == "AdamW":
            kwargs["lr"] = trial.suggest_float("AdamW_learning_rate", 1e-5, 1e-2 , log=True)
            kwargs["weight_decay"] = trial.suggest_float("AdamW_weight_decay", 1e-12, 1e-1, log=True)
        elif optimizer_selected == "SGD":
            kwargs["lr"] = trial.suggest_float("SGD_learning_rate", 1e-5, 1e-2 , log=True)
            kwargs["weight_decay"] = trial.suggest_float("SGD_weight_decay", 1e-12, 1e-1, log=True)

        optimizer = getattr(optim, optimizer_selected)(model.parameters(), **kwargs)

        # define the lr scheduler
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='max', 
                                      factor=0.5, 
                                      patience=5, 
                                      verbose=True)
        # define trainer
        if self.config.experiment.type == 'flat':
            trainer = FlatTrainerHp(trial,
                                    model,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    self.config,
                                    self.data['X_t_steps'], 
                                    self.data['X_t_features'],
                                    n_classes)
        elif self.config.experiment.type == 'local':
            trainer = LocalTrainerHp(trial,
                                    model,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    self.config,
                                    self.data['X_t_steps'], 
                                    self.data['X_t_features'],
                                    n_classes)
        elif self.config.experiment.type == 'global':
            trainer = GlobalTrainerHp(trial,
                                    model,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    self.config,
                                    self.data['X_t_steps'], 
                                    self.data['X_t_features'],
                                    n_classes)

        # set checkpoint
        self.checkpoint_base = self.config.train.checkpoint.dir
        if self.config.experiment.type == "flat":
            self.checkpoint_dir = os.path.join(self.checkpoint_base, 
                                               'hp_tuning', 
                                               self.config.model.type,
                                               self.config.data.norm_type,
                                               self.config.experiment.monitored_metric)
        if self.config.experiment.type == "local":
            self.checkpoint_dir = os.path.join(self.checkpoint_base, 
                                               'hp_tuning', 
                                               self.config.model.type,
                                               'node-{}'.format(self.config.experiment.local_task),
                                               self.config.data.norm_type,
                                               self.config.experiment.monitored_metric)
        elif self.config.experiment.type == "global":
            self.checkpoint_dir = os.path.join(self.checkpoint_base, 
                                               'hp_tuning', 
                                               self.config.model.type,
                                               self.config.data.norm_type,
                                               self.config.experiment.monitored_metric)
        
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.config.train.early_stopping.patience, 
                                       path=os.path.join(self.checkpoint_dir, 'best_checkpoint(trial{}).pt'.format(trial.number)),
                                       verbose=True)
        # train
        self.best_epoch = -1
        self.best_performance = 0.0
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

        for epoch in range(self.config.train.start_epoch, self.config.train.end_epoch):
            start_time = time.time()
            train_loss, train_metrics, train_predicts = trainer.train(train_loader, epoch)
            valid_loss, valid_metrics, valid_predicts = trainer.eval(validation_loader, epoch, "VALIDATION")

            monitored_metric = self.config.experiment.monitored_metric

            if valid_metrics[monitored_metric] > self.best_performance:

                self.best_epoch = epoch
                self.best_performance = valid_metrics[monitored_metric]

            log["train"]["loss"].append(train_loss)
            log["train"]["metric"].append({"epoch {0}".format(epoch): train_metrics})
            log["train"]["predict_info"].append({"epoch {0}".format(epoch): train_predicts})
            log["valid"]["loss"].append(valid_loss)
            log["valid"]["metric"].append({"epoch {0}".format(epoch): valid_metrics})
            log["valid"]["predict_info"].append({"epoch {0}".format(epoch): valid_predicts})
            log["best_epoch"] = self.best_epoch
            log["best_performance"] = self.best_performance

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_metrics[monitored_metric], model, optimizer, self.best_epoch, self.best_performance, monitored_metric)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            logger.info('Epoch {}: --- Time Cost {} secs. \n'.format(epoch, time.time() - start_time))
        
        # load the last checkpoint with the best model
        model.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, 
                                    'best_checkpoint(trial{}).pt'.format(trial.number)))['model'])
        optimizer.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, 
                                    'best_checkpoint(trial{}).pt'.format(trial.number)))['optimizer'])

        self._best_model = model
        self._best_optimizer = optimizer
        self._best_trial_log = log

        return self.best_performance

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            torch.save({
                'epoch': self.best_epoch,
                'best_performance': self.best_performance,
                'model': self._best_model.state_dict(),
                'optimizer': self._best_optimizer.state_dict()},
                os.path.join(self.checkpoint_dir, 'best_best_checkpoint(i.e.trial{}).pt'.format(trial.number)))
            
            with open(os.path.join(self.checkpoint_dir, 'best_best_trial_log(i.e.trial{}).pkl'.format(trial.number)), 'wb') as f:
                pickle.dump(self._best_trial_log, f)

def run_experiment_hp(config, _run):
    """
    :param config: Object
    :param _run: the run object of this experiment
    """
    T = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # Load the preprocessed data
    dp = DataPreprocess(config)
    data, label, indices = dp.load()

    objective = Objective(config, data, label, indices)
    storage = optuna.storages.RDBStorage(
        url='postgresql://postgres:postgres@xx.xx.xx.xx:xxxx/postgres',
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
            }
    )
    if config.model.type == "PreAttnMMs_LCPN":
        study = optuna.create_study(direction='maximize', 
                                    storage=storage,
                                    sampler= optuna.samplers.TPESampler(),
                                    study_name='{0}({1})_{2}_{3}'.format(config.model.type, 'node-{}'.format(config.experiment.local_task), config.experiment.monitored_metric, T))
    else:  
        study = optuna.create_study(direction='maximize', 
                                    storage=storage,
                                    sampler= optuna.samplers.TPESampler(),
                                    study_name='{0}_{1}_{2}'.format(config.model.type, config.experiment.monitored_metric, T))    

    study.optimize(objective, 
                   n_trials=100,
                   callbacks=[objective.callback])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: {}".format(len(study.trials)))
    logger.info("  Number of pruned trials: {}".format(len(pruned_trials)))
    logger.info("  Number of complete trials: {}".format(len(complete_trials)))

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

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

    # save the visualization plot
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(checkpoint_dir, "optimization_history.png"))

    fig = plot_param_importances(study)
    fig.write_image(os.path.join(checkpoint_dir, "param_importances.png"))

    fig = plot_slice(study)
    fig.write_image(os.path.join(checkpoint_dir, "plot_slice.png"))

    fig = plot_intermediate_values(study)
    fig.write_image(os.path.join(checkpoint_dir, "plot_intermediate_values.png"))

    # Get the optimal hyperparameters and save
    best_hps_dict = {'best_hps': trial.params}
    np.save(os.path.join(checkpoint_dir, 'BestHPs.npy'), best_hps_dict)

    logger.info("The hyperparameter process finished!")

    return objective.best_performance
