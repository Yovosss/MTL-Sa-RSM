#!/usr/bin/env python
# coding:utf-8

import numpy as np
import optuna
import torch
import torch.nn.functional as F
import torchnet as tnt
import tqdm

import helper.logger as logger
from helper.utils import AveragePrecisionMeter, t_SNE, visualization
from train_modules.evaluation_metrics import evaluate
from train_modules.logger_information import logger_print


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_classes = n_classes
    
    def run(self, data_loader, epoch, stage, mode="TRAIN"):
        pass

    def _check_input(self, batch, expected_n_steps, expected_n_features, out_dtype="tensor"):
        """Check value type and shape of batch input

        Parameters
        ----------
        expected_n_steps : int
            Number of time steps of input time series (X) that the model expects.
            This value is the same with the argument `n_steps` used to initialize the model.

        expected_n_features : int
            Number of feature dimensions of input time series (X) that the model expects.
            This value is the same with the argument `n_features` used to initialize the model.

        batch : Dict-like,
                Dict of input data, including 'X_t', 'X_t_mask', 'deltaT_t', 'X_t_filledLOCF', 'empirical_mean', 'y'.
            
        out_dtype : str, default='tensor'
            Data type of the output, should be torch.Tensor

        Returns
        -------
        batch : tensor
        """
        assert out_dtype == 'tensor', f'out_dtype should be "tensor", but got {out_dtype}'

        is_dict = isinstance(batch, dict)
        assert is_dict, TypeError(
            "batch should be an instance of Dict, but got {}".format(type(batch))
            )

        for key, value in batch.items():
            # check the type
            is_tensor = isinstance(value, torch.Tensor)
            assert is_tensor, TypeError(
                "{0} should be an instance of torch.Tensor, but got {1}".format(key, type(value))
            )

            # check the shape of value
            shape = value.size()

            if key in ['X_t', 'X_t_mask', 'deltaT_t', 'X_t_filledLOCF']:
                
                assert len(shape) == 3, (
                    f"{key} should have 3 dimensions [batch_size, seq_len, n_features],"
                    f"but got shape={shape}"
                )

                assert shape[1] == expected_n_steps, (
                    f"expect the second dimension of {key} to be {expected_n_steps}, but got {shape[1]}"
                )

                assert shape[2] == expected_n_features, (
                    f"expect the third dimension of {key} to be {expected_n_features}, but got {shape[2]}"
                )
            elif key == 'empirical_mean':
                assert shape[1] == expected_n_features, (
                    f"expect the second dimension of {key} to be {expected_n_features}, but got {shape[1]}"
                )

            # convert the data type if in need
            if out_dtype == "tensor":
                batch[key] = value.to(self.config.train.device_setting.device)

        return batch

    def update_lr(self):
        """
        Callback Function
        update learning rate according to the decay weight
        (!!!decrapted)
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'TRAIN', mode="TRAIN")
    
    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode="EVAL")

class FlatTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        """
        Trainer class for flat classification
        """
        super().__init__(model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes)

    def run(self, data_loader, epoch, stage, mode="TRAIN"):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        metrics = {}
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": []}
        train_losses = []
        valid_losses = []
        
        if mode == "TRAIN":
            for idx, batch in enumerate(data_loader):
                inputs = self._check_input(batch, self.n_steps, self.n_features)
                logits = self.model(inputs)

                if self.config.model.type == "PreAttnMMs_FCAN":
                    predictions = torch.sigmoid(logits)
                    loss = self.criterion(logits, inputs['label'])

                    predict_info['predict_probas'].extend(predictions.cpu().tolist())
                    predict_info["predict_labels"].extend((predictions >= .5).int().data.cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                elif self.config.model.type == "PreAttnMMs_FCLN":
                    predictions = F.softmax(logits, dim=1)
                    loss = self.criterion(F.log_softmax(logits, dim=1),
                                        inputs['label'])

                    predict_info['predict_probas'].extend(predictions.cpu().tolist())
                    predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
                                    
                train_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = np.mean(train_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)
            # print the above information about model training
            logger_print(self.config, stage, epoch, train_loss, metrics)

            return train_loss, metrics, predict_info

        elif mode == "EVAL":
            with torch.no_grad():
                for idx, batch in enumerate(data_loader):
                    inputs = self._check_input(batch, self.n_steps, self.n_features)
                    logits = self.model(inputs)

                    if self.config.model.type == "PreAttnMMs_FCAN":
                        predictions = torch.sigmoid(logits)
                        loss = self.criterion(logits, inputs['label'])

                        predict_info['predict_probas'].extend(predictions.cpu().tolist())
                        predict_info["predict_labels"].extend((predictions >= .5).int().data.cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                    elif self.config.model.type == "PreAttnMMs_FCLN":
                        predictions = F.softmax(logits, dim=1)
                        loss = self.criterion(F.log_softmax(logits, dim=1),
                                            inputs['label'])

                        predict_info['predict_probas'].extend(predictions.cpu().tolist())
                        predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                    valid_losses.append(loss.item())

            valid_loss = np.mean(valid_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)

            logger_print(self.config, stage, epoch, valid_loss, metrics)
            if stage == 'VALIDATION':
                # update learning rate
                logger.info("epoch {0}: --- learning rate = {1}".format(epoch, self.optimizer.param_groups[0]['lr']))

                self.scheduler.step(metrics[self.config.experiment.monitored_metric])

            return valid_loss, metrics, predict_info

class LocalTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        """
        Trainer class for local classification
        """
        super().__init__(model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes)

    def run(self, data_loader, epoch, stage, mode="TRAIN"):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        metrics = {}
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": []}
        train_losses = []
        valid_losses = []
        
        if mode == "TRAIN":
            for idx, batch in enumerate(data_loader):
                inputs = self._check_input(batch, self.n_steps, self.n_features)
                logits = self.model(inputs)
                predictions = F.softmax(logits, dim=1)
                loss = self.criterion(F.log_softmax(logits, dim=1),
                                    inputs['label'])
                                    
                train_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predict_info['predict_probas'].extend(predictions.cpu().tolist())
                predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

            train_loss = np.mean(train_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)

            # print the above information about model training
            logger_print(self.config, stage, epoch, train_loss, metrics)

            return train_loss, metrics, predict_info

        elif mode == "EVAL":
            with torch.no_grad():
                for idx, batch in enumerate(data_loader):
                    inputs = self._check_input(batch, self.n_steps, self.n_features)
                    logits = self.model(inputs)
                    predictions = F.softmax(logits, dim=1)
                    loss = self.criterion(F.log_softmax(logits, dim=1),
                                    inputs['label'])
                    valid_losses.append(loss.item())

                    predict_info['predict_probas'].extend(predictions.cpu().tolist())
                    predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
            
            valid_loss = np.mean(valid_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)

            logger_print(self.config, stage, epoch, valid_loss, metrics)

            if stage == 'VALIDATION':
                # update learning rate
                logger.info("epoch {0}: --- learning rate = {1}".format(epoch, self.optimizer.param_groups[0]['lr']))

                self.scheduler.step(metrics[self.config.experiment.monitored_metric])

            return valid_loss, metrics, predict_info

class GlobalTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        """
        Trainer class for global classification
        """
        super().__init__(model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes)

    def run(self, data_loader, epoch, stage, mode="TRAIN"):

        metrics = {}
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": [],
                        "graph_embeddings": []}
        train_losses = []
        valid_losses = []

        if mode == "TRAIN":
            for idx, batch in enumerate(data_loader):
                inputs = self._check_input(batch, self.n_steps, self.n_features)

                if self.config.model.type == "PreAttnMMs_HMCN":
                    logits = self.model(inputs)
                    loss = self.criterion(logits, inputs['label'].float())

                    predict_info['predict_probas'].extend(logits.cpu().tolist())
                    predict_info["predict_labels"].extend((logits >= .5).int().data.cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                elif self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                    logits, graph_embedding = self.model(inputs)
                    loss = self.criterion(logits, inputs['label'])

                    predict_info['predict_probas'].extend(logits.cpu().tolist())
                    predict_info["predict_labels"].extend((logits >= 0).int().data.cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
                    predict_info['graph_embeddings'].extend(graph_embedding.cpu().tolist())
                
                elif self.config.model.type in ["PreAttnMMs_MTL_LCL", "PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, inputs['label'])

                    for i in range(len(self.n_classes)):
                        prediction = F.softmax(logits[i], dim=1)
                        if idx == 0:
                            predict_info['predict_probas'].append([])
                            predict_info["predict_labels"].append([])
                            predict_info['target_labels'].append([])
                        predict_info['predict_probas'][i].extend(prediction.cpu().tolist())
                        predict_info["predict_labels"][i].extend(prediction.max(1)[-1].cpu().tolist())
                        predict_info["target_labels"][i].extend(inputs['label'][i].cpu().tolist())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            if self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                metrics = evaluate(self.config,
                                metrics,
                                np.asarray(predict_info['predict_probas'])[:,1:],
                                np.asarray(predict_info["predict_labels"])[:,1:],
                                np.asarray(predict_info['target_labels'])[:,1:],
                                self.n_classes)
            else:
                metrics = evaluate(self.config,
                                   metrics,
                                   predict_info['predict_probas'],
                                   predict_info["predict_labels"],
                                   predict_info['target_labels'],
                                   self.n_classes)

            # print the above information about model training
            logger_print(self.config, stage, epoch, train_loss, metrics)

            return train_loss, metrics, predict_info

        elif mode == "EVAL":
            with torch.no_grad():
                for idx, batch in enumerate(data_loader):
                    inputs = self._check_input(batch, self.n_steps, self.n_features)

                    if self.config.model.type == "PreAttnMMs_HMCN":
                        logits = self.model(inputs)
                        loss = self.criterion(logits, inputs['label'].float())

                        predict_info['predict_probas'].extend(logits.cpu().tolist())
                        predict_info["predict_labels"].extend((logits >= .5).int().data.cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                    elif self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                        logits, graph_embedding = self.model(inputs)
                        loss = self.criterion(logits, inputs['label'])

                        predict_info['predict_probas'].extend(logits.cpu().tolist())
                        predict_info["predict_labels"].extend((logits >= 0).int().data.cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
                        predict_info['graph_embeddings'].extend(graph_embedding.cpu().tolist())

                    elif self.config.model.type in ["PreAttnMMs_MTL_LCL", "PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
                        logits = self.model(inputs)
                        loss = self.criterion(logits, inputs['label'])

                        for i in range(len(self.n_classes)):
                            prediction = F.softmax(logits[i], dim=1)
                            if idx == 0:
                                predict_info['predict_probas'].append([])
                                predict_info["predict_labels"].append([])
                                predict_info['target_labels'].append([])
                            predict_info['predict_probas'][i].extend(prediction.cpu().tolist())
                            predict_info["predict_labels"][i].extend(prediction.max(1)[-1].cpu().tolist())
                            predict_info["target_labels"][i].extend(inputs['label'][i].cpu().tolist())
                    
                    valid_losses.append(loss.item())

                valid_loss = np.mean(valid_losses)
                if self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                    metrics = evaluate(self.config,
                                    metrics,
                                    np.asarray(predict_info['predict_probas'])[:,1:],
                                    np.asarray(predict_info["predict_labels"])[:,1:],
                                    np.asarray(predict_info['target_labels'])[:,1:],
                                    self.n_classes)
                else:
                    metrics = evaluate(self.config,
                                    metrics,
                                    predict_info['predict_probas'],
                                    predict_info["predict_labels"],
                                    predict_info['target_labels'],
                                    self.n_classes)

                logger_print(self.config, stage, epoch, valid_loss, metrics)

                if stage == 'VALIDATION':
                    # update learning rate
                    logger.info("epoch {0}: --- learning rate = {1}".format(epoch, self.optimizer.param_groups[0]['lr']))

                    self.scheduler.step(metrics[self.config.experiment.monitored_metric])

            return valid_loss, metrics, predict_info

class FlatTrainerHp(Trainer):
    def __init__(self, trial, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        """
        Trainer class for flat classification
        """
        super().__init__(model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes)
        self.trial = trial

    def run(self, data_loader, epoch, stage, mode="TRAIN"):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        metrics = {}
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": []}
        train_losses = []
        valid_losses = []

        if mode == "TRAIN":
            for idx, batch in enumerate(data_loader):
                inputs = self._check_input(batch, self.n_steps, self.n_features)
                logits = self.model(inputs)

                if self.config.model.type == "PreAttnMMs_FCAN":
                    predictions = torch.sigmoid(logits)
                    loss = self.criterion(logits, inputs['label'])

                    predict_info['predict_probas'].extend(predictions.cpu().tolist())
                    predict_info["predict_labels"].extend((predictions >= .5).int().data.cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                elif self.config.model.type == "PreAttnMMs_FCLN":
                    predictions = F.softmax(logits, dim=1)
                    loss = self.criterion(F.log_softmax(logits, dim=1),
                                        inputs['label'])

                    predict_info['predict_probas'].extend(predictions.cpu().tolist())
                    predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
                                    
                train_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = np.mean(train_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)
            # print the above information about model training
            logger_print(self.config, stage, epoch, train_loss, metrics)

            return train_loss, metrics, predict_info
        
        elif mode == "EVAL":
            with torch.no_grad():
                for idx, batch in enumerate(data_loader):
                    inputs = self._check_input(batch, self.n_steps, self.n_features)
                    logits = self.model(inputs)

                    if self.config.model.type == "PreAttnMMs_FCAN":
                        predictions = torch.sigmoid(logits)
                        loss = self.criterion(logits, inputs['label'])

                        predict_info['predict_probas'].extend(predictions.cpu().tolist())
                        predict_info["predict_labels"].extend((predictions >= .5).int().data.cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                    elif self.config.model.type == "PreAttnMMs_FCLN":
                        predictions = F.softmax(logits, dim=1)
                        loss = self.criterion(F.log_softmax(logits, dim=1),
                                            inputs['label'])

                        predict_info['predict_probas'].extend(predictions.cpu().tolist())
                        predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                    valid_losses.append(loss.item())

            valid_loss = np.mean(valid_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)
            
            logger_print(self.config, stage, epoch, valid_loss, metrics)
            if stage == 'VALIDATION':
                # update learning rate
                logger.info("epoch {0}: --- learning rate = {1}".format(epoch, self.optimizer.param_groups[0]['lr']))

                self.scheduler.step(metrics[self.config.experiment.monitored_metric])

            self.trial.report(metrics[self.config.experiment.monitored_metric], epoch)

            # Handle pruning based on the intermediate value.
            # if self.trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

            return valid_loss, metrics, predict_info

class LocalTrainerHp(Trainer):
    def __init__(self, trial, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        """
        Hyperparameter tuning trainer class for local classification
        """
        super().__init__(model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes)
        self.trial = trial

    def run(self, data_loader, epoch, stage, mode="TRAIN"):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        metrics = {}
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": []}
        train_losses = []
        valid_losses = []

        if mode == "TRAIN":
            for idx, batch in enumerate(data_loader):
                inputs = self._check_input(batch, self.n_steps, self.n_features)
                logits = self.model(inputs)
                predictions = F.softmax(logits, dim=1)
                loss = self.criterion(F.log_softmax(logits, dim=1),
                                    inputs['label'])
                train_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predict_info['predict_probas'].extend(predictions.cpu().tolist())
                predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

            train_loss = np.mean(train_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)
            # print the above information about model training
            logger_print(self.config, stage, epoch, train_loss, metrics)

            return train_loss, metrics, predict_info

        # validation of the model
        elif mode == "EVAL":
            with torch.no_grad():
                for idx, batch in enumerate(data_loader):
                    inputs = self._check_input(batch, self.n_steps, self.n_features)
                    logits = self.model(inputs)
                    predictions = F.softmax(logits, dim=1)
                    loss = self.criterion(F.log_softmax(logits, dim=1),
                                    inputs['label'])
                    valid_losses.append(loss.item())

                    predict_info['predict_probas'].extend(predictions.cpu().tolist())
                    predict_info["predict_labels"].extend(predictions.max(1)[-1].cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
            
            valid_loss = np.mean(valid_losses)
            metrics = evaluate(self.config,
                               metrics,
                               predict_info['predict_probas'],
                               predict_info["predict_labels"],
                               predict_info['target_labels'],
                               self.n_classes)

            logger_print(self.config, stage, epoch, valid_loss, metrics)
                        
            if stage == 'VALIDATION':
                # update learning rate
                logger.info("epoch {0}: --- learning rate = {1}".format(epoch, self.optimizer.param_groups[0]['lr']))

                self.scheduler.step(metrics[self.config.experiment.monitored_metric])
            self.trial.report(metrics[self.config.experiment.monitored_metric], epoch)

            # Handle pruning based on the intermediate value.
            # if self.trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

            return valid_loss, metrics, predict_info

class GlobalTrainerHp(Trainer):
    def __init__(self, trial, model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes):
        """
        Hyperparameter tuning trainer class for global classification
        """
        super().__init__(model, criterion, optimizer, scheduler, config, n_steps, n_features, n_classes)
        self.trial = trial

    def run(self, data_loader, epoch, stage, mode="TRAIN"):

        metrics = {}
        predict_info = {"predict_probas": [], 
                        "predict_labels": [], 
                        "target_labels": [],
                        "graph_embeddings": []}
        train_losses = []
        valid_losses = []

        if mode == "TRAIN":
            for idx, batch in enumerate(data_loader):
                inputs = self._check_input(batch, self.n_steps, self.n_features)

                if self.config.model.type == "PreAttnMMs_HMCN":
                    logits = self.model(inputs)
                    loss = self.criterion(logits, inputs['label'].float())
                
                    predict_info['predict_probas'].extend(logits.cpu().tolist())
                    predict_info["predict_labels"].extend((logits >= .5).int().data.cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                elif self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                    logits, graph_embedding = self.model(inputs)
                    loss = self.criterion(logits, inputs['label'])

                    predict_info['predict_probas'].extend(logits.cpu().tolist())
                    predict_info["predict_labels"].extend((logits >= 0).int().data.cpu().tolist())
                    predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
                    predict_info['graph_embeddings'].extend(graph_embedding.cpu().tolist())

                elif self.config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, inputs['label'])

                    for i in range(len(self.n_classes)):
                        prediction = F.softmax(logits[i], dim=1)
                        if idx == 0:
                            predict_info['predict_probas'].append([])
                            predict_info["predict_labels"].append([])
                            predict_info['target_labels'].append([])
                        predict_info['predict_probas'][i].extend(prediction.cpu().tolist())
                        predict_info["predict_labels"][i].extend(prediction.max(1)[-1].cpu().tolist())
                        predict_info["target_labels"][i].extend(inputs['label'][i].cpu().tolist())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            if self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                metrics = evaluate(self.config,
                                metrics,
                                np.asarray(predict_info['predict_probas'])[:,1:],
                                np.asarray(predict_info["predict_labels"])[:,1:],
                                np.asarray(predict_info['target_labels'])[:,1:],
                                self.n_classes)
            else:
                metrics = evaluate(self.config,
                                   metrics,
                                   predict_info['predict_probas'],
                                   predict_info["predict_labels"],
                                   predict_info['target_labels'],
                                   self.n_classes)

            # print the above information about model training
            logger_print(self.config, stage, epoch, train_loss, metrics)

            return train_loss, metrics, predict_info

        elif mode == "EVAL":
            with torch.no_grad():
                for idx, batch in enumerate(data_loader):
                    inputs = self._check_input(batch, self.n_steps, self.n_features)

                    if self.config.model.type == "PreAttnMMs_HMCN":
                        logits = self.model(inputs)
                        loss = self.criterion(logits, inputs['label'].float())

                        predict_info['predict_probas'].extend(logits.cpu().tolist())
                        predict_info["predict_labels"].extend((logits >= .5).int().data.cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())

                    elif self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                        logits, graph_embedding = self.model(inputs)
                        loss = self.criterion(logits, inputs['label'])

                        predict_info['predict_probas'].extend(logits.cpu().tolist())
                        predict_info["predict_labels"].extend((logits >= 0).int().data.cpu().tolist())
                        predict_info['target_labels'].extend(inputs['label'].cpu().tolist())
                        predict_info['graph_embeddings'].extend(graph_embedding.cpu().tolist())

                    elif self.config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
                        logits = self.model(inputs)
                        loss = self.criterion(logits, inputs['label'])

                        for i in range(len(self.n_classes)):
                            prediction = F.softmax(logits[i], dim=1)
                            if idx == 0:
                                predict_info['predict_probas'].append([])
                                predict_info["predict_labels"].append([])
                                predict_info['target_labels'].append([])
                            predict_info['predict_probas'][i].extend(prediction.cpu().tolist())
                            predict_info["predict_labels"][i].extend(prediction.max(1)[-1].cpu().tolist())
                            predict_info["target_labels"][i].extend(inputs['label'][i].cpu().tolist())

                    valid_losses.append(loss.item())
                
                valid_loss = np.mean(valid_losses)
                if self.config.model.type == "PreAttnMMs_GCN_MAP_V1":
                    metrics = evaluate(self.config,
                                    metrics,
                                    np.asarray(predict_info['predict_probas'])[:,1:],
                                    np.asarray(predict_info["predict_labels"])[:,1:],
                                    np.asarray(predict_info['target_labels'])[:,1:],
                                    self.n_classes)
                else:
                    metrics = evaluate(self.config,
                                    metrics,
                                    predict_info['predict_probas'],
                                    predict_info["predict_labels"],
                                    predict_info['target_labels'],
                                    self.n_classes)

                logger_print(self.config, stage, epoch, valid_loss, metrics)

                if stage == 'VALIDATION':
                    # update learning rate
                    logger.info("epoch {0}: --- learning rate = {1}".format(epoch, self.optimizer.param_groups[0]['lr']))

                    self.scheduler.step(metrics[self.config.experiment.monitored_metric])
                
                self.trial.report(metrics[self.config.experiment.monitored_metric], epoch)

            return valid_loss, metrics, predict_info