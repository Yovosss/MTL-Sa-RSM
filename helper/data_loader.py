#!/usr/bin/env python
# coding:utf-8

from torch.utils.data import DataLoader

import helper.logger as logger
from helper.collator import FlatCollator, GlobalCollator, LocalCollator
from helper.dataset import FlatDataset, GlobalHcDataset, LocalHcDataset


def data_loaders(config, data, label, indices):
    """
    get data loaders for training and evaluation
    :param config: Object
    :param data: 
    :param label: 
    :param indices:
    :return: -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    if config.experiment.type == "flat":
        """
        Generate data loader for flat experiment
        """
        collate_fn = FlatCollator(config, label)
        train_dataset = FlatDataset(config, data, label, indices, stage="TRAIN")
        train_loader = DataLoader(train_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=True,
                                num_workers=config.train.device_setting.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True)

        validation_dataset = FlatDataset(config, data, label, indices, stage="VALIDATION")
        validation_loader = DataLoader(validation_dataset,
                                    batch_size=config.train.batch_size,
                                    shuffle=False,
                                    num_workers=config.train.device_setting.num_workers,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    drop_last=True)
        
        test_dataset = FlatDataset(config, data, label, indices, stage="TEST")
        test_loader = DataLoader(test_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=False,
                                num_workers=config.train.device_setting.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True)

    elif config.experiment.type == "local":
        """
        Generate data loader for local experiment
        """
        collate_fn = LocalCollator(config, label)
        train_dataset = LocalHcDataset(config, data, label, indices, stage="TRAIN")
        train_loader = DataLoader(train_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=True,
                                num_workers=config.train.device_setting.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True)

        validation_dataset = LocalHcDataset(config, data, label, indices, stage="VALIDATION")
        validation_loader = DataLoader(validation_dataset,
                                    batch_size=config.train.batch_size,
                                    shuffle=False,
                                    num_workers=config.train.device_setting.num_workers,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    drop_last=True)
        
        test_dataset = LocalHcDataset(config, data, label, indices, stage="TEST")
        test_loader = DataLoader(test_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=False,
                                num_workers=config.train.device_setting.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True)

    elif config.experiment.type == "global":
        """
        Generate data loader for global experiment
        """
        collate_fn = GlobalCollator(config, label)
        train_dataset = GlobalHcDataset(config, data, label, indices, stage="TRAIN")
        train_loader = DataLoader(train_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=True,
                                num_workers=config.train.device_setting.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True)

        validation_dataset = GlobalHcDataset(config, data, label, indices, stage="VALIDATION")
        validation_loader = DataLoader(validation_dataset,
                                    batch_size=config.train.batch_size,
                                    shuffle=False,
                                    num_workers=config.train.device_setting.num_workers,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    drop_last=True)
        
        test_dataset = GlobalHcDataset(config, data, label, indices, stage="TEST")
        test_loader = DataLoader(test_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=False,
                                num_workers=config.train.device_setting.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True)
    
    else:
        logger.error("The loaded dataset is not flat, local dataset or global dataset!")

    return train_loader, validation_loader, test_loader

