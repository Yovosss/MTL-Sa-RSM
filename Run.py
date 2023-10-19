#!/usr/bin/env python
# coding:utf-8
import os
import sys
import time

import torch
from addict import Dict
from sacred import Experiment
from sacred.observers import FileStorageObserver

import helper.logger as logger
from helper.utils import get_config, set_seed
from models.main import run_experiment, run_experiment_test, run_experiment_btsp
from models.main_hp import run_experiment_hp
from models.visualize import run_visualization

ex = Experiment()

@ex.main
def start(_config, _run):
    """
    Sacred内置的变量
    _conifg: 所有的参数作为一个字典（只读的）
    _run: 当前实验运行时的run对象
    """
    config = Dict(_config)
    logger.info("CONFIGURE: {}".format(config))
    set_seed(seed=config.seed)
    if config.experiment.hp_tuning:
        # hyperparameter tune
        result = run_experiment_hp(config, _run)
    else:
        if config.experiment.test == True:
            result = run_experiment_test(config, _run)
        elif config.experiment.btsp == True:
            result = run_experiment_btsp(config, _run)
        elif config.experiment.visualize == True:
            result = run_visualization(config, _run)
        else:
            # directly training
            result = run_experiment(config, _run)

    return result


if __name__ == "__main__":

    CONFIG_ID = 'PreAttnMMs_MTL_IMP2'
    config = get_config(config_id=CONFIG_ID)
    ex.add_config(config)

    logger.Logger(config)

    # set the log configuration
    options = {}
    options['--name'] = 'exp_{}'.format(CONFIG_ID)
    if config["logging"]["use_mongo"]:
        options['--mongo_db'] = '{}:{}:{}'.format(config["logging"]["log"]["mongo_host"],
                                        config["logging"]["log"]["mongo_port"],
                                        config["logging"]["log"]["mongo_db"])
    else:
        base_path = str(os.path.dirname(os.path.realpath(__file__)))
        log_path = os.path.join(base_path, config["logging"]["dir"])
        ex.observers.append(FileStorageObserver.create(log_path))
    
    ex.run(options=options)


