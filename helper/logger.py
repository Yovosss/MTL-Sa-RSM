#!/usr/bin/env python
# coding:utf-8

import logging
import os

logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}


def debug(msg):
    logging.debug(msg)
    print('DEBUG: ', msg)


def info(msg):
    logging.info(msg)
    print('INFO: ', msg)


def warning(msg):
    logging.warning(msg)
    print('WARNING: ', msg)


def error(msg):
    logging.error(msg)
    print('ERROR: ', msg)


def fatal(msg):
    logging.critical(msg)
    print('FATAL: ', msg)


class Logger(object):
    def __init__(self, config):
        """
        set the logging module
        :param config: helper.configure, Configure object
        """
        super(Logger, self).__init__()
        assert config["logger"]["level"] in logging_level.keys()
        logging.getLogger('').handlers = []

        # define log file path
        log_base = config["logger"]["dir"]
        if config["experiment"]["hp_tuning"]:
            if config["experiment"]["type"] == "flat":
                log_dir = os.path.join(log_base, 
                                       'hp_tuning', 
                                       config["model"]["type"],
                                       config["data"]["norm_type"],
                                       config["experiment"]["monitored_metric"])
            elif config["experiment"]["type"] == "local":
                log_dir = os.path.join(log_base, 
                                       'hp_tuning', 
                                       config["model"]["type"],
                                       'node-{}'.format(config["experiment"]["local_task"]),
                                       config["data"]["norm_type"],
                                       config["experiment"]["monitored_metric"])
            elif config["experiment"]["type"] == "global":
                log_dir = os.path.join(log_base, 
                                       'hp_tuning', 
                                       config["model"]["type"],
                                       config["data"]["norm_type"],
                                       config["experiment"]["monitored_metric"])
        else:
            if config["experiment"]["type"] == "flat":
                log_dir = os.path.join(log_base, 
                                       'no_hp_tuning', 
                                       config["model"]["type"],
                                       config["data"]["norm_type"],
                                       config["experiment"]["monitored_metric"])
            elif config["experiment"]["type"] == "local":
                log_dir = os.path.join(log_base, 
                                       'no_hp_tuning', 
                                       config["model"]["type"],
                                       'node-{}'.format(config["experiment"]["local_task"]),
                                       config["data"]["norm_type"],
                                       config["experiment"]["monitored_metric"])
            elif config["experiment"]["type"] == "global":
                log_dir = os.path.join(log_base, 
                                       'no_hp_tuning', 
                                       config["model"]["type"],
                                       config["data"]["norm_type"],
                                       config["experiment"]["monitored_metric"])

        if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

        logging.basicConfig(filename=os.path.join(log_dir, 'logger.log'),
                            level=logging_level[config["logger"]["level"]],
                            format='%(asctime)s - %(levelname)s : %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')

