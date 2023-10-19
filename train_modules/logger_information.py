#!/usr/bin/env python
# coding:utf-8


import helper.logger as logger


def logger_print(config, stage, epoch, loss, metrics):
    """
    Functions: print the key information during training, validation and test

    config: dict, config parameters
    stage: str, e.g. 'TRAIN'/'DEV'/'TEST'
    epoch: int, epoch value
    loss: float, loss mean of each epoch
    metrics: dict, all metrics during training, validation and test

    Return: None
    """
    # codes here can only be used for flat classification mode of PreAttnMMs_FCLN
    if config.model.type == "PreAttnMMs_FCLN(for flat mode)":
        logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
        logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Accuracy: {acc:.4f}\t"
                    "Macro-AUC: {mAUC:.4f}\t"
                    "Macro-AP: {mAP:.4f}\t"
                    "Micro-AP: {miAP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        acc=metrics['acc'], 
                                                        mAUC=metrics['macro-auc'], 
                                                        mAP=metrics['macro-ap'],
                                                        miAP=metrics['micro-ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"]
                                                        ))
        logger.info("Precison: \t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(bac=metrics['precision'][0], 
                                                            vir=metrics['precision'][1],
                                                            fun=metrics['precision'][2],
                                                            ad=metrics['precision'][3],
                                                            aid=metrics['precision'][4],
                                                            hm=metrics['precision'][5],
                                                            sm=metrics['precision'][6]))
        logger.info("Recall: \t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(bac=metrics['recall'][0], 
                                                            vir=metrics['recall'][1],
                                                            fun=metrics['recall'][2],
                                                            ad=metrics['recall'][3],
                                                            aid=metrics['recall'][4],
                                                            hm=metrics['recall'][5],
                                                            sm=metrics['recall'][6]))
        logger.info("F1: \t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(bac=metrics['f1'][0], 
                                                            vir=metrics['f1'][1],
                                                            fun=metrics['f1'][2],
                                                            ad=metrics['f1'][3],
                                                            aid=metrics['f1'][4],
                                                            hm=metrics['f1'][5],
                                                            sm=metrics['f1'][6]))
        logger.info("AP: \t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(bac=metrics['ap'][0], 
                                                            vir=metrics['ap'][1],
                                                            fun=metrics['ap'][2],
                                                            ad=metrics['ap'][3],
                                                            aid=metrics['ap'][4],
                                                            hm=metrics['ap'][5],
                                                            sm=metrics['ap'][6]))

    elif config.model.type in ["PreAttnMMs_FCAN", "PreAttnMMs_HMCN", "PreAttnMMs_GCN_MAP_V1"]:
        logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
        logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Exact Match Ratio: {emr:.4f}\t"
                    "Macro-AUC: {mAUC:.4f}\t"
                    "Macro-AP: {mAP:.4f}\t"
                    "Micro-AP: {miAP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t"
                    "0-1 Loss: {zoloss:.4f}\t"
                    "Hamming Loss: {hmloss:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        emr=metrics['exact_match_ratio'], 
                                                        mAUC=metrics['macro-auc'],
                                                        mAP=metrics['macro-ap'],
                                                        miAP=metrics['micro-ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"],
                                                        zoloss=metrics['01loss'], 
                                                        hmloss=metrics['hamming_loss']))
        logger.info("Precison: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['precision'][0],
                                                            noninfec=metrics['precision'][1],
                                                            bac=metrics['precision'][2], 
                                                            vir=metrics['precision'][3],
                                                            fun=metrics['precision'][4],
                                                            niid=metrics['precision'][5],
                                                            neo=metrics['precision'][6],
                                                            ad=metrics['precision'][7],
                                                            aid=metrics['precision'][8],
                                                            hm=metrics['precision'][9],
                                                            sm=metrics['precision'][10]))
        logger.info("Recall: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['recall'][0],
                                                            noninfec=metrics['recall'][1],
                                                            bac=metrics['recall'][2], 
                                                            vir=metrics['recall'][3],
                                                            fun=metrics['recall'][4],
                                                            niid=metrics['recall'][5],
                                                            neo=metrics['recall'][6],
                                                            ad=metrics['recall'][7],
                                                            aid=metrics['recall'][8],
                                                            hm=metrics['recall'][9],
                                                            sm=metrics['recall'][10]))
        logger.info("F1: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['f1'][0],
                                                            noninfec=metrics['f1'][1],
                                                            bac=metrics['f1'][2], 
                                                            vir=metrics['f1'][3],
                                                            fun=metrics['f1'][4],
                                                            niid=metrics['f1'][5],
                                                            neo=metrics['f1'][6],
                                                            ad=metrics['f1'][7],
                                                            aid=metrics['f1'][8],
                                                            hm=metrics['f1'][9],
                                                            sm=metrics['f1'][10]))
        logger.info("AP: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['ap'][0],
                                                            noninfec=metrics['ap'][1],
                                                            bac=metrics['ap'][2], 
                                                            vir=metrics['ap'][3],
                                                            fun=metrics['ap'][4],
                                                            niid=metrics['ap'][5],
                                                            neo=metrics['ap'][6],
                                                            ad=metrics['ap'][7],
                                                            aid=metrics['ap'][8],
                                                            hm=metrics['ap'][9],
                                                            sm=metrics['ap'][10]))

    elif config.model.type == "PreAttnMMs_LCPN":
        if config.experiment.local_task == 0:
            logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
            logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Accuracy: {acc:.4f}\t"
                    "AUC: {AUC:.4f}\t"
                    "AP: {AP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        acc=metrics['acc'], 
                                                        AUC=metrics['auc'], 
                                                        AP=metrics['ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"]))
            logger.info("Precison: \t"
                        "infections: {infec:.4f}\t"
                        "noninfections: {noninfec:.4f}\t".format(infec=metrics['precision'][0],
                                                                   noninfec=metrics['precision'][1]))
            logger.info("Recall: \t"
                        "infections: {infec:.4f}\t"
                        "noninfections: {noninfec:.4f}\t".format(infec=metrics['recall'][0],
                                                                   noninfec=metrics['recall'][1]))
            logger.info("F1: \t"
                        "infections: {infec:.4f}\t"
                        "noninfections: {noninfec:.4f}\t".format(infec=metrics['f1'][0],
                                                                   noninfec=metrics['f1'][1]))
        elif config.experiment.local_task == 1:
            logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
            logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Accuracy: {acc:.4f}\t"
                    "Macro-AUC: {mAUC:.4f}\t"
                    "Macro-AP: {mAP:.4f}\t"
                    "Micro-AP: {miAP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        acc=metrics['acc'], 
                                                        mAUC=metrics['macro-auc'], 
                                                        mAP=metrics['macro-ap'],
                                                        miAP=metrics['micro-ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"]
                                                        ))
            logger.info("Precison: \t"
                        "bacterial infection: {bac:.4f}\t"
                        "viral infection: {vir:.4f}\t"
                        "fungal infection: {fun:.4f}\t".format(bac=metrics['precision'][0],
                                                                 vir=metrics['precision'][1],
                                                                 fun=metrics['precision'][2]))
            logger.info("Recall: \t"
                        "bacterial infection: {bac:.4f}\t"
                        "viral infection: {vir:.4f}\t"
                        "fungal infection: {fun:.4f}\t".format(bac=metrics['recall'][0],
                                                                 vir=metrics['recall'][1],
                                                                 fun=metrics['recall'][2]))
            logger.info("F1: \t"
                        "bacterial infection: {bac:.4f}\t"
                        "viral infection: {vir:.4f}\t"
                        "fungal infection: {fun:.4f}\t".format(bac=metrics['f1'][0],
                                                                 vir=metrics['f1'][1],
                                                                 fun=metrics['f1'][2]))
            logger.info("AP: \t"
                        "bacterial infection: {bac:.4f}\t"
                        "viral infection: {vir:.4f}\t"
                        "fungal infection: {fun:.4f}\t".format(bac=metrics['ap'][0],
                                                                 vir=metrics['ap'][1],
                                                                 fun=metrics['ap'][2]))
        elif config.experiment.local_task == 2:
            logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
            logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Accuracy: {acc:.4f}\t"
                    "AUC: {AUC:.4f}\t"
                    "AP: {AP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        acc=metrics['acc'], 
                                                        AUC=metrics['auc'], 
                                                        AP=metrics['ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"]))
            logger.info("Precison: \t"
                        "NIID: {niid:.4f}\t"
                        "neoplastic disease: {neo:.4f}\t".format(niid=metrics['precision'][0],
                                                                   neo=metrics['precision'][1]))
            logger.info("Recall: \t"
                        "NIID: {niid:.4f}\t"
                        "neoplastic disease: {neo:.4f}\t".format(niid=metrics['recall'][0],
                                                                   neo=metrics['recall'][1]))
            logger.info("F1: \t"
                        "NIID: {niid:.4f}\t"
                        "neoplastic disease: {neo:.4f}\t".format(niid=metrics['f1'][0],
                                                                   neo=metrics['f1'][1]))
        elif config.experiment.local_task == 6:
            logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
            logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Accuracy: {acc:.4f}\t"
                    "AUC: {AUC:.4f}\t"
                    "AP: {AP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        acc=metrics['acc'], 
                                                        AUC=metrics['auc'], 
                                                        AP=metrics['ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"]))
            logger.info("Precison: \t"
                        "autoimmune disease: {ad:.4f}\t"
                        "autoinflammatory disease: {aid:.4f}\t".format(ad=metrics['precision'][0],
                                                                         aid=metrics['precision'][1]))
            logger.info("Recall: \t"
                        "autoimmune disease: {ad:.4f}\t"
                        "autoinflammatory disease: {aid:.4f}\t".format(ad=metrics['recall'][0],
                                                                         aid=metrics['recall'][1]))
            logger.info("F1: \t"
                        "autoimmune disease: {ad:.4f}\t"
                        "autoinflammatory disease: {aid:.4f}\t".format(ad=metrics['f1'][0],
                                                                         aid=metrics['f1'][1]))
        elif config.experiment.local_task == 7:
            logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
            logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Accuracy: {acc:.4f}\t"
                    "AUC: {AUC:.4f}\t"
                    "AP: {AP:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        acc=metrics['acc'], 
                                                        AUC=metrics['auc'], 
                                                        AP=metrics['ap'],
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"]))
            logger.info("Precison: \t"
                        "hematological malignancy: {hm:.4f}\t"
                        "solid malignancy: {sm:.4f}\t".format(hm=metrics['precision'][0],
                                                                sm=metrics['precision'][1]))
            logger.info("Recall: \t"
                        "hematological malignancy: {hm:.4f}\t"
                        "solid malignancy: {sm:.4f}\t".format(hm=metrics['recall'][0],
                                                                sm=metrics['recall'][1]))
            logger.info("F1: \t"
                        "hematological malignancy: {hm:.4f}\t"
                        "solid malignancy: {sm:.4f}\t".format(hm=metrics['f1'][0],
                                                                sm=metrics['f1'][1]))

    elif config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss", "PreAttnMMs_FCLN", "PreAttnMMs_MTL_IMP3"]:
        logger.info("------------------------------{0}------------------------------".format(str.upper(stage)))
        logger.info("epoch: {0}\t" 
                    "Loss: {loss:.4f}\t"
                    "Exact Match Ratio: {emr:.4f}\t"
                    "Macro-Precision: {macroP:.4f}\t"
                    "Micro-Precision: {microP:.4f}\t"
                    "Macro-Recall: {macroR:.4f}\t"
                    "Micro-Recall: {microR:.4f}\t"
                    "Macro-F1: {macroF1:.4f}\t"
                    "Micro-F1: {microF1:.4f}\t"
                    "0-1 Loss: {zoloss:.4f}\t"
                    "Hamming Loss: {hmloss:.4f}\t".format(epoch, 
                                                        loss=loss, 
                                                        emr=metrics['exact_match_ratio'], 
                                                        macroP=metrics["macro-precision"],
                                                        microP=metrics["micro-precision"],
                                                        macroR=metrics["macro-recall"],
                                                        microR=metrics["micro-recall"],
                                                        macroF1=metrics["macro-f1"],
                                                        microF1=metrics["micro-f1"],
                                                        zoloss=metrics['01loss'], 
                                                        hmloss=metrics['hamming_loss']))
        logger.info("Precison: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['precision'][0],
                                                            noninfec=metrics['precision'][1],
                                                            bac=metrics['precision'][2], 
                                                            vir=metrics['precision'][3],
                                                            fun=metrics['precision'][4],
                                                            niid=metrics['precision'][5],
                                                            neo=metrics['precision'][6],
                                                            ad=metrics['precision'][7],
                                                            aid=metrics['precision'][8],
                                                            hm=metrics['precision'][9],
                                                            sm=metrics['precision'][10]))
        logger.info("Recall: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['recall'][0],
                                                            noninfec=metrics['recall'][1],
                                                            bac=metrics['recall'][2], 
                                                            vir=metrics['recall'][3],
                                                            fun=metrics['recall'][4],
                                                            niid=metrics['recall'][5],
                                                            neo=metrics['recall'][6],
                                                            ad=metrics['recall'][7],
                                                            aid=metrics['recall'][8],
                                                            hm=metrics['recall'][9],
                                                            sm=metrics['recall'][10]))
        logger.info("F1: \t"
                    "infections: {infec:.4f}\t"
                    "noninfections: {noninfec:.4f}\t"
                    "bacterial infection: {bac:.4f}\t"
                    "viral infection: {vir:.4f}\t"
                    "fungal infection: {fun:.4f}\t"
                    "NIID: {niid:.4f}\t"
                    "neoplastic disease: {neo:.4f}\t"
                    "autoimmune disease: {ad:.4f}\t"
                    "autoinflammatory disease: {aid:.4f}\t"
                    "hematological malignancy: {hm:.4f}\t"
                    "solid malignancy: {sm:.4f}\t".format(infec=metrics['f1'][0],
                                                            noninfec=metrics['f1'][1],
                                                            bac=metrics['f1'][2], 
                                                            vir=metrics['f1'][3],
                                                            fun=metrics['f1'][4],
                                                            niid=metrics['f1'][5],
                                                            neo=metrics['f1'][6],
                                                            ad=metrics['f1'][7],
                                                            aid=metrics['f1'][8],
                                                            hm=metrics['f1'][9],
                                                            sm=metrics['f1'][10]))