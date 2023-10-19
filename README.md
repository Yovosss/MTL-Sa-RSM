# Hierarchical classification for differential diagnosis of fever of unknown origin: A Multi-Task Learning Approach with self-adaptive Representation Sharing

This repository implements the models for paper '[Hierarchical classification for differential diagnosis of fever of unknown origin: A Multi-Task Learning Approach with self-adaptive Representation Sharing]'.

We will keep update this repository, if there are any problems, please leave your questions in section Issues.

## Original Data

The directory `./data/Fuo/48hours` contains a `.pkl` file (i.e., `data(48hrs)(ratio=811).pkl`). which is just our provided demo data. There are only two samples under each data items.

> We would like to emphasize that the dataset we used in our study was extracted from the EHR system at the First Affiliated Hospital of Zhejiang University School of Medicine (FAHZU) and contains real-world patient information. We have taken appropriate measures to protect patient privacy, including anonymization and obtaining approval from our Institutional Review Board (IRB). However, we must consider data security issues to ensure that the data is not misused or accessed inappropriately. Therefore, we are unable to make the dataset publicly available. However, we are willing to share the dataset with other researchers who are interested in our research for nonprofit research purposes on a case-by-case basis.

Next, we will provide a detailed explanation of the specific format and organizational structure of our input data.

First, you can read the `.pkl` file using the below code:
```python
import pickle as pkl
data_path = "./data/Fuo/48hours/data(48hrs)(ratio=811).pkl"
with open(data_path, 'rb') as f:
    processed_dict = pkl.load(f, encoding='bytes', errors='ignore')

data = processed_dict['data']
label = processed_dict['label']
indices = processed_dict['indices']
```
+ `data` is still a dict and includes all data elements, such as the **non-time-series data** and **time-series data**. All of the keys of this dict are shown as follows:
  > X_t <br>
    X_t_mask <br>
    T_t <br>
    T_t_rel <br>
    deltaT_t <br>
    static_data_val <br>
    static_data_cat <br>
    static_data_cat_onehot <br>
    X_t_features <br>
    X_t_steps <br>
    X_features <br>
+ `label` is also a dict and contains all the label-related data. All of the keys of this dict can be seen as follows:
  > label_data <br>
    y_classes <br>
    label2id <br>
    y_classes_unique <br>
    unique_label_number <br>
    taxonomy <br>
+ `indices` is also a dict and contains the sample indices after the dataset partition and statistical information for each dataset.
  > folds_idx <br>
    folds_idx_with_txy <br>
    folds_stats <br>

## Run Experiment

We have conducted many local and global model for hierarchical classification, you can execute the code below to run the models. Here is the label hierachy we have constructed for differential diagnosis of FUO etiologies.

<div align=center><img height='250' width='800' src=./figures/hierarchy.png></div>

### Flat models

**1. a backbone model with Flat Classifier for Leaf Nodes (FCLN)</br>**
   i.e. with 7 elements in output layer, and get the parent node label with hierarchical constraint;</br>

+ `PreAttnMMs_FCLN.json`: the model have been finished, and it will be a multi-class classification.

+ Run `PreAttnMMs_FCLN.json` for hyperparameter tunning.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_FCLN/Standardization/log.log &

+ Run TEST for `PreAttnMMs_FCLN`.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_FCLN/Standardization/log.log &

+ Run bootstrap for test dataset, and save the results for later comparison in article.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_FCLN/btsp/log.log &

**2. a backbone model with Flat Classifier for all Nodes (FCAN) </br>**
  i.e. with 11 elements in output layer, and get the all node label once.

+ `PreAttnMMs_FCAN.json`: the model has been finished, and it will be a multi-label classification.

+ Run `PreAttnMMs_FCAN.json` for hyperparameter tunning.
  > > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_FCAN/Standardization/log.log &

+ Run TEST for `PreAttnMMs_FCAN`.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_FCAN/Standardization/log.log &

+ Run bootstrap for test dataset, and save the results for later comparison in article.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_FCAN/Standardization/btsp.log &

### Local models

**1. Top-down mode with Local Classifier per Parent Node (LCPN) </br>**

+ `PreAttnMMs_LCPN.json`: transform the hierarchical classification into multiple parent-node multiclass classification, and under constraints of label hierarchy during test phase.

+ Run `PreAttnMMs_LCPN.json` for hyperparameter tunning.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_LCPN/node-0/Standardization/log.log &

  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_LCPN/node-1/Standardization/log.log &

  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_LCPN/node-2/Standardization/log.log &

  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_LCPN/node-6/Standardization/log.log &

  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_LCPN/node-7/Standardization/log.log &

+ Run hierarchical test classification for all local task, the script is `2_Test_LCPN.ipynb`.
+ Run bootstrap for test dataset, and save the results for later comparison in article.
  > nohup python 2_Test_LCPN_btsp.py >> ./shell_log/hp_tuning/PreAttnMMs_LCPN/btsp/log.log &

### Gloabl models

**1. HMCN </br>**

+ `PreAttnMMS_HMCN.json`: the model has been finished, and it will be multi-label classification.

+ Run `PreAttnMMs_HMCN.json` for hyperparameter tunning.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_HMCN/Standardization/log.log &

+ Run TEST for `PreAttnMMs_HMCN`.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_HMCN/Standardization/log.log &

+ Run bootstrap for test dataset, and save the results for later comparison in article.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_HMCN/Standardization/btsp.log &

**2. ML_GCN </br>**

+ `PreAttnMMs_GCN_MAP_V1.json`: using GCN to model the label hierarchy and map the hierarchical classification into a multi-label classification.

+ Run `PreAttnMMs_GCN_MAP_V1.json` for hyperparameter tunning.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GCN_MAP_V1/Standardization/log.log &

+ Run TEST for `PreAttnMMs_GCN_MAP_V1`.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GCN_MAP_V1/Standardization/log.log &

+ Run bootstrap for test dataset, and save the results for later comparison in article.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GCN_MAP_V1/Standardization/btsp.log &

**3. Multi-task learning with Parent node-wise Task Decomposition (MTL-PN) </br>**

+ `PreAttnMMs_MTL_IMP2.json`:
  transform the hierarchical classification into multi-task learning, and decompose the original hierarchical classification into 5 parent-node multi-class classification task.

+ Run `PreAttnMMs_MTL_IMP2.json` for hyperparameter tunning
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_MTL_IMP2/Standardization/log.log &
  
+ Run TEST for `PreAttnMMs_MTL_IMP2`.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_MTL_IMP2/Standardization/log.log &  
  
+ Run bootstrap for test dataset, and save the results for later comparison in article.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_MTL_IMP2/Standardization/btsp.log &

**4. Multi-task Learning with Level-wise Task Decomposition (MTL-L) </br>**
+ `PreAttnMMs_MTL_LCL.json`:  
  The model is also based on the multi-task neural network architecture, and its special feature is that its task decomposition adopts the level-by-level approach that has been used most in previous studies, that is, the categories within each level in the label hierarchy corresponds to a classification head.

+ Run `PreAttnMMs_MTL_LCL.json` for hyperparameter tunning
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_MTL_LCL/Standardization/log.log &

+ Run TEST for `PreAttnMMs_MTL_LCL`.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_MTL_LCL/Standardization/log.log &  

+ Run bootstrap for test dataset, and save the results for later comparison in article.
  > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_MTL_LCL/Standardization/btsp.log &

**5. Our Proposed Model (OURS) </br>**

+ `PreAttnMMs_GAT_IMP8_GC.json`:
  	Our proposed model contains our feature extraction module and Sa-RSM with K classification heads.

+ Run `PreAttnMMs_GAT_IMP8_GC.json` for hyperparameter tunning
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GAT_IMP8_GC/Standardization/log.log &

+ Run TEST for `PreAttnMMs_GAT_IMP8_GC`.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GAT_IMP8_GC/Standardization/log.log &

+ Run bootstrap for test dataset, and save the results for later comparison in article.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GAT_IMP8_GC/Standardization/btsp.log &

**6. Our Proposed Model with Weighted Loss (OURS-WL) </br>**

+ `PreAttnMMs_GAT_IMP8_GC_WeightedLoss.json`:
  The only difference between OURS and this model is that it automatically weights multiple loss functions by considering the homoscedastic uncertainty of each task.

+ Run `PreAttnMMs_GAT_IMP8_GC_WeightedLoss.json` for hyperparameter tunning
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GAT_IMP8_GC_WeightedLoss/Standardization/log.log &

+ Run TEST for `PreAttnMMs_GAT_IMP8_GC_WeightedLoss`.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GAT_IMP8_GC_WeightedLoss/Standardization/log.log &

+ Run bootstrap for test dataset, and save the results for later comparison in article.
    > nohup python Run.py >> ./shell_log/hp_tuning/PreAttnMMs_GAT_IMP8_GC_WeightedLoss/Standardization/btsp.log &


## ***ENVIRONMENT***
The main environment configuration can be seen below:
```
conda env update -n hie_attn -f environment.yml
source activate hie_attn
pip install -e .
```
there may be some conflicts that need to be manually fixed.
