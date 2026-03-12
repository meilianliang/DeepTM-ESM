import os
import torch
import numpy as np

# path
Dataset_Path = '../Data/'
Model_Path = '../Model/'
Result_Path = '../Result/'
TrainData_Path = '../TmpreTrain.csv'
TestData_Path = '../pet_more.csv'

# Seed
SEED = 2333

# Model parameters
dataset = os.environ.get('DATASET', 'default')
if dataset == 'DeepSTABp':
    LENG_SIZE = 1750
else:
    LENG_SIZE = 1028
NUMBER_EPOCHS = 50
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 1E-4
BATCH_SIZE = 32
NUM_CLASSES = 1
LENG_SIZE = 1028
TRADITION_SIZE = 48
FEAT_LEN = 1328
FEAT_FLAG = 'noblhhm'

# MLP parameters
MLP_IN_DIM = 1280
MLP_HIDDEN_DIM = 512
MLP_OUTPUT_DIM = 128

# GCN parameters
GCN_FEATURE_DIM = 176
GCN_HIDDEN_DIM = 1024
GCN_OUTPUT_DIM = 128

# Attention parameters
DENSE_DIM = 64
ATTENTION_HEADS = 4

device_ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


amino_acid = list("ACDEFGHIKLMNPQRSTVWY")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

aalist = list('ACDEFGHIKLMNPQRSTVWY')
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
polar_aa = 'AVLIFWMP'
nonpolar_aa = 'GSTCYNQHKRDE'