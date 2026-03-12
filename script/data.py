import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from config import *

def load_ogt_dict(Ogt):
    ogt_dic = {}
    ogt_dic_uniprotid = np.array([str(n) for n in Ogt['uniprot_id'].values])
    #ogt_dic_uniprotid = Ogt['uniprot_id'].values
    ogt_dic_topt = Ogt['ogt'].values
    for i in range(len(ogt_dic_uniprotid)):
        ogt_dic[ogt_dic_uniprotid[i]] = ogt_dic_topt[i]
    return ogt_dic

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_sequences(sequence_path):
    names, sequences, labels = ([] for i in range(3))
    for file_name in tqdm(os.listdir(sequence_path)):
        with open(sequence_path + file_name, 'r') as file_reader:
            lines = file_reader.read().split('\n')
            names.append(file_name)
            sequences.append(lines[1])
            labels.append(int(lines[2]))
    return pd.DataFrame({'names': names, 'sequences': sequences, 'labels': labels})


def load_features(uniprot_id, ogt_dic, mean, std):
    # len(sequence) * 1382
    feature_matrix1 = np.load(Dataset_Path + f'node_features_{FEAT_FLAG}/' + uniprot_id + '.npy')
    feature_matrix2 = np.ones(feature_matrix1.shape[0])
    feature_matrix2 = feature_matrix2 * ogt_dic[uniprot_id]
    feature_matrix = np.concatenate((feature_matrix1, feature_matrix2.reshape(-1,1)),axis=1)
    feature_matrix = (feature_matrix - mean) / std
    if "nobl" not in FEAT_FLAG:
        part1 = feature_matrix[:,0:20]
        part2 = feature_matrix[:,23:]  
        
        feature_matrix = np.concatenate([part1,part2],axis=1)
    # print(uniprot_id, feature_matrix.shape)
    return feature_matrix

def load_graph(sequence_name):
    matrix = np.load(Dataset_Path + 'edge_features_Tm/' + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix

class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = np.array([str(n) for n in dataframe['uniprot_id'].values])
        #self.names = dataframe['uniprot_id'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['tm'].values
        self.mean, self.std = self.load_values()
        self.ogt_dic = load_ogt_dict(dataframe)
        
    def load_blosum(self):
        with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
            result = {}
            next(f)
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                result[line[0]] = [int(i) for i in line[1:]]
        return result

    def load_values(self):
        mean1 = np.load(Dataset_Path + f'mean_{FEAT_FLAG}.npy')
        std1 = np.load(Dataset_Path + f'std_{FEAT_FLAG}.npy')
        mean2 = np.load(Dataset_Path + 'mean_ogt.npy')
        std2 = np.load(Dataset_Path + 'std_ogt.npy')        
        mean = np.concatenate([mean1, mean2])
        std = np.concatenate([std1, std2])
        return mean, std
        
    def __getitem__(self, index):
        uniprot_id = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        # L * 94
        sequence_feature = load_features(uniprot_id, self.ogt_dic, self.mean, self.std)
        
        pad_size_feature = LENG_SIZE - len(sequence)
        sequence_feature = np.pad(sequence_feature, ((0, pad_size_feature), (0, 0)), 'constant')
        
        assert sequence_feature.shape == (LENG_SIZE, FEAT_LEN), f"Wrong: {sequence_feature.shape}"
        # L * L
        sequence_graph = load_graph(uniprot_id)
        pad_size_graph = LENG_SIZE - len(sequence)
        # print(sequence_graph.shape)
        if pad_size_graph < 0:
            raise ValueError(f"Sequence {uniprot_id} too long")
        if sequence_graph.ndim == 2:
            sequence_graph = np.pad(sequence_graph, ((0, pad_size_graph), (0, pad_size_graph)), 'constant')
        elif sequence_graph.ndim == 3:
            sequence_graph = np.pad(sequence_graph, ((0, pad_size_graph), (0, pad_size_graph), (0, 0)), 'constant')
        else:
            raise ValueError(f"Unsupported graph dimension: {sequence_graph.ndim}") 
        # print(sequence_graph.shape)
        assert sequence_graph.shape == (LENG_SIZE, LENG_SIZE), f"Wrong: {sequence_graph.shape}"
        
        # print(f"Features size: {sequence_feature.nbytes / 1e6:.2f} MB")
        # print(f"Graph size: {sequence_graph.nbytes / 1e6:.2f} MB")
      
        
        return uniprot_id, sequence, label, sequence_feature, sequence_graph

    def __len__(self):
        return len(self.labels)