import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
import argparse

# path
Dataset_Path = '../Data/'
Pt_Path = '../pt/'
out_dir = '../Data/node_features_noblhhm/'


Node_Feature_num = 1328

aalist = list('ACDEFGHIKLMNPQRSTVWY')
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
polar_aa = 'AVLIFWMP'
nonpolar_aa = 'GSTCYNQHKRDE'

with open('./aa_phy7','r') as f:
    pccp = f.read().splitlines()
    pccp = [i.split() for i in pccp]
    pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}

def read_pccp(seq):
    return np.array([pccp_dic[i] for i in seq])

def get_AAfq(seq):
    AAfq_dic = dict()
    AAfq = np.array([seq.count(x) for x in aalist])/len(seq)
    for (key,value) in zip(aalist,AAfq):
         AAfq_dic[key] = value
    seq_AAfq = np.array([AAfq_dic[x] for x in seq])
    return seq_AAfq

def do_count(seq):
    dimers = Counter()
    for i in range(len(seq)-1):
        dimers[seq[i:i+2]] += 1.0
    return dimers

def get_dipfq(seq):
    result = do_count(seq)
    dimers = sum(result.values())
    dimers_fq = dict()
    for a1 in amino_acids:
        for a2 in amino_acids:
            dimers_fq[a1+a2] = (result[a1+a2]*1.0)/dimers
    dipfq ={}
    for x in aalist:
        a=[]
        b=[]
        for y in aalist:
            a.append(dimers_fq[x+y])
            b.append(dimers_fq[y+x])
        dipfq[x] = np.hstack((a,b))
    seq_dipfq = []
    for x in seq:
        seq_dipfq.append(dipfq[x].tolist())
    return seq_dipfq
    
def get_matrix(df):
    for i in tqdm(range(len(df))):
        uniprot_id = str(df.loc[i,"uniprot_id"])
        sequence = df.loc[i,"sequence"]   
        # L * 1
        AAfq = get_AAfq(sequence)       
        # L * 40
        dipfq = get_dipfq(sequence)
        # L * 7
        PP7 = read_pccp(sequence)
        # L * 1280
        esm_feature_path = Pt_Path +  uniprot_id + '.pt'  
        esm_feat = torch.load(esm_feature_path)["representations"][33].numpy()          
        esm_feat_truncated = esm_feat[:, 1:-1, :]  # [1, 116, 1028]
        
        esm = esm_feat_truncated[0]
        # print(esm.shape)
        matrix = np.concatenate([PP7,np.array(AAfq).reshape(-1,1),np.array(dipfq), esm],axis=1)
        # print(uniprot_id, matrix.shape)
        np.save(out_dir + uniprot_id + '.npy',matrix)


def cal_mean_std(df):
    fastalist = df["uniprot_id"].astype(str).to_list()
    ogt = df['ogt'].values
    total_length = 0
    oneD_mean = np.zeros(Node_Feature_num)
    oneD_mean_square = np.zeros(Node_Feature_num)
    for name in tqdm(fastalist):
        matrix = np.load(out_dir + name+'.npy')
        total_length += matrix.shape[0]
        oneD_mean += np.sum(matrix, axis=0)
        oneD_mean_square += np.sum(np.square(matrix),axis=0)
    oneD_mean /= total_length  # EX
    oneD_mean_square /= total_length  # E(X^2)
    oneD_std = np.sqrt(np.subtract(oneD_mean_square, np.square(oneD_mean))) 
    np.save(f'{Dataset_Path}mean_noblhhm.npy', oneD_mean)
    np.save(f'{Dataset_Path}std_noblhhm.npy', oneD_std)
    np.save(f'{Dataset_Path}mean_ogt.npy', np.array([np.mean(ogt)]))
    np.save(f'{Dataset_Path}std_ogt.npy', np.array([np.std(ogt)]))

def main():
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../TmpredTrain.csv",
                    help='Please input the path of CSV (default：../TmpredTrain.csv)')
    args = parser.parse_args()
    df = pd.read_csv(args.input)    
    os.makedirs(out_dir, exist_ok=True)
    get_matrix(df)
    cal_mean_std(df)
    
if __name__ == "__main__":
    sys.exit(main())