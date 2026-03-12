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
fastalist = []


def read_pccp(seq):
    if not pccp_dic:
        raise ValueError("pccp_dic is empty!")
    dummy_zero = np.zeros_like(next(iter(pccp_dic.values())))
    return np.array([pccp_dic.get(i, dummy_zero) for i in seq])


def get_AAfq(seq):
    AAfq_dic = dict()
    AAfq = np.array([seq.count(x) for x in aalist])/len(seq)
    for (key,value) in zip(aalist,AAfq):
         AAfq_dic[key] = value
    seq_AAfq = np.array([AAfq_dic.get(x, 0) for x in seq])
    return seq_AAfq


def do_count(seq):
    dimers = Counter()    
    for i in range(len(seq)-1):
        if seq[i] in amino_acids and seq[i+1] in amino_acids:
           dimers[seq[i:i+2]] += 1.0
    return dimers

def get_dipfq(seq):
    result = do_count(seq)
    
    dimers = sum(result.values())
    
    if dimers == 0:
        dimers = 1
    
    dimers_fq = dict()
    for a1 in amino_acids:
        for a2 in amino_acids:
            dimer = a1 + a2
            
            dimers_fq[dimer] = result.get(dimer, 0) / dimers
    
    dipfq = {}
    for x in aalist:
        a = [dimers_fq.get(x + y, 0) for y in aalist]
        b = [dimers_fq.get(y + x, 0) for y in aalist]
        dipfq[x] = np.hstack((a, b))
    
    seq_dipfq = []
    zero_vector = np.zeros(2 * len(aalist))
    for x in seq:
        seq_dipfq.append(dipfq.get(x, zero_vector).tolist())
    return seq_dipfq

   
def get_matrix(df):
    for i in tqdm(range(len(df))):
        uniprot_id = str(df.loc[i,"uniprot_id"])
        sequence = df.loc[i,"sequence"]  
        sequence = sequence[:1750]
        
        # L * 1
        AAfq = get_AAfq(sequence)
        # L * 40
        dipfq = get_dipfq(sequence)
        # L * 7
        PP7 = read_pccp(sequence)
        # L * 1028
        esm_feature_path = Pt_Path +  uniprot_id + '.pt'           
        esm_feat = torch.load(esm_feature_path)["representations"][33].numpy()          
        esm_feat_truncated = esm_feat[:, 1:-1, :]  # [1, 116, 1028]

        esm = esm_feat_truncated[0]
        print(esm.shape)
        matrix = np.concatenate([PP7,np.array(AAfq).reshape(-1,1),np.array(dipfq), esm],axis=1)
        print(uniprot_id, matrix.shape)
        np.save(out_dir + uniprot_id + '.npy',matrix)

def main():
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../DeepStabpTrain.csv",
                    help='Please input the path of CSV (default：../DeepStabpTrain.csv)')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    fastalist = df["uniprot_id"].astype(str).to_list()
    os.makedirs(out_dir, exist_ok=True)
    get_matrix(df)
    
if __name__ == "__main__":
    sys.exit(main())