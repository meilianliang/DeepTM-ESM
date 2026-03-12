import os
import numpy as np
import torch
import argparse
import pandas as pd
from tqdm import tqdm  
import argparse

def normalize_adj(mx):    
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv @ mx @ r_mat_inv

def get_esm_contacts(ptdir, name, key=33):    
    data = torch.load(os.path.join(ptdir, f'{name}.pt'))
    contacts = data['contacts'].detach().float().cpu().numpy()  # [L, L]
    return contacts

def save_esm_contacts_as_npy(names, sequences, ptdir, output_dir, key=33):    
    os.makedirs(output_dir, exist_ok=True)    
    for name, sequence in tqdm(zip(names, sequences), total=len(names)):        
        contacts = get_esm_contacts(ptdir, name, key=key)  # [L, L]        
        
        L = len(sequence)
        if contacts.shape[0] != L:
            contacts = contacts[:L, :L]          
        
        mask1 = np.tril(np.ones((L, L)), -3)  
        mask2 = np.triu(np.ones((L, L)), 3)   
        mask = mask1 + mask2
        contacts = contacts * mask  
        contacts = contacts[0]
        # print(name,contacts.shape)        
        
        np.save(os.path.join(output_dir, f'{name}.npy'), contacts)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../TmpredTrain.csv",
                    help='Please input the path of CSV (default：../TmpredTrain.csv)')
    args = parser.parse_args()
    ptdir = "../pt/"    
    output_dir = '../Data/edge_features_Tm/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    key = 33
    df = pd.read_csv(args.input) 
    names = df["uniprot_id"]
    sequences = df["sequence"]   
    save_esm_contacts_as_npy(
        names=names,
        sequences=sequences,
        ptdir=ptdir,
        output_dir=output_dir,
        key=key
    )