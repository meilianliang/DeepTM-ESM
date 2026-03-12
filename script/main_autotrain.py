import os
import torch
import numpy as np
import pandas as pd
from config import *
from train import train_gridsearch
from model import Model
from sklearn.model_selection import train_test_split
import argparse


learning_rates = [4E-4]
weight_decays = [5E-5]

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../TmpredTrain.csv",
                    help='Please input the path of CSV (default：../TmpredTrain.csv)')
    args = parser.parse_args()  
    full_train_df = pd.read_csv(args.input)
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)   

    for lr in learning_rates:
        for wd in weight_decays:            
            LEARNING_RATE = lr
            WEIGHT_DECAY = wd
            
            model = Model()
            if torch.cuda.is_available():
                model = model.cuda()
            
            param_suffix = f"_lr{lr}_wd{wd}_{GCN_HIDDEN_DIM}_{GCN_OUTPUT_DIM}"
                       
            print(f"\nTraining with LEARNING_RATE={lr}, WEIGHT_DECAY={wd}")
            train_gridsearch(
                model=model,
                train_dataframe=train_df,
                val_dataframe=val_df,
                num_epochs=NUMBER_EPOCHS,
                batch_size=BATCH_SIZE,
                model_save_path=Model_Path,
                result_save_path=Result_Path,
                param_suffix=param_suffix  
            )
                        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()