import pandas as pd
from config import *
from train import cross_validation
from model import Model
import argparse

if __name__ == "__main__":
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../TmpredTrain.csv",
                    help='Please input the path of CSV (default：../TmpredTrain.csv)')
    args = parser.parse_args()  
    train_dataframe = pd.read_csv(args.input,sep=',')
    cross_validation(train_dataframe,fold_number=5)