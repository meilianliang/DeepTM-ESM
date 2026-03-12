import pandas as pd
from config import *
from test import test
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../TmpredTest.csv",
                    help='Please input the path of CSV (default：../TmpredTest.csv)')
    args = parser.parse_args()  

    test_dataframe = pd.read_csv(args.input, sep=',')
    test(test_dataframe) 