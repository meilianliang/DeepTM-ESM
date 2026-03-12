import pandas as pd
from config import *
from train import train_full
from model import Model
import argparse


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../TmpredTrain.csv",
                    help='Please input the path of CSV (default：../TmpredTrain.csv)')
    args = parser.parse_args()  
    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()

    full_train_df = pd.read_csv(args.input)

    train_full(
        model=model,
        train_dataframe=full_train_df,
        num_epochs=NUMBER_EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=Model_Path,
        result_save_path=Result_Path
    )       