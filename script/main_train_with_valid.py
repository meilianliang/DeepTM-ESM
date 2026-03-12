import pandas as pd
from config import *
from train import train_with_validation
from model import Model
from sklearn.model_selection import train_test_split
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

    # train_df = pd.read_csv(TrainData_Path)
    # val_df = pd.read_csv(ValidData_Path)
    full_train_df = pd.read_csv(args.input)
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)

    train_with_validation(
        model=model,
        train_dataframe=train_df,
        val_dataframe=val_df,
        num_epochs=NUMBER_EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=Model_Path,
        result_save_path=Result_Path
    )       