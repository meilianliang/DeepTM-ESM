import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
from data import ProDataset
from model import Model

def predict(model, data_loader):
    model.eval()
    all_names = []
    all_preds = []

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, _, sequence_features, sequence_graphs = data

            if torch.cuda.is_available():
                features = Variable(sequence_features.cuda())
                graphs = Variable(sequence_graphs.cuda())
            else:
                features = Variable(sequence_features)
                graphs = Variable(sequence_graphs)

            y_pred = model(features, graphs)
            y_pred = torch.squeeze(y_pred)
            
            if y_pred.dim() == 0:
                y_pred = y_pred.unsqueeze(0)

            preds = y_pred.cpu().numpy().tolist()
            if isinstance(preds, float):
                preds = [preds]

            all_names.extend(sequence_names)
            all_preds.extend(preds)

    return all_names, all_preds

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model (no-OGT version)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file (must contain columns: uniprot_id, sequence)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pkl)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to output CSV file (default: predictions.csv)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = Model()
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    print(f"Reading input CSV: {args.input}")
    df = pd.read_csv(args.input)

    required_cols = ['uniprot_id', 'sequence']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Input CSV must contain column '{col}'")
            sys.exit(1)

    df['tm'] = 0.0

    print("Creating data loader...")
    dataset = ProDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    print("Running inference...")
    names, preds = predict(model, loader)
    
    preds_tm = [p * 120 for p in preds]
    
    result_df = pd.DataFrame({
        'uniprot_id': names,
        'predicted_Tm': preds_tm
    })
    result_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()