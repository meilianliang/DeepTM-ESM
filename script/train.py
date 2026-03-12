import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics
from scipy.stats import pearsonr
from config import *
from data import ProDataset
from collections import Counter
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import sys
np.set_printoptions(threshold=sys.maxsize)
from model import Model

def train_one_epoch(model, data_loader, epoch):

    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        _, _, labels, sequence_features, sequence_graphs = data

        # sequence_features = torch.squeeze(sequence_features)
        # sequence_graphs = torch.squeeze(sequence_graphs)

        if torch.cuda.is_available():
            features = Variable(sequence_features.cuda())
            graphs = Variable(sequence_graphs.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_features)
            graphs = Variable(sequence_graphs)
            y_true = Variable(labels)
        y_pred = model(features, graphs)
        y_pred = torch.squeeze(y_pred)
        y_true = y_true.float()/120.0
        # calculate loss
        if(len(y_pred.size())==0):
            y_pred = y_pred.unsqueeze(0)
        loss = model.criterion(y_pred, y_true)
        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg

def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, sequence_features, sequence_graphs = data

            # sequence_features = torch.squeeze(sequence_features)
            # sequence_graphs = torch.squeeze(sequence_graphs)

            if torch.cuda.is_available():
                features = Variable(sequence_features.cuda())
                graphs = Variable(sequence_graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_features)
                graphs = Variable(sequence_graphs)
                y_true = Variable(labels)
            y_pred = model(features, graphs)
            y_pred = torch.squeeze(y_pred)
            y_true = y_true.float()/120.0
            if(len(y_pred.size())==0):
                y_pred = y_pred.unsqueeze(0)
            loss = model.criterion(y_pred, y_true)
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            flag = isinstance(y_pred,float)
            if(flag):
                a = []
                a.append(y_pred)
                y_pred = a
            valid_pred.extend(y_pred)
            valid_true.extend(y_true)
            valid_name.extend(sequence_names)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches

    return epoch_loss_avg, valid_true, valid_pred, valid_name


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    train_losses = []
    train_pearson = []
    train_r2 = []
    
    valid_losses = []
    valid_pearson = []
    valid_r2 = []

    best_val_loss = 1000
    best_train_loss = 1000
    best_epoch = 0
    best_epoch_train = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, train_name = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", np.sqrt(epoch_loss_train_avg))
        print("Train pearson:", result_train['pearson'])
        print("Train r2:", result_train['r2'])

        train_losses.append(np.sqrt(epoch_loss_train_avg))
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        
        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", np.sqrt(epoch_loss_valid_avg))
        print("Valid pearson:", result_valid['pearson'])
        print("Valid r2:", result_valid['r2'])
        
        valid_losses.append(np.sqrt(epoch_loss_valid_avg))
        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])

        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            valid_true_T=list(np.array(valid_true) * 120)
            valid_pred_T=list(np.array(valid_pred) * 120)

            valid_detail_dataframe = pd.DataFrame({'uniprot_id': valid_name, 'y_true': valid_true, 'y_pred': valid_pred, 'Tm':valid_true_T, 'prediction':valid_pred_T})
            valid_detail_dataframe.sort_values(by=['y_true'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
            train_true_T=list(np.array(train_true) * 120)
            train_pred_T=list(np.array(train_pred) * 120)

            train_detail_dataframe = pd.DataFrame({'uniprot_id': train_name, 'y_true': train_true, 'y_pred': train_pred, 'Tm':train_true_T, 'prediction':train_pred_T})
            train_detail_dataframe.sort_values(by=['y_true'], inplace=True)
            train_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_train_detail.csv", header=True, sep=',')

    # save calculation information
    result_all = {
        'Train_loss': train_losses,
        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        
        'Valid_loss': valid_losses,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        
        'Best_epoch': [best_epoch for _ in range(len(train_losses))]
    }
    result = pd.DataFrame(result_all)
    print("Fold", str(fold), "Best epoch at", str(best_epoch))
    result.to_csv(Result_Path + "Fold" + str(fold) + "_result.csv", sep=',')

def analysis(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 120 
    result = {
        'pearson': pearson,
        'r2': r2,
        'rmse': rmse,
    }
    return result

def cross_validation(all_dataframe,fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['uniprot_id'].values
    sequence_labels = all_dataframe['tm'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model()
        # if torch.cuda.is_available():
        #     model.cuda()
        if torch.cuda.device_count()>1:
            print(f"use {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model,device_ids=device_ids)
        model.to(device)

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1

def train_full(model, train_dataframe, num_epochs, batch_size, model_save_path, result_save_path):
    train_loader = DataLoader(
        dataset=ProDataset(train_dataframe), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    train_losses = []       
    train_pearson = []      
    train_r2 = []           
    
    best_train_loss = float('inf')  
    best_epoch = 0                  

    for epoch in range(num_epochs):
        print(f"\n========== Training Epoch {epoch + 1}/{num_epochs} ==========")
        
        
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        
        
        print("========== Evaluating on Training Set ==========")
        _, train_true, train_pred, train_name = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        
        
        train_rmse = np.sqrt(epoch_loss_train_avg)
        train_losses.append(train_rmse)
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train Pearson: {result_train['pearson'][0]:.4f}")
        print(f"Train R²: {result_train['r2']:.4f}")
        
        
        if epoch_loss_train_avg < best_train_loss:
            best_train_loss = epoch_loss_train_avg
            best_epoch = epoch + 1
            
            
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pkl'))
            print(f"Saved new best model at epoch {best_epoch} with train loss {train_rmse:.4f}")
            
            
            train_true_T = [t * 120 for t in train_true]  
            train_pred_T = [p * 120 for p in train_pred]  
            
            train_detail_df = pd.DataFrame({
                'uniprot_id': train_name,
                'y_true': train_true,
                'y_pred': train_pred,
                'Tm': train_true_T,
                'prediction': train_pred_T
            })
            train_detail_df.sort_values(by=['y_true'], inplace=True)
            train_detail_df.to_csv(os.path.join(result_save_path, 'best_train_detail.csv'), index=False)
    
    
    result_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'train_pearson': [p[0] for p in train_pearson],  # 提取Pearson值
        'train_r2': train_r2,
        'best_epoch': [best_epoch] * num_epochs
    })
    result_df.to_csv(os.path.join(result_save_path, 'training_results.csv'), index=False)
    
    print(f"\nTraining completed! Best model saved at epoch {best_epoch} with loss {np.sqrt(best_train_loss):.4f}")


def train_with_validation(model, train_dataframe, val_dataframe, num_epochs, batch_size, model_save_path, result_save_path):
    
    train_loader = DataLoader(
        dataset=ProDataset(train_dataframe), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=ProDataset(val_dataframe), 
        batch_size=batch_size, 
        shuffle=False,  
        num_workers=2,
        drop_last=False  
    )
    
    train_losses = []       
    train_pearson = []      
    train_r2 = []           
    
    val_losses = []         
    val_pearson = []        
    val_r2 = []             
    
    best_val_loss = float('inf')  
    best_epoch = 0                

    for epoch in range(num_epochs):
        print(f"\n========== Training Epoch {epoch + 1}/{num_epochs} ==========")
        
        
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        
        
        print("========== Evaluating on Training Set ==========")
        _, train_true, train_pred, train_name = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        
        
        train_rmse = np.sqrt(epoch_loss_train_avg)
        train_losses.append(train_rmse)
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train Pearson: {result_train['pearson'][0]:.4f}")
        print(f"Train R²: {result_train['r2']:.4f}")
        
        
        print("========== Evaluating on Validation Set ==========")
        model.eval()
        with torch.no_grad():
            val_loss, val_true, val_pred, val_name = evaluate(model, val_loader)
            result_val = analysis(val_true, val_pred)
        
        
        val_rmse = np.sqrt(val_loss)
        val_losses.append(val_rmse)
        val_pearson.append(result_val['pearson'])
        val_r2.append(result_val['r2'])
        
        print(f"Val RMSE: {val_rmse:.4f}")
        print(f"Val Pearson: {result_val['pearson'][0]:.4f}")
        print(f"Val R²: {result_val['r2']:.4f}")
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pkl'))
            print(f"Saved new best model at epoch {best_epoch} with val loss {val_rmse:.4f}")
            
            
            train_true_T = [t * 120 for t in train_true]  
            train_pred_T = [p * 120 for p in train_pred]  
            
            train_detail_df = pd.DataFrame({
                'uniprot_id': train_name,
                'y_true': train_true,
                'y_pred': train_pred,
                'Tm': train_true_T,
                'prediction': train_pred_T
            })
            train_detail_df.sort_values(by=['y_true'], inplace=True)
            train_detail_df.to_csv(os.path.join(result_save_path, 'best_train_detail.csv'), index=False)
            
            
            val_true_T = [t * 120 for t in val_true]  
            val_pred_T = [p * 120 for p in val_pred]  
            
            val_detail_df = pd.DataFrame({
                'uniprot_id': val_name,
                'y_true': val_true,
                'y_pred': val_pred,
                'Tm': val_true_T,
                'prediction': val_pred_T
            })
            val_detail_df.sort_values(by=['y_true'], inplace=True)
            val_detail_df.to_csv(os.path.join(result_save_path, 'best_val_detail.csv'), index=False)
    
    
    result_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'train_pearson': [p[0] for p in train_pearson],  
        'train_r2': train_r2,
        'val_loss': val_losses,
        'val_pearson': [p[0] for p in val_pearson],      
        'val_r2': val_r2,
        'best_epoch': [best_epoch] * num_epochs
    })
    result_df.to_csv(os.path.join(result_save_path, 'training_results.csv'), index=False)
    
    print(f"\nTraining completed! Best model saved at epoch {best_epoch} with val loss {np.sqrt(best_val_loss):.4f}")



def train_gridsearch(model, train_dataframe, val_dataframe, num_epochs, batch_size, model_save_path, result_save_path, param_suffix=""):
    
    train_loader = DataLoader(
        dataset=ProDataset(train_dataframe), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=ProDataset(val_dataframe), 
        batch_size=batch_size, 
        shuffle=False,  
        num_workers=2,
        drop_last=False  
    )
    
    train_losses = []      
    train_pearson = []     
    train_r2 = []          
    
    val_losses = []        
    val_pearson = []       
    val_r2 = []            
    
    best_val_loss = float('inf')  
    best_epoch = 0                

    for epoch in range(num_epochs):
        print(f"\n========== Training Epoch {epoch + 1}/{num_epochs} ==========")
        
        
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        
        
        print("========== Evaluating on Training Set ==========")
        _, train_true, train_pred, train_name = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        
        
        train_rmse = np.sqrt(epoch_loss_train_avg)
        train_losses.append(train_rmse)
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train Pearson: {result_train['pearson'][0]:.4f}")
        print(f"Train R²: {result_train['r2']:.4f}")
        
        
        print("========== Evaluating on Validation Set ==========")
        model.eval()
        with torch.no_grad():
            val_loss, val_true, val_pred, val_name = evaluate(model, val_loader)
            result_val = analysis(val_true, val_pred)
        
        
        val_rmse = np.sqrt(val_loss)
        val_losses.append(val_rmse)
        val_pearson.append(result_val['pearson'])
        val_r2.append(result_val['r2'])
        
        print(f"Val RMSE: {val_rmse:.4f}")
        print(f"Val Pearson: {result_val['pearson'][0]:.4f}")
        print(f"Val R²: {result_val['r2']:.4f}")
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            
            model_filename = f"best_model{param_suffix}.pkl"
            torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))
            print(f"Saved new best model at epoch {best_epoch} with val loss {val_rmse:.4f}")
            
            
            train_true_T = [t * 120 for t in train_true]  
            train_pred_T = [p * 120 for p in train_pred]  
            
            train_detail_filename = f"best_train_detail{param_suffix}.csv"
            train_detail_df = pd.DataFrame({
                'uniprot_id': train_name,
                'y_true': train_true,
                'y_pred': train_pred,
                'Tm': train_true_T,
                'prediction': train_pred_T
            })
            train_detail_df.sort_values(by=['y_true'], inplace=True)
            train_detail_df.to_csv(os.path.join(result_save_path, train_detail_filename), index=False)
            
            
            val_true_T = [t * 120 for t in val_true]  
            val_pred_T = [p * 120 for p in val_pred]  
            
            val_detail_filename = f"best_val_detail{param_suffix}.csv"
            val_detail_df = pd.DataFrame({
                'uniprot_id': val_name,
                'y_true': val_true,
                'y_pred': val_pred,
                'Tm': val_true_T,
                'prediction': val_pred_T
            })
            val_detail_df.sort_values(by=['y_true'], inplace=True)
            val_detail_df.to_csv(os.path.join(result_save_path, val_detail_filename), index=False)
            
             
    result_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'train_pearson': [p[0] for p in train_pearson],  # 提取Pearson值
        'train_r2': train_r2,
        'val_loss': val_losses,
        'val_pearson': [p[0] for p in val_pearson],      # 提取Pearson值
        'val_r2': val_r2,
        'best_epoch': [best_epoch] * num_epochs
    })
    result_df.to_csv(os.path.join(result_save_path, f'training_results{param_suffix}.csv'), index=False)

    print(f"\nTraining completed! Best model saved at epoch {best_epoch} with val loss {np.sqrt(best_val_loss):.4f}")
