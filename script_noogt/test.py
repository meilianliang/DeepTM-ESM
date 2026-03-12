from train import *
def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        state_dict = model.state_dict()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        epoch_loss_test_avg, test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        print("\n========== Evaluate Test set ==========")
        print("Test loss: ", np.sqrt(epoch_loss_test_avg))
        print("Test pearson:", result_test['pearson'])
        print("Test r2:", result_test['r2'])
        print("Test rmse:", result_test['rmse'])

        test_result[model_name] = [
            np.sqrt(epoch_loss_test_avg),
            result_test['pearson'],
            result_test['r2'],
            result_test['rmse']
        ]
        test_true_T=list(np.array(test_true) * 120)
        test_pred_T=list(np.array(test_pred) * 120)
        test_detail_dataframe = pd.DataFrame({'uniprot_id': test_name, 'y_true': test_true, 'y_pred': test_pred,'Tm':test_true_T,'prediction':test_pred_T})
        test_detail_dataframe.sort_values(by=['uniprot_id'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=['loss', 'pearson', 'r2', 'rmse'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')