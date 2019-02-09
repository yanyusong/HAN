import os
from config import workspace_path
from train import train
from config import Model_Config
from prepare import prepare
from predict import predict
from evaluate import evaluate

print(os.getcwd())
os.chdir(workspace_path)
os.mkdir('log')
os.mkdir('hdf')
print(os.getcwd())
print(os.listdir('./'))

if __name__ == '__main__':
    # prepare
    X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size, label_distribute_dict_list = prepare()
    # train
    model, history = train(X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size,
                           label_distribute_dict_list)
    # predict
    preds = predict(X_val_processed, Model_Config.model_saved_filepath)
    # evaluate
    validate_df, df_pcf = evaluate(preds)
    # print precision、recall、f1 值
    print(df_pcf)
