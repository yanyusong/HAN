from config import Model_Config, train_file, validate_file
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from data_factory import Data_Factory


def evaluate(preds):
    validate_df = Data_Factory(train_file, validate_file).validate_df.copy()
    for label, pred in zip(Model_Config.current_trained_labels, preds):
        validate_df['pre_{}'.format(label)] = pred

    df_pcf = compute_pcf(validate_df, Model_Config.current_trained_labels)
    return validate_df, df_pcf


def compute_pcf(df, current_trained_labels):
    df_result = pd.DataFrame([])
    total_num = df.shape[0]
    for label in current_trained_labels:
        t_num = df.apply(lambda row: int(row[label]) + 2 == row['pre_' + label], axis=1)
        tp = len([i for i in t_num.values.tolist() if i]) / total_num
        df_result.loc[label, 'precision'] = tp

        df[label] = df[label].apply(lambda x: int(x) + 2)
        # sk
        df_result.loc['sk_' + label, 'precision'] = precision_score(df[label].values, df['pre_' + label].values,
                                                                    average='macro')
        df_result.loc['sk_' + label, 'recall'] = recall_score(df[label].values, df['pre_' + label].values,
                                                              average='macro')
        df_result.loc['sk_' + label, 'f1'] = f1_score(df[label].values, df['pre_' + label].values, average='macro')

    return df_result

# def evaluate(X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size,
#              label_distribute_dict_list, ):
#
#     score = model.evaluate(x=X_val_processed,
#                            y={'small_categorical_1_preds': Y_val_list[0],
#                               'small_categorical_2_preds': Y_val_list[1],
#                               'small_categorical_3_preds': Y_val_list[2]},
#                            verbose=1)
#     print(score)
#     return model, history
