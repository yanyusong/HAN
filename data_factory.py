import pandas as pd
import matplotlib.pyplot as plt
from utils import *


class Data_Factory():
    def __init__(self, train_file, validate_file, sep=','):
        self.train_df = pd.read_csv(train_file, sep=sep, encoding='utf-8', error_bad_lines=False, dtype=np.str)[:]
        self.validate_df = pd.read_csv(validate_file, sep=sep, encoding='utf-8', error_bad_lines=False, dtype=np.str)[:]
        self.label_columns = [
            'location_traffic_convenience',
            'location_distance_from_business_district',
            'location_easy_to_find',
            'service_wait_time',
            'service_waiters_attitude',
            'service_parking_convenience',
            'service_serving_speed',
            'price_level',
            'price_cost_effective',
            'price_discount',
            'environment_decoration',
            'environment_noise',
            'environment_space',
            'environment_cleaness',
            'dish_portion',
            'dish_taste',
            'dish_look',
            'dish_recommendation',
            'others_overall_experience',
            'others_willing_to_consume_again']
        self.label_distribute_dict_list = self.label_distribute()

    def data(self, label_list):
        label_map = {'-2': 0, '-1': 1, '0': 2, '1': 3}
        X_train = self.train_df['content'].values
        X_val = self.validate_df['content'].values

        Y_train_list = []
        Y_val_list = []

        for current_train_label in label_list:
            Y_train = self.train_df[current_train_label].map(label_map).values
            Y_train = keras.utils.to_categorical(Y_train, num_classes=4)
            Y_train_list.append(Y_train)

            Y_val = self.validate_df[current_train_label].map(label_map).values
            Y_val = keras.utils.to_categorical(Y_val, num_classes=4)
            Y_val_list.append(Y_val)
        return X_train, X_val, Y_train_list, Y_val_list, self.label_distribute_dict_list

    def label_distribute(self):
        dict_list = {}
        for label in self.label_columns[:]:
            test_df = pd.DataFrame(self.train_df[[label, 'id']].groupby(label).size(), columns=['count'])
            test_df['rate'] = test_df['count'].apply(lambda x: int(100 * round(int(x) / 105000, 2)))
            test_df.reset_index(inplace=True)
            for i in test_df.values:
                if label not in dict_list:
                    dict_list[label] = {}
                dict_list[label][int(i[0])+2] = i[2]
        return dict_list


if __name__ == '__main__':
    from config import *

    data_factory = Data_Factory(train_file, validate_file)
    # shape
    print(data_factory.train_df.shape)
    print(data_factory.validate_df.shape)
    # 查看各label分布
    dict_list = {}
    for label in data_factory.label_columns[:]:
        test_df = pd.DataFrame(data_factory.train_df[[label, 'id']].groupby(label).size(), columns=['count'])
        test_df['rate'] = test_df['count'].apply(lambda x: int(100 * round(int(x) / 105000, 2)))
        test_df.reset_index(inplace=True)
        for i in test_df.values:
            if label not in dict_list:
                dict_list[label] = {}
            dict_list[label][int(i[0])] = i[2]
        print(test_df)
    # sample
    data_factory.train_df['content'].head(2)
    # 查看 doc 字数分布
    docs = data_factory.train_df['content'][:]
    doc_lens = [len(doc) for doc in docs]
    n, bins, patches = plt.hist(x=doc_lens, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    # 查看 doc 中句子数量分布
    docs_sentence_list = [cut_doc_2_sentences(doc, ['.', '!', '?', ';', '。', '！', '？', '；'], 10, 80) for doc in docs[:]]
    # docs_sentence_list[:3]
    sentence_lens = [len(sentence) for sentence in docs_sentence_list]
    n, bins, patches = plt.hist(x=sentence_lens, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    # 查看句子长度分布
    sentence_lens = [[len(seq) for seq in sentence] for sentence in docs_sentence_list]
    seq_lens = []
    for i in sentence_lens:
        seq_lens.extend(i)
    n, bins, patches = plt.hist(x=seq_lens, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    # 句子中词数分布
    sentence_lens = [[len(jieba.lcut(sentence)) for sentence in sentence_list] for sentence_list in
                     docs_sentence_list[:]]
    seq_lens = []
    for i in sentence_lens:
        seq_lens.extend(i)
    n, bins, patches = plt.hist(x=seq_lens, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

''' 各label分布
                                  count  rate
location_traffic_convenience             
-1                             1318     1
-2                            81382    78
0                              1046     1
1                             21254    20
                                          count  rate
location_distance_from_business_district             
-1                                          586     1
-2                                        83680    80
0                                           533     1
1                                         20201    19
                       count  rate
location_easy_to_find             
-1                      3976     4
-2                     80605    77
0                       2472     2
1                      17947    17
                   count  rate
service_wait_time             
-1                  3034     3
-2                 92763    88
0                   4382     4
1                   4821     5
                          count  rate
service_waiters_attitude             
-1                         8684     8
-2                        42410    40
0                         12534    12
1                         41372    39
                             count  rate
service_parking_convenience             
-1                            1323     1
-2                           98276    94
0                             1456     1
1                             3945     4
                       count  rate
service_serving_speed             
-1                      5487     5
-2                     88700    84
0                       2379     2
1                       8434     8
             count  rate
price_level             
-1           12375    12
-2           52820    50
0            24249    23
1            15556    15
                      count  rate
price_cost_effective             
-1                     3011     3
-2                    80242    76
0                      3072     3
1                     18675    18
                count  rate
price_discount             
-1               1716     2
-2              64243    61
0               18255    17
1               20786    20
                        count  rate
environment_decoration             
-1                       2139     2
-2                      53916    51
0                        9492     9
1                       39453    38
                   count  rate
environment_noise             
-1                  3077     3
-2                 73445    70
0                   4843     5
1                  23635    23
                   count  rate
environment_space             
-1                  5706     5
-2                 65398    62
0                   9262     9
1                  24634    23
                      count  rate
environment_cleaness             
-1                     4513     4
-2                    66598    63
0                      4703     4
1                     29186    28
              count  rate
dish_portion             
-1            10018    10
-2            56917    54
0              9506     9
1             28559    27
            count  rate
dish_taste             
-1           4363     4
-2           5070     5
0           40200    38
1           55367    53
           count  rate
dish_look             
-1          3178     3
-2         75975    72
0           4675     4
1          21172    20
                     count  rate
dish_recommendation             
-1                    2275     2
-2                   84767    81
0                     1988     2
1                    15970    15
                           count  rate
others_overall_experience             
-1                          9384     9
-2                          2110     2
0                          23436    22
1                          70070    67
                                 count  rate
others_willing_to_consume_again             
-1                                4159     4
-2                               65600    62
0                                 2913     3
1                                32328    31
    '''
