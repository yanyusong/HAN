from keras import regularizers, optimizers

# workspace
workspace_path = '/Users/mac/2019/MyBag19/细粒度情感分析/workspace'
# source data
train_file = './sentiment_analysis_trainingset.csv'
validate_file = './sentiment_analysis_validationset.csv'
# cache
train_processed_file = './X_train_processed_file.txt'
tokenizer_file = './X_train_tokenizer_file.txt'
val_processed_file = './X_val_processed_file.txt'
# log
log_dir = './log'


class Model_Config():
    embedding_dim = 100  # 词向量的维度为100
    rnn_unit_num = 400  # rnn cell 的隐藏单元数量，也即：output_size/state_size
    fc_num = 200  # 最后 softmax 之前全连接的维度
    td_fc_unit_num = 300  # 双向 GRU 后 与 Attention 之间的 TimeDistributed fc
    td_doc_fc_unit_num = 400  #
    epochs = 3  # epoch
    batch_size = 16  # batch_size
    doc_max_sentence_num = 35  # 文档中句子最多的数量
    sentence_max_word_num = 85  # 句子中最大的词数量,sequence_length
    class_num = 4  # 类别数量
    drop_rate = 0.5  # drop_out 层的 drop 比例
    lr = 0.001
    optimizer = optimizers.Adam(lr=lr)

    REG_PARAM = 1e-13
    l2_reg = regularizers.l2(REG_PARAM)

    current_trained_labels = ['location_traffic_convenience', 'location_distance_from_business_district',
                              'location_easy_to_find']

    mutil_layers_num = len(current_trained_labels)
    # model
    model_saved_filepath = './hdf/location_model.hdf5'

    model_checkpoint_saved_filepath = './hdf/location_checkpoint-{epoch:02d}e-val_loss_{val_loss:.2f}.hdf5'
