from utils import *
from config import Model_Config


# 预处理 训练集
def pre_process_train_docs(docs, doc_max_sentence_num, sentence_max_word_num):
    docs_cut = cut_docs(docs)  # 分词分句
    start_time = time.time()
    print('start build_vocabulary_tokenizer...')
    tokenizer = build_vocabulary_tokenizer(docs_cut)
    print('end build_vocabulary_tokenizer, Cost time = {}'.format(time.time() - start_time))
    index_docs = index_docs_func(tokenizer, docs_cut)
    data = pad_docs(index_docs, doc_max_sentence_num, sentence_max_word_num)
    vocabulary_size = len(tokenizer.word_index.values()) + 1
    return data, vocabulary_size, tokenizer


# 预处理 验证集
def pre_process_val_docs(tokenizer, docs, doc_max_sentence_num, sentence_max_word_num):
    docs_cut = cut_docs(docs)  # 分词分句
    index_docs = index_docs_func(tokenizer, docs_cut)
    data = pad_docs(index_docs, doc_max_sentence_num, sentence_max_word_num)
    return data


# 如果固化的pickle有则不重新建
# rebuild = False 有则不重建，True 不管有没有每次都重建
def process_train_val_data(X_train, X_val, X_train_processed_file, X_val_processed_file, X_train_tokenizer_file,
                           doc_max_sentence_num, sentence_max_word_num,
                           rebuild=False):
    if not os.path.exists(X_train_processed_file) or not os.path.exists(X_val_processed_file) or not os.path.exists(
            X_train_tokenizer_file) or rebuild:
        start_time = time.time()
        print('start pre_process_train_docs...')
        X_train_processed, vocabulary_size, tokenizer = pre_process_train_docs(X_train,
                                                                               doc_max_sentence_num,
                                                                               sentence_max_word_num)
        dump_data(X_train_processed, X_train_processed_file)
        dump_data(tokenizer, X_train_tokenizer_file)
        print('vocabulary_size = {}'.format(vocabulary_size))
        print('end pre_process_train_docs, Total cost time = {}'.format(time.time() - start_time))
        start_time = time.time()
        print('start pre_process_val_docs...')
        X_val_processed = pre_process_val_docs(tokenizer, X_val,
                                               doc_max_sentence_num,
                                               sentence_max_word_num)
        dump_data(X_val_processed, X_val_processed_file)
        print('end pre_process_val_docs, Total cost time = {}'.format(time.time() - start_time))
    else:
        X_train_processed = load_data(X_train_processed_file)
        tokenizer = load_data(X_train_tokenizer_file)
        X_val_processed = load_data(X_val_processed_file)
        vocabulary_size = len(tokenizer.word_index.values()) + 1

    return X_train_processed, X_val_processed, vocabulary_size


def prepare():
    from config import train_file, validate_file, train_processed_file, val_processed_file, tokenizer_file, Model_Config
    from data_factory import Data_Factory

    data_factory = Data_Factory(train_file, validate_file)
    X_train, X_val, Y_train_list, Y_val_list, label_distribute_dict_list = data_factory.data(
        Model_Config.current_trained_labels)

    X_train_processed, X_val_processed, vocabulary_size = process_train_val_data(X_train, X_val, train_processed_file,
                                                                                 val_processed_file, tokenizer_file,
                                                                                 Model_Config.doc_max_sentence_num,
                                                                                 Model_Config.sentence_max_word_num,
                                                                                 rebuild=False)
    return X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size,label_distribute_dict_list
