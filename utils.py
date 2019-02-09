import os
import time as time
import jieba
import keras
import numpy as np
import pickle


def cut_doc_2_sentences(doc, sentence_flags=None, skip_limit=5, long_cut_limit=130,
                        all_flags=[',', '.', '!', '?', ';', '~', '，', '。', '！', '？', '；', '～', '\n', ' '],
                        strip_flags=None):
    if strip_flags is None:
        strip_flags = [' ']
    if sentence_flags is None:
        sentence_flags = all_flags
    last_flag = 0
    sentence_list = []
    doc_length = len(doc)
    for i in range(doc_length):
        cut_flags = sentence_flags
        if i + 1 - last_flag > long_cut_limit:
            cut_flags = all_flags
        if (i <= doc_length - 2 and doc[i] in cut_flags and doc[
            i + 1] not in cut_flags) or i == doc_length - 1:
            temp = doc[last_flag:i + 1]
            chars_no_flags = [char for char in temp if char not in cut_flags]
            if len(chars_no_flags) < skip_limit:
                # 句子内非标点句长小于阀值 skip_limit 的并入下一个分句
                continue
            # 分完句以后去掉前后无用的字符
            for flag in strip_flags:
                temp = temp.strip(flag)
            sentence_list.append(temp)
            last_flag = i + 1
    return sentence_list


def cut_docs(docs):
    start_time = time.time()
    print('start 分句...')
    docs_sentence_list = [cut_doc_2_sentences(doc) for doc in docs]
    print('end 分句,Total docs = {},Cost time = {}'.format(len(docs), time.time() - start_time))
    start_time = time.time()
    print('start 分词...')
    docs_cut = [[jieba.lcut(sentence) for sentence in sentence_list] for sentence_list in docs_sentence_list]
    print('end 分词, Cost time = {}'.format(time.time() - start_time))
    return docs_cut


# 根据训练集生成 vocabulary，返回 fit 后的 tokenizer
def build_vocabulary_tokenizer(docs_cut):
    vocabulary = []
    for doc_sentence_list in docs_cut:
        for sentence_list in doc_sentence_list:
            for word in sentence_list:
                vocabulary.append(word)
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([vocabulary])
    return tokenizer


# 根据fit后的tokenizer，将分词分句后的doc中的词替换成index
def index_docs_func(tokenizer, docs_cut):
    index_docs = []
    for doc_sentence_list in docs_cut:
        index_docs.append(tokenizer.texts_to_sequences(doc_sentence_list))
    return index_docs


def pad_docs(index_docs, doc_max_sentence_num, sentence_max_word_num, padding_value=0):
    data = []
    for doc in index_docs:
        doc_data = []
        for sentence in doc:
            # 句子 word 数补齐成 sentence_max_word_num
            if len(sentence) < sentence_max_word_num:
                sentence.extend([padding_value] * (sentence_max_word_num - len(sentence)))
            doc_data.append(sentence[:sentence_max_word_num])
        # 每篇文章句子数补够 doc_max_sentence_num
        if len(doc_data) < doc_max_sentence_num:
            doc_data.extend([[padding_value] * sentence_max_word_num] * (doc_max_sentence_num - len(doc_data)))
        data.append(doc_data[:doc_max_sentence_num])
    data = np.array(data)
    return data


def dump_data(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data
