import pickle
import numpy as np
from gensim.models import KeyedVectors


# 将词向量文件保存为二进制文件
def trans_bin(path1, path2):
    """
    将Word2Vec格式的词向量文件转换为二进制格式并保存。

    参数：
    path1 -- 源Word2Vec格式的词向量文件路径
    path2 -- 目的二进制格式的词向量文件路径
    """
    # 加载非二进制格式的词向量文件
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)

    # 初始化相似性矩阵，提高后续加载速度
    wv_from_text.init_sims(replace=True)

    # 保存为二进制格式
    wv_from_text.save(path2)


# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    """
    构建新的词典和词向量矩阵，并保存为二进制文件。

    参数：
    type_vec_path -- 已转换的二进制词向量文件路径
    type_word_path -- 输入的词汇表文件路径
    final_vec_path -- 输出的词向量矩阵保存路径
    final_word_path -- 输出的词典保存路径
    """
    # 加载二进制格式的词向量文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    # 加载词汇表
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())

    # 初始化输出词典和词向量列表
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 特殊标记：0-PAD, 1-SOS, 2-EOS, 3-UNK

    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]

    for word in total_word:
        try:
            # 加载词向量
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except KeyError:
            fail_word.append(word)

    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))

    # 保存词向量矩阵
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    # 保存词典
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    """
    获取文本中每个词在词典中的位置索引。

    参数：
    type -- 文本类型（'code' or 'text'）
    text -- 输入文本列表
    word_dict -- 字典映射

    返回：
    返回词在词典中的位置索引列表
    """
    location = []

    if type == 'code':
        # 代码类型的文本处理
        location.append(1)  # 添加SOS标记
        len_c = len(text)

        if len_c + 1 < 350:
            # 短文本处理
            if len_c == 1 and text[0] == '-1000':
                location.append(2)  # 空白结束标记
            else:
                for word in text:
                    index = word_dict.get(word, word_dict['UNK'])
                    location.append(index)
                location.append(2)  # 添加EOS标记
        else:
            # 长文本处理，截取前348个词
            for word in text[:348]:
                index = word_dict.get(word, word_dict['UNK'])
                location.append(index)
            location.append(2)  # 添加 普通文本处理
        if len(text) == 0 or text[0] == '-10000':
            location.append(0)  # 空文本 in text:
            index = word_dict.get(word, word_dict['UNK'])
            location.append(index)
    return location


# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    """
    将训练、测试、验证语料序列化

    参数：
    word_dict_path -- 字典文件路径
    type_path -- 原始语料文件路径
    final_type_path -- 序列化后的语料保存路径
    """
    # 加载字典
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    # 加载语料
    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for instance in corpus:
        qid = instance[0]

        # 获取索引列表
        Si_word_list = get_index('text', instance[1][0], word_dict)
        Si1_word_list = get_index('text', instance[1][1], word_dict)
        tokenized_code = get_index('code', instance[2][0], word_dict)
        query_word_list = get_index('text', instance[3], word_dict)

        block_length = 4
        label = 0

        # 调整长度，少于目标长度用0填充，多于目标长度截断
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (
                    100 - len(Si1_word_list))
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (
                    25 - len(query_word_list))

        # 组织数据
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    # 保存序列化后的语料
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)



     # 主程序入口
if __name__ == '__main__':
        # 词向量文件路径
        ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
        sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

        # ==========================最初基于StaQC的词典和词向量路径==========================
        python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
        python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
        python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

        sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
        sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
        sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

        # 调用函数生成新的词典和词向量文件
        # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
        # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

        # =======================================最后打标签的语料路径========================================
        # SQL待处理语料地址
        new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
        new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
        large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

        # SQL最后的词典和对应的词向量路径
        sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
        sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

        # 调用函数生成新词典和附加的词典
        # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
        # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)

        staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
        large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
        # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
        # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)、

        # 序列化数据
        # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
        serialization(python_final_word_dict_path, new_python_large, large_python_f)

        print('序列化完毕')

        # 测试函数，基于实际情况调用
        # test2(test_python1, test_python2, python_final_word_dict_path, python_final_word_vec_path)

