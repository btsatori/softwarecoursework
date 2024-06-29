import pickle
from collections import Counter



def load_pickle(filename):
    """从文件加载Pickle对象"""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data


def split_data(total_data, qids):
    """根据问题ID的出现次数，将数据拆分为single和multiple两部分"""
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    for data in total_data:
        if result[data[0][0]] == 1:
            total_data_single.append(data)
        else:
            total_data_multiple.append(data)
    return total_data_single, total_data_multiple


def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """处理STAQC数据，并保存为single和multiple数据文件"""
    with open(filepath, 'r') as f:
        total_data = eval(f.read())  # 注意使用eval可能存在安全隐患，需确保源数据可信
    qids = [data[0][0] for data in total_data]
    total_data_single, total_data_multiple = split_data(total_data, qids)

    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


def data_large_processing(filepath, save_single_path, save_multiple_path):
    """处理大规模数据，并保存为single和multiple数据文件"""
    total_data = load_pickle(filepath)
    qids = [data[0][0] for data in total_data]
    total_data_single, total_data_multiple = split_data(total_data, qids)

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


def single_unlabeled_to_labeled(input_path, output_path):
    """将单标签的未标记数据转换为带标签的数据"""
    total_data = load_pickle(input_path)
    labels = [[data[0], 1] for data in total_data]  # 给每个数据项添加标签1
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 先按第一个元素排序，再按第二个元素排序
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))  # 注意，这里使用str保存可能并不像pickle那样高效


if __name__ == "__main__":
    # 定义文件路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'

    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'

    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'

    # 创建目录（如果还不存在）
    for path in [
        staqc_python_single_save, staqc_python_multiple_save,
        staqc_sql_single_save, staqc_sql_multiple_save,
        large_python_single_save, large_python_multiple_save,
        large_sql_single_save, large_sql_multiple_save,
        large_sql_single_label_save, large_python_single_label_save
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # 处理STAQC的数据
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 处理大型数据集
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    # 将未标记的单标签数据保存为标记的单标签数据
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)