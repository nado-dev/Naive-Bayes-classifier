import math
import re
import pandas as pd
import codecs
import jieba


def load_formatted_data():
    """
    加载格式化后的标签-路径列表
    spam列为1代表是垃圾邮件，0代表普通邮件
    path列代表该邮件路径
    :return:(DataFrame)index
    """
    # 加载数据集
    index = pd.read_csv('index', sep=' ', names=['spam', 'path'])
    index.spam = index.spam.apply(lambda x: 1 if x == 'spam' else 0)
    index.path = index.path.apply(lambda x: x[1:])
    return index


def load_stop_word():
    """
    读出停用词列表
    :return: (List)_stop_words
    """
    with codecs.open("stop", "r") as f:
        lines = f.readlines()
    _stop_words = [i.strip() for i in lines]
    return _stop_words


def get_mail_content(path):
    """
    遍历得到每封邮件的词汇字符串
    :param path: 邮件路径
    :return:(Str)content
    """
    with codecs.open(path, "r", encoding="gbk", errors="ignore") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[i] == '\n':
            # 去除第一个空行，即在第一个空行之前的邮件协议内容全部舍弃
            lines = lines[i:]
            break
    content = ''.join(''.join(lines).strip().split())
    # print(content)
    return content


def create_word_dict(content, stop_words_list):
    """
    依据邮件的词汇字符串统计词汇出现记录，依据停止词列表除去某些词语
    :param content: 邮件的词汇字符串
    :param stop_words_list:停止词列表
    :return:(Dict)word_dict
    """
    word_list = []
    word_dict = {}
    # word_dict key:word, value:1
    content = re.findall(u"[\u4e00-\u9fa5]", content)
    content = ''.join(content)
    word_list_temp = jieba.cut(content)
    for word in word_list_temp:
        if word != '' and word not in stop_words_list:
            word_list.append(word)
    for word in word_list:
        word_dict[word] = 1
    return word_dict


def train_dataset(dataset_to_train):
    """
    对数据集进行训练, 统计训练集中某个词在普通邮件和垃圾邮件中的出现次数
    :param dataset_to_train: 将要用来训练的数据集
    :return:Tuple(词汇出现次数字典_train_word_dict, 垃圾邮件总数spam_count, 正常邮件总数ham_count)
    """
    _train_word_dict = {}
    # train_word_dict内容，训练集中某个词在普通邮件和垃圾邮件中的出现次数
    for word_dict, spam in zip(dataset_to_train.word_dict, dataset_to_train.spam):
        # word_dict某封信的词汇表 spam某封信的状态
        for word in word_dict:
            # 对每封信的每个词在该邮件分类进行出现记录 出现过为则记录数加1 未出现为0
            _train_word_dict.setdefault(word, {0: 0, 1: 0})
            _train_word_dict[word][spam] += 1
    ham_count = dataset_to_train.spam.value_counts()[0]
    spam_count = dataset_to_train.spam.value_counts()[1]
    return _train_word_dict, spam_count, ham_count


def predict_dataset(_train_word_dict, _spam_count, _ham_count, data):
    """
    测试算法
    :param _train_word_dict:词汇出现次数字典
    :param _spam_count:垃圾邮件总数
    :param _ham_count:正常邮件总数
    :param data:测试集
    :return:
    """
    total_count = _ham_count + _spam_count
    word_dict = data['word_dict']

    # 先验概率 已经取了对数
    ham_probability = math.log(float(_ham_count) / total_count)
    spam_probability = math.log(float(_spam_count) / total_count)

    for word in word_dict:
        word = word.strip()
        _train_word_dict.setdefault(word, {0: 0, 1: 0})

        # 求联合概率密度 += log
        # 拉普拉斯平滑
        word_occurs_counts_ham = _train_word_dict[word][0]
        # 出现过这个词的信件数 / 垃圾邮件数
        ham_probability += math.log((float(word_occurs_counts_ham) + 1) / _ham_count + 2)

        word_occurs_counts_spam = _train_word_dict[word][1]
        # 出现过这个词的信件数 / 普通邮件数
        spam_probability += math.log((float(word_occurs_counts_spam) + 1) / (_spam_count + 2))

    if spam_probability > ham_probability:
        is_spam = 1
    else:
        is_spam = 0

    # 返回预测正确状态
    if is_spam == data['spam']:
        return 1
    else:
        return 0


def save_train_word_dict(train_word_dict):
    with codecs.open("train_word_dict", "w", encoding="gbk", errors="ignore") as f:
        f.write(train_word_dict)


if __name__ == '__main__':

    index_list = load_formatted_data()
    stop_words = load_stop_word()
    # get_mail_content(index_list.path[0])
    index_list['content'] = index_list.path.apply(lambda x: get_mail_content(x))
    index_list['word_dict'] = index_list.content.apply(lambda x: create_word_dict(x, stop_words))

    train_set = index_list.loc[:len(index_list) * 0.8]
    test_set = index_list.loc[len(index_list) * 0.8:]

    train_word_dict, spam_count, ham_count = train_dataset(train_set)

    save_train_word_dict(train_word_dict)

    test_mails_predict = test_set.apply(
        lambda x: predict_dataset(train_word_dict, spam_count, ham_count, x), axis=1)

    corr_count = 0
    false_count = 0
    for i in test_mails_predict.values.tolist():
        if i == 1:
            corr_count += 1
        if i == 0:
            false_count += 1

    print("测试集样本总数", (corr_count + false_count))
    print("正确预计个数", corr_count)
    print("错误预测个数", false_count)

    result = float(corr_count / (corr_count + false_count))
    print('预测准确率：', result)
