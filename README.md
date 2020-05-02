

## 朴素贝叶斯分类器实现垃圾邮件分类

朴素贝叶斯分类器原理：[朴素贝叶斯分类器 - 维基百科，自由的百科全书]([https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8](https://zh.wikipedia.org/wiki/朴素贝叶斯分类器))

朴素贝叶斯分类器做垃圾分类：[贝叶斯推断及其互联网应用（二）：过滤垃圾邮件 - 阮一峰的网络日志](http://www.ruanyifeng.com/blog/2011/08/bayesian_inference_part_two.html)

[数据集](https://drive.google.com/open?id=15Yi14PBw9P1pb045_aIRa-C3cdP0PKT_)

将数据集解压到仓库路径下即可

运行：`python new.py`

## 实验数据及结果分析：

部分训练字典如下 

```
{'非': {0: 337, 1: 784}, 
 '财务': {0: 72, 1: 8821}, 
 '纠淼': {0: 0, 1: 214},
 '牟': {0: 1, 1: 149}, 
 '莆': {0: 1, 1: 183}, 
 '窆': {0: 7, 1: 130}, 
 '沙盘': {0: 0, 1: 117}, 
 '模拟': {0: 56, 1: 761}, 
 '运用': {0: 95, 1: 1267}, ...} 
```



当测试集选为20% 训练集选为80% 拉普拉斯平滑参数λ=1 时

```
测试集样本总数 12924
正确预计个数 11326
错误预测个数 1598
预测准确率： 0.8763540699473847
```



## 步骤解析

### 1. 收集数据，收集垃圾邮件文本数据以及停用词

仓库包含了两个文件`index`以及`stop`， 前者包含了邮件目录和标签，后者包含了停用词列表。邮件被分成两类：`spam`垃圾邮件以及`ham`正常邮件。

首先要从`index`和`stop`中读出标签-路径列表和停用词列表，由函数`load_formatted_data()`

和`load_stop_word()`实现。

针对`DataFrame`类的特点，使用函数式编程(lambda表达式)可以大幅提高效率以及代码可读性。

`main`函数内：

```python
if __name__ == '__main__':
    index_list = load_formatted_data()
    stop_words = load_stop_word()
```

对应函数：

```python
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

```

```python
def load_stop_word():
    """
    读出停用词列表
    :return: (List)_stop_words
    """
    with codecs.open("stop", "r") as f:
        lines = f.readlines()
    _stop_words = [i.strip() for i in lines]
    return _stop_words
```

### 2.加载数据，使用pandas加载数据，并使用函数式编程(lambda)来对数据进行预处理。

依据上一步读取到的标签-路径列表，遍历得到每封邮件的词汇字符串，得到字符串之后，制作每封邮件的词汇字典(Dictionary)，形如`{word:1}`。此处采用的是词集模型，即全部文档中的所有单词构成的集合，每个单词只出现一次，仅仅考虑词是否在文本中出现，而不考虑词频。

`main`函数内：

```python
if __name__ == '__main__':
	...
index_list['content'] = index_list.path.apply(lambda x: get_mail_content(x))
index_list['word_dict'] = index_list.content.apply(lambda x: create_word_dict(x, stop_words))
```

对应函数：

```python
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
```

```python
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
    # 限定只查找中文字符
    content = re.findall(u"[\u4e00-\u9fa5]", content)
    content = ''.join(content)
    word_list_temp = jieba.cut(content)
    for word in word_list_temp:
        if word != '' and word not in stop_words_list:
            word_list.append(word)
    for word in word_list:
        word_dict[word] = 1
    return word_dict

```

中文分词是通过第三方库`jieba`实现的

### 3. 训练算法

首先需要设置训练集和测试集，假设选定前80%的data数据作为训练集，20%作为测试集

```python
if __name__ == '__main__':
	...
 	train_set = index_list.loc[:len(index_list) * 0.8]
 	test_set = index_list.loc[len(index_list) * 0.8:]
```

对数据集进行训练, 统计训练集中某个词在普通邮件和垃圾邮件中的出现次数, 为计算先验概率和后验概率提供数据。

```python
if __name__ == '__main__':
	...
	train_word_dict, spam_count, ham_count = train_dataset(train_set)
```



```python
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

```

### 4. 测试算法

先验概率P(s)极大似然估计

P(spam) =  垃圾邮件数/邮件总数

P(ham) = 正常邮件数/邮件总数

为了计算避免数字过小丢失精度，以下计算均以对数形式进行

用W表示某个词，现在需要计算P(S|W)的值，即在某个词语（W）已经存在的条件下，垃圾邮件（S）的概率有多大。

![img](http://chart.googleapis.com/chart?cht=tx&chl=P(S%7CW)%3D%5Cfrac%7BP(W%7CS)P(S)%7D%7BP(W%7CS)P(S)%2BP(W%7CH)P(H)%7D&chs=70)

又因为对于每个词P(S | H) ，分母与上式一致，所以只需比较分子即可得出结论——大者为更可能的分类结果。

为了增强信度，基于上面的推理，需要计算联合概率密度，对每个出现在一信件中的所有词汇的所有后验概率计算联合概率，大者为更可能的分类结果。

```python
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
        spam_probability += math.log((float(word_occurs_counts_spam) + 1) / _spam_count + 2)

    if spam_probability > ham_probability:
        is_spam = 1
    else:
        is_spam = 0

    # 返回预测正确状态
    if is_spam == data['spam']:
        return 1
    else:
        return 0

```

拉普拉斯平滑：发现0概率会给后验概率计算带来致命影响，从实际意义上看，未出现在训练集中的词语不能说是不可能的，所以有必要指定一个默认值。这个过程称为拉普拉斯平滑。

### 5. 使用算法

测试算法得出的预测结论与实际分类情况做比较，得出准确率

```python
if __name__ == '__main__':
	...
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
```

