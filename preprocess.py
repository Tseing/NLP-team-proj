# coding = UTF-8

import os
import thulac
import pickle
import numpy as np

dir_path = "../dataset"
csv_path = "/".join([dir_path, "Train.csv"])  # csv 文件
articles_path = "/".join([dir_path, "articles.txt"])  # csv 中新闻原文拼合成长文的 txt
total_words_path = "/".join([dir_path, "total_words.txt"])  # 经过分词得到全部词语的 txt

# 训练集与验证集的词语 txt
words_path = {
    "train": "/".join([dir_path, "train_set.txt"]),
    "val": "/".join([dir_path, "val_set.txt"]),
    "_train": "/".join([dir_path, "train_set.txt"])
}

# 训练集与验证集的词语索引 dict
vocab_path = {
    "train": "/".join([dir_path, "train_vocab.pkl"]),
    "val": "/".join([dir_path, "val_vocab.pkl"]),
    "_train": "/".join([dir_path, "train_vocab.pkl"]),
}

# 训练集与验证集的用索引表示的新闻原文
corpus_path = {
    "train": "/".join([dir_path, "train_corpus.npy"]),
    "val": "/".join([dir_path, "val_corpus.npy"]),
    "_train": "/".join([dir_path, "train_corpus.npy"]),
}


def get_lines(filename, head=1, end=None):
    """
    按行读文件，并去除每行的非法字符（GBK 以外字符）
    :param filename:
    :param head:
    :param end:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()[head:end]
        for line in lines:
            yield line.encode("GBK", "ignore").decode("GBK")


def process_line(filename):
    for line in get_lines(filename):
        processed_line = line.split(",", 2)[2].strip()
        yield processed_line


def csv2json(filename):
    with open("_".join([os.path.basename(filename).split(".")[0], "precessed.json"]),
              "w", encoding='utf-8') as f:
        f.write("[\n")
        for line in process_line(filename):
            f.write("\t\"")
            f.write(line)
            f.write("\",\n")
        f.write("\t\" \"\n")
        f.write("]")
    print("Done!")


def get_articles(filename):
    """
    每行一篇新闻，以 <eos> 为结束符，写入 txt
    :param filename:
    :return:
    """
    with open(articles_path,
              "w", encoding='utf-8') as f:
        for line in process_line(filename):
            f.write(line)
            f.write("<eos>\n")
    print("Articles File Done!")


def articles2words(output_path):
    """
    分词引擎，将新闻划分为词语，以空格分隔
    :param output_path:
    :return:
    """
    word_sep_model = thulac.thulac(seg_only=True, user_dict="../user_dict.txt")
    word_sep_model.cut_f(articles_path, output_path)


def load_words(data_type="train", is_multi_file=False, file_idx=None):
    """
    通过词语 txt 生成词语索引的 dict
    :param data_type:
    :return:
    """

    word_to_id = {}
    id_to_word = {}

    words = open(words_path[data_type], encoding="utf-8").read().strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path[data_type], 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_vocab(data_type="train", is_multi_file=False, file_idx=None):
    """
    判断相应数据文件是否存在，生成相应格式的数据文件，得到词语与索引的 dict
    :param data_type:
    :return:
    """

    if os.path.exists(vocab_path[data_type]):
        with open(vocab_path[data_type], 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    elif os.path.exists(words_path[data_type]):
        return load_words(data_type)

    elif os.path.exists(total_words_path):
        if is_multi_file:
            divide_multi_dataset(total_words_path)
        else:
            divide_dataset(total_words_path)
        return load_words(data_type)

    else:
        get_articles(csv_path)
        articles2words(output_path=total_words_path)
        if is_multi_file:
            divide_multi_dataset(total_words_path)
        else:
            divide_dataset(total_words_path)
        return load_words(data_type)


def load_data(data_type="train", is_multi_file=False, file_idx=None):
    """
    读取数据集，corpus 为以词索引表示的新闻，word_to_id 与 id_to_word 为查询词或索引的 dict
    :param is_multi_file:
    :param file_idx:
    :param data_type:
    :return:
    """

    if is_multi_file:
        words_path["train"] = str(file_idx).join(os.path.splitext(words_path["_train"]))
        vocab_path["train"] = str(file_idx).join(os.path.splitext(vocab_path["_train"]))
        corpus_path["train"] = str(file_idx).join(os.path.splitext(corpus_path["_train"]))
        word_to_id, id_to_word = load_vocab(data_type, is_multi_file=True, file_idx=file_idx)
    else:
        word_to_id, id_to_word = load_vocab(data_type)

    if os.path.exists(corpus_path[data_type]):
        corpus = np.load(corpus_path[data_type])
        return corpus, word_to_id, id_to_word

    words = open(words_path[data_type], encoding="utf-8").read().strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(corpus_path[data_type], corpus)
    return corpus, word_to_id, id_to_word


def divide_dataset(filename, size=100):
    """
    划分训练集与验证集
    :param filename:
    :param size:
    :return:
    """
    lines_len = 160332
    idx = 0
    np.random.seed(0)
    random_idx = iter(np.sort(np.random.choice(lines_len, size, replace=False)))
    chosen_idx = next(random_idx)

    with open(words_path["train"], "w", encoding="utf-8") as train_set:
        with open(words_path["val"], "w", encoding="utf-8") as val_set:
            for line in get_lines(filename):
                try:
                    if idx == chosen_idx:
                        val_set.write(line)
                        idx += 1
                        chosen_idx = next(random_idx)
                    else:
                        train_set.write(line)
                        idx += 1
                except StopIteration:
                    train_set.write(line)

    print("Dataset Division Done!")


def divide_multi_dataset(filename, size=25000):
    lines_len = 160332
    cnt = 0
    file_num = lines_len // size

    for file_idx in range(file_num):
        with open(str(file_idx).join(os.path.splitext(words_path["_train"])),
                  "w", encoding="utf-8"):
            continue
    val_set = open(words_path["val"], "w", encoding="utf-8")
    val_set.close()

    for line in get_lines(filename):
        if cnt <= file_num * size:
            file_idx = np.random.randint(0, file_num)
            with open(str(file_idx).join(os.path.splitext(words_path["_train"])),
                      "a", encoding="utf-8") as train_set:
                train_set.write(line)
                cnt += 1
        else:
            with open(words_path["val"], "a", encoding="utf-8") as val_set:
                val_set.write(line)

    print("Dataset Division Done!")
