import jieba
import random
import pkuseg
import zhconv
import pickle
import json

class Single_tokenizer():
    def cut(self, words):
        seg_words = list(words)
        return seg_words


class Date_Process():
    def __init__(self, file_path, tokenizer_name, userdict_path, stopwords_path, cleaning_parameters):
        self.file_path = file_path
        self.tokenizer_name = tokenizer_name
        self.userdict_path = userdict_path
        self.stopwords_path = stopwords_path
        self.cleaning_parameters = cleaning_parameters
        self.tokenizer = None
        self.stopwords = []
        self.raw_all_data = []
        self.all_data = []
        self.train_data = []
        self.test_data = []
        self.word2id = {}
        self.id2word = {}
        self.tag2id = {}
        self.id2tag = {}

    def init(self):
        self.read_data()
        self.tokenizer, self.stopwords = self.make_tokenizer(self.tokenizer_name, self.userdict_path, self.stopwords_path)
        self.data_process()

    # 读取数据
    def read_data(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                one_line = json.loads(line)
                self.raw_all_data.append([one_line['sentence'], one_line['label_desc'][5:]])

    # 选取分词器
    def make_tokenizer(self, tokenizer_name, userdict_path="", stopwords_path=""):
        if stopwords_path != "":
            # 将停用词读出来放在stopwords这个列表中
            stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
        else:
            stopwords = []
        if tokenizer_name == "pkuseg":
            if userdict_path != "":
                pku = pkuseg.pkuseg(user_dict=userdict_path)
            else:
                pku = pkuseg.pkuseg()
            return pku, stopwords
        elif tokenizer_name == "single":
            single_tokenizer = Single_tokenizer()
            return single_tokenizer, stopwords
        else:
            # 默认使用结巴分词
            if userdict_path != "":
                jieba.load_userdict(userdict_path)
            return jieba, stopwords

    # 全角字符到半角字符的转换
    def full2half(self, string):
        rstring = ""
        for char in string:
            inside_code = ord(char)
            if inside_code == 12288:
                # 全角空格直接转换
                inside_code = 32
                rstring += chr(inside_code)
            elif inside_code >= 65281 and inside_code <= 65374:
                # 全角字符 (除空格) 根据关系转化
                inside_code -= 65248
                rstring += chr(inside_code)
            else:
                rstring += chr(inside_code)
        return rstring

    # 数据清洗
    def data_cleaning(self, words, cleaning_parameters):
        if cleaning_parameters[0]:
            words = zhconv.convert(words, 'zh-cn')
        if cleaning_parameters[1]:
            words = words.lower()
        if cleaning_parameters[2]:
            words = "".join(words.split())
        if cleaning_parameters[3]:
            words = self.full2half(words)
        return words

    # 数据分词，分词前进行数据清洗
    def word_seg(self, all_data, tokenizer, stopwords, cleaning_parameters):
        segmented_all_data = []
        for sentences, tag in all_data:
            sentences = self.data_cleaning(sentences, cleaning_parameters)
            seg_list = tokenizer.cut(sentences)
            seg_list = [i for i in seg_list if i not in stopwords]
            segmented_all_data.append((seg_list, tag))
        return segmented_all_data

    # 制作词到ID的映射，标签到ID的映射
    def make_map_dict(self, segmented_all_data):
        all_tag = []
        all_words = []
        all_words.append('<PAD>')
        all_words.append('<UNK>')
        for seg_list, tag in segmented_all_data:
            for word in seg_list:
                if word not in all_words:
                    all_words.append(word)
            if tag not in all_tag:
                all_tag.append(tag)
        word2id = {all_words[i]: i for i in range(len(all_words))}
        id2word = {v: k for k, v in word2id.items()}
        tag2id = {all_tag[i]: i for i in range(len(all_tag))}
        id2tag = {v: k for k, v in tag2id.items()}
        return word2id, id2word, tag2id, id2tag

    # 数据处理
    def data_process(self):
        self.all_data = self.word_seg(self.raw_all_data[:50000], self.tokenizer, self.stopwords, self.cleaning_parameters)
        self.word2id, self.id2word, self.tag2id, self.id2tag = self.make_map_dict(self.all_data)

        random.seed(1)
        random.shuffle(self.all_data)
        data_len = len(self.all_data)
        train = int(data_len * 0.8)
        self.train_data = self.all_data[:train]
        self.test_data = self.all_data[train:]

    # 保存数据
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump([self.train_data, self.test_data, 
                         self.tag2id, self.id2tag, self.word2id, self.id2word], f)

    # 恢复数据
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, \
            self.tag2id, self.id2tag, self.word2id, self.id2word = pickle.load(f)
