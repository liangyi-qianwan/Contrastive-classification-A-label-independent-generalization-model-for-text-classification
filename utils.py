# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            l=0
            for line in tqdm(f):
                l+=1
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content, label = lin.split('\t')
                except:
                    continue
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids,int(label),seq_len, mask))
        return contents

    train_question = load_dataset(config.train_question_path, config.pad_size)
    test_question = load_dataset(config.test_question_path, config.pad_size)
    test_answer = load_dataset(config.test_answer_path, 100)
    return train_question,test_question,test_answer


# 数据集迭代器
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
    #转成向量
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas])
        y = torch.LongTensor([_[1] for _ in datas])
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas])
        mask = torch.LongTensor([_[3] for _ in datas])
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
