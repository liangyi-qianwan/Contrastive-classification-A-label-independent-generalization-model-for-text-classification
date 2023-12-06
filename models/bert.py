# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_question_path = dataset + '/data/Yelp/train.txt'                                # 训练集
        self.test_question_path = dataset + '/data/Yelp/test.txt'                                  # 测试集
        self.test_answer_path = dataset + '/data/Yelp/class.txt'                                  #标签
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/Yelp/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/Supervised/bert.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 17710                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                             # epoch数
        self.batch_size = 20                                           # mini-batch大小
        self.pad_size = 512                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        # self.label_number = 10                                             #标签的数量

class Answer_bert(nn.Module):

    def __init__(self, config):
        super(Answer_bert, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x):
        #获得文本经过bert处理后的输出
        context = x[0].to('cuda:3')  # 输入的句子
        mask = x[2].to('cuda:3')  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False) #pooled为[CLS]的输出
        return pooled


class Question_bert(nn.Module):

    def __init__(self, config):
        super(Question_bert, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        #获得文本经过bert处理后的输出
        context = x[0].to('cuda:3')  # 输入的句子
        mask = x[2].to('cuda:3')  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False) #pooled为[CLS]的输出
        return pooled

class Model(nn.Module):


    def __init__(self, config):
        super(Model, self).__init__()
        self.quention = Question_bert(config)
        self.answer = Answer_bert(config)

    def forward(self, x,y):
        #获得文本经过bert处理后的输出
        question = self.quention(x)
        answer = self.answer(y)

        logits_per_question = F.cosine_similarity(question.unsqueeze(1), answer.unsqueeze(0), dim=-1)
        logits_per_answer = logits_per_question.t()
        return logits_per_question, logits_per_answer



