# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network,test
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif


def _to_tensor(datas):
    x = torch.LongTensor([_[0] for _ in datas])
    # y = torch.LongTensor([_[1] for _ in datas]).to(device)
    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([_[2] for _ in datas])
    mask = torch.LongTensor([_[3] for _ in datas])
    return (x, seq_len, mask)


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 使用什么模型
    model_name = 'bert'  # bert
    # 导入响应的模型
    x = import_module('models.' + model_name)
    # 相应模型的配置
    config = x.Config(dataset)
    # 随机种子
    np.random.seed(1)
    # 设置随机种子
    torch.manual_seed(1)

    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 构建数据
    train_question,test_question,test_answer = build_dataset(config)


    train_question = build_iterator(train_question, config)
    test_question = build_iterator(test_question, config)
    test_answer = _to_tensor(test_answer)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_question, test_question,test_answer)
    # test(config,model,test_question,test_answer)
