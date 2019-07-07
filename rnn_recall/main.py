# -*- coding: utf-8 -*-
"""
 Module Summmary: RNN实现序列召回  
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/07/06
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model
import utils

class Args():
    """设置参数"""
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 50
    dropout_p_hidden=1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    is_training = True
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

if __name__ == '__main__':
    print "Seve - Tez Cadey"
    ###加载数据集###
    TRAIN = './data/rsc15_train_full.txt'
    TEST = './data/rsc15_test.txt'
    data = pd.read_csv(TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(TEST, sep='\t', dtype={'ItemId': np.int64})
    ###参数定义###
    args = Args()
    args.n_items = len(data['ItemId'].unique())
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        gru.fit(data)
    ###验证集效果###
    res = utils.evaluate(gru, data, valid)
    print 'Recall@20', res[0]
    print 'MRR@20', res[1]


