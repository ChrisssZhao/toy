# -*- coding: utf-8 -*-
"""
 Module Summmary: 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/03/31
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#数据集，全局变量
from tensorflow.examples.tutorials.mnist import input_data
minst = input_data.read_data_sets("./data/", one_hot=True) #需要科学上网


def BiLSTM(x, w, b, num_hidden, steps):
    """
    定义基本单元
    输入:
        x 数据输入，结构为（batch_size, steps, n_inpu）
        w,b 为参数向量  
        num_hidden 为隐藏层数
        steps 步长
    """
    x = tf.unstack(x, steps, 1) #矩阵分解，取出tensor
    #定义前向和后向基本单元
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    #获取输出
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], w) + b

def train_model(X, Y, w, b, num_hidden, steps, training_steps, learning_rate, num_input, batch_size, display_step):
    """
    训练函数
    输入：
        X 输入矩阵
        Y labels
        w,b 参数向量
        其他参数 为网络结构和训练参数
    """
    #单元结构
    logits = BiLSTM(X, w, b, num_hidden, steps)
    #定义损失函数和优化函数
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    #评价函数
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #开始训练
    print "start training"
    with tf.Session() as sess:
        init = tf.global_variables_initializer() #初始化参数
        sess.run(init)
        for step in range(training_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size) 
            batch_x = batch_x.reshape((batch_size, steps, num_input)) #保证28*28
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 :#打印训练情况
                #计算损失和准确度   
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print "step: ",step
                print "loss: ",loss
                print "acc: ",acc
    print "training finished"

    #预测测试集
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, steps, num_input))
    test_label = mnist.test.labels[:test_len]
    print "test acc:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label})

if __name__ == "__main__":
    print "lucky happens."
    #超参数
    learning_rate = 0.001
    training_steps = 1000
    batch_size = 128
    display_step = 100
    #网络结构参数
    num_input = 28 # MNIST 图片大小28*28
    steps = 28 #步长
    num_hidden = 128  
    num_classes = 10 #数字0-9
    #输出输出
    X = tf.placeholder("float", [None, steps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    #w,b参数向量
    w = tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
    b = tf.Variable(tf.random_normal([num_classes]))
    #训练
    train_model(X, Y, w, b, num_hidden, steps, training_steps, learning_rate, num_input, batch_size, display_step):



