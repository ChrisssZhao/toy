# -*- coding: utf-8 -*-
"""
 Module Summmary: 实现Q learning的小小demo 
                  让一个机器人在包含障碍物N*N的棋盘中从左上角走到右下角
                  最后输出学习到的策略的q table
Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/03/09
"""

import numpy as np
import pandas as pd
import random

N = 5 #棋盘的大小,偷懒直接用5，不用N...

#对应Q learning公式中的超参数
GAMMA = 0.8 #衰减系数
ACTIONS = 4 #动作个数
EPSOLON = 0.9 #90%的情况下选择最佳策略，10%随机选择策略
learning_rate = 0.05 #学习率

#四个动作及其Reward，(x,y)=(0,1)代码往上, 使用了state编号进行标记，(x,y)的标记方式舍弃
#X_IDX = [0, 0, -1 , 1] #依次是上下左右
#Y_IDX = [1, -1, 0, 0]

#board表示棋盘状态，从上到下，从左到右依次编号
#0代表可以通过1代表不能通过
board = np.zeros(25)
#随机设置2个障碍
board[11] = 1
board[18] = 1

#初始化Q table,行代表棋盘位置，列代表动作
q_table = pd.DataFrame(np.zeros(100).reshape(25,4),index=np.arange(25),columns=range(4))
#定义棋盘的边界，比如在(0,-)就不能往下走，q_table中应该为无穷小
q_table.loc[:4, 0] = float('-inf') #最上面一行不能往上走
q_table.loc[20: 24, 1] = float('-inf') #最下面不能往下走
q_table.loc[:20:5, 2] = float('-inf') #最左边不能往左走
q_table.loc[4:24:5, 3] = float('inf') #最右边不能往右走
pos_shift = [-5, 5, -1, 1]#标记上下左右动作变化时，状态的变化 
#print q_table

#开始迭代, demo固定轮数
for episode in range(100):
    initial_s = 0 #从左上角开始目标是右下角
    terminal_s = 24
    s = initial_s
    max_step = 50 #最大的尝试步数
    step = 0
    while( (s != terminal_s) and (step < max_step) ):
        actions = q_table.loc[s, :] #动作和对应的Q值
        if np.random.uniform() < EPSOLON: #选择最佳策略
            #从当前状态选出最佳的一步，可能不止一个
            next_action = np.random.choice(actions[ actions == np.max(actions) ].index)
        else:
            next_action = np.random.choice(actions[ actions != float('inf') ].index) #从可动作中取一个
        #更新状态
        #print "action: ", next_action,
        s_ = s + pos_shift[next_action]
        #奖惩
        #print "state: ", s_,' '
        if (s_<0) or (s_>24):
            break
        if board[s_] == 1:
            reward = -20 #撞上障碍物给予惩罚
        else:
            reward = -1 #走一步也要惩罚，让agent学习出少走几步测策略
        #根据公式更新q table
        max_q = np.max(q_table.loc[s_, :])
        q_table.loc[s, next_action] = q_table.loc[s, next_action] + learning_rate * (reward + GAMMA * max_q - q_table.loc[s, next_action] ) 
        
        s = s_
        step = step + 1

#最终的q_table, 通过q_table可以找出路径
print q_table

