# -*- coding: utf-8 -*-
# Copyright (c) 2019 ke.com, Inc. All Rights Reserved
"""
 Module Summmary: 实现简单的bitmap
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/05/12
"""

import random

class BitMap(object):
    def __init__(self, m):
        """
        m bitmap的大小
        """
        self.max_size = self.get_index(m, True)
        self.array = [0 for i in range(self.max_size)] #排序数组

    def get_index(self, val, up=False):
        """
        获取某个值的坐标
        """
        if up:
            return int((val + 31 - 1) / 31)
        return val / 31

    def get_bit_pos(self, val):
        """获取val的位索引"""
        return val % 31

    def set_one(self, val):
        """将val对应的位索引置1"""
        idx = self.get_index(val) #数组idx
        bit_pos = self.get_bit_pos(val)
        num = self.array[idx]
        self.array[idx] = num | (1 << bit_pos) #移位操作

    def is_exist(self, val):
        """检查val是否已存在"""
        idx = self.get_index(val) #数组idx
        bit_pos = self.get_bit_pos(val)
        num = self.array[idx]
        if self.array[idx] & ( 1 << bit_pos):
            return True
        return False

if __name__ == "__main__":
    print "Lemon - 迷津玄师"
    MAX = 10000
    #随机数数组
    array = []
    for x in range(10):
        array.append(random.randrange(1, MAX))
    res = []
    bitmap = BitMap(MAX)
    for num in array:
        bitmap.set_one(num)

    for i in range(MAX + 1):
        if bitmap.is_exist(i):
            res.append(i)

    print "oringal array: ", array
    print "sorted: ", res

