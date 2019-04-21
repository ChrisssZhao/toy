# -*- coding: utf-8 -*-
"""
 Module Summmary: 简单的实现astar算法 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/04/20
"""

def get_manhattan(grid, goal):
    """
    计算最短距离，也就是曼哈顿距离
    """
    manhattan = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            manhattan[i][j] = abs(i - goal[0]) + abs(j - goal[1])
            if grid[i][j] == 1:
                manhattan[i][j] = 99 #标记障碍物
    return manhattan 

def search(grid,init,goal,cost,manhattan):
    """
    a* 搜索算法
    """
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))] #存放上一步状态
    action = [[0 for col in range(len(grid[0]))] for row in range(len(grid))] #存放当前状态
    #初始化
    closed[init[0]][init[1]] = 1
    x = init[0]
    y = init[1]
    g = 0 #g 代表贪婪值
    f = g + manhattan[init[0]][init[0]] 
    cell = [[f, g, x, y]] #cell记录状态
    #标识位
    found = False #是否找到
    resign = False #是否没有路径
    #迭代搜索
    delta = [[-1, 0 ], [ 0, -1], [ 1, 0 ], [ 0, 1 ]] #定义四个方向坐标变化
            
    while not found and not resign:
        if len(cell) == 0:
            resign = True
            print 'failed'
            return None
        else:
            cell.sort()
            cell.reverse()
            next = cell.pop() 
            #更新状态
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + manhattan[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i
    invpath = [] #存放路径
    x = goal[0]
    y = goal[1]
    invpath.insert(0, [x, y])
    while x != init[0] or y != init[1]:
        x2 = x - delta[action[x][y]][0]
        y2 = y - delta[action[x][y]][1]
        x = x2
        y = y2
        invpath.insert(0, [x, y])
    return invpath

if __name__ == '__main__':
    print "Mine - Phoebe Ryan"
    #简单的迷宫矩阵，目的是寻找左上角到右下角的路径
    grid = [[0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]]
    init = [0, 0] #出发点
    goal = [len(grid)-1, len(grid[0])-1] #目标点, 第一个表示y 第二个为x 
    #到目标点的曼哈顿距离（忽略障碍物）
    manhattan = get_manhattan(grid, goal)
    cost = 1 #每一步的代价
    a = search(grid,init,goal,cost,manhattan)
    print "path: "
    for x in a:
        print x
