# -*- coding: utf-8 -*-
"""
 Module Summmary: 实现简单的tarjan算法，可用于求有向图的强连通分量 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/05/03
"""

from collections import deque

def create_graph(n, edges):
    """
    构建图
    输入：
        n 顶点数
        edges 边 list
    输出：
        g 图,g[i]代表与以顶点i为起点的终点集合 list
    """
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
    return g

def tarjan(g):
    """
    tarjan算法求最大连通分量
    输入：
        g 图 list
    输出：
        components 强连通分量 list
    """
    #初始化参数
    n = len(g)
    stack = deque() #用于遍历
    on_stack = [False for _ in range(n)]
    index_of = [-1 for _ in range(n)]
    lowlink_of = index_of[:] #low值用于记录dfs遍历

    def strong_connect(v, index, components):
        """
        寻找某个顶点的最大连通分量
        输入：
            v dfs遍历到的点 int
            index 某次遍历的index int
            components 返回强连通分量 list
        输出：
            无
        """
        index_of[v] = index  
        lowlink_of[v] = index  
        index += 1
        stack.append(v)
        on_stack[v] = True
        for w in g[v]:
            if index_of[w] == -1:
                index = strong_connect(w, index, components)
                lowlink_of[v] = lowlink_of[w] if lowlink_of[w] < lowlink_of[v] else lowlink_of[v]
            elif on_stack[w]:
                lowlink_of[v] = lowlink_of[w] if lowlink_of[w] < lowlink_of[v] else lowlink_of[v]

        if lowlink_of[v] == index_of[v]:
            #利用栈进行遍历
            component = []
            w = stack.pop()
            on_stack[w] = False
            component.append(w)
            while w != v:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
            components.append(component)
        return index

    components = []
    #dfs遍历求强连通分量
    for v in range(n):
        if index_of[v] == -1:
            strong_connect(v, 0, components)
    return components

if __name__ == "__main__":
    print "Breath and Life - Audio Machine"
    vertice_cnt = 7 #定点个数
    #source->target表示, source表示起点，target表示终点 
    source = [0, 0, 1, 2, 3, 3, 4, 4, 6]
    target = [1, 3, 2, 0, 1, 4, 5, 6, 5]
    edges = [(x, y) for x, y in zip(source, target)]
    g = create_graph(vertice_cnt, edges)
    print "edges: ", edges
    print "strongly connected components: ", tarjan(g)

