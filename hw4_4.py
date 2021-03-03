import numpy as np
from collections import namedtuple
from hw4_1 import forward_bellman_ford

Edge = namedtuple('Edge', ['source', 'target', 'weight'])

n = 15
v = [3, 4, 5, 7, 1, 9, 6, 9, 4, 5, 1, 2, 7, 4, 9]
w = [2, 10, 6, 8, 10, 3, 5, 5, 8, 9, 2, 2, 4, 1, 6]
C = 25

source_node = n
target_node = n + 1


def knapsack(v, w, C, n):
    nodes = {(i, w) for i in range(n+1) for w in range(C + 1)}
    nodes.add('z')

    edges = set()
    for i in range(n):
        for c in range(C + 1):
            edges.add(Edge((i, c), (i+1, c), 0))
            if c + w[i] <= C:
                edges.add(Edge((i, c), (i+1, c+w[i]), -v[i]))
    for c in range(C + 1):
        edges.add(Edge((n, c), 'z', 0))

    node2index = {node: i for i, node in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes))) + np.inf
    for edge in edges:
        i = node2index[edge.source]
        j = node2index[edge.target]
        W[i, j] = edge.weight

    s = node2index[(0, 0)]
    t = node2index['z']
    p, wp = forward_bellman_ford(W, s, t)
    index2node = {i: node for node, i in node2index.items()}
    S = set()
    for i in range(n):
        node1 = index2node[p[i]]
        node2 = index2node[p[i+1]]
        if node1[1] < node2[1]:
            S.add(i)
    print(', '.join(map(str, S)))
    print(-wp)
    return S


S = knapsack(v, w, C, n)

