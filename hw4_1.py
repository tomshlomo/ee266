import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

data = json.load(open('shortest_path_data.json'))


def reverse_bellman_ford(W, s, t):
    W = W.astype(float)
    W[W >= 9999999] = np.inf
    n = W.shape[0]
    v = np.inf + np.zeros((n + 1, n))
    v[0, t] = 0
    next_node = -np.ones(n, dtype=int)
    for k in range(n):
        for i in range(n):
            j = np.argmin(v[k] + W[i])
            x = v[k, j] + W[i, j]
            if x < v[k, i]:
                v[k + 1, i] = x
                next_node[i] = j
            else:
                v[k + 1, i] = v[k, i]
    wp = v[-1, s]
    p = [s]
    while p[-1] != t:
        p.append(next_node[p[-1]])
    print(f'reverse: {wp}')
    print('->'.join(map(str, p)))
    return p, wp


def forward_bellman_ford(W, s, t):
    W = W.astype(float)
    W[W >= 9999999] = np.inf
    n = W.shape[0]
    v = np.inf + np.zeros((n + 1, n))
    v[0, s] = 0
    prev_node = -np.ones(n, dtype=int)
    for k in range(n):
        for i in range(n):
            j = np.argmin(v[k] + W[:, i])
            x = v[k, j] + W[j, i]
            if x < v[k, i]:
                v[k + 1, i] = x
                prev_node[i] = j
            else:
                v[k + 1, i] = v[k, i]
    wp = v[-1, t]
    p = [t]
    while p[-1] != s:
        p.append(prev_node[p[-1]])
    p.reverse()
    print(f'forward: {wp}')
    print('->'.join(map(str, p)))
    return p, wp


if __name__ == '__main__':
    for i in map(str, range(1, 6)):
        for f in [reverse_bellman_ford, forward_bellman_ford]:
            p, wp = f(np.array(data['W' + i]['data']),
                      np.array(data['s' + i]['data']) - 1,
                      np.array(data['t' + i]['data']) - 1)


