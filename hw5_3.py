import itertools
import json
import numpy as np


def value(f, g, g_final, wdist, T):
    # T, n, m = g.shape
    n, m, p = f.shape
    v = np.zeros((T+1, n))
    mu = np.zeros((T, n), dtype=int)
    v[-1] = g_final
    for t in range(T-1, -1, -1):
        for x in range(n):
            vt = v[t+1]
            ev = vt[f[x]]
            ev = ev @ wdist
            mu[t, x] = np.argmin(g[t, x] + ev)
            v[t, x] = np.min(g[t, x] + ev)
    return mu, v


def value_with_observed_w1(f, g, g_final, wdist, T):
    # f - n, m, p1, p2
    # g - t, n, p1, p2, m
    # g_final - n
    # wdist - p1 x p2

    n, m, p1, p2 = f.shape
    assert g.shape == (T, n, p1, p2, m)
    assert g_final.shape == (n,)
    assert wdist.shape == (p1, p2)
    wdist2 = wdist.sum(axis=0)
    v = np.zeros((T + 1, n))
    mu = np.zeros((T, n, p1), dtype=int)
    v[-1] = g_final
    for t in range(T - 1, -1, -1):
        for x in range(n):
            vt = v[t + 1]
            for w1 in range(p1):
                ev = vt[f[x, :, w1, :]]
                ev = ev @ wdist2
                mu[t] = lambda x, w1: np.argmin(wdist2 @ g[t, x, w1] + ev)
            v_before_e = np.zeros((p1, p2))
            for w1, w2 in itertools.product(range(p1), range(p2)):
                v_before_e[w1, w2] = g[t, x, w1, w2, mu[t, x, w1]] + \
                                     v[t + 1, f[x, mu[t, x, w1], w1, w2]]
            v[t, x] = np.sum(v_before_e * wdist)
    return mu, v


data = json.load(open('appliance_data.json'))
T = data['T']['data']
p_var = np.array(data['p_var']['data'])
p_mu = np.array(data['p_mu']['data'])
e_c = np.array(data['e_c']['data'])
C = data['C']['data']

n = C+1
m = 2
p = 1
f = np.zeros((n, m, p), dtype=int)
g = np.zeros((T, n, m))
for x in range(n):
    f[x, 0] = x
    f[x, 1] = x + 1
    for t in range(T):
        if x == C:
            g[t, x, 1] = np.inf
        else:
            g[t, x, 1] = e_c[x] * p_mu[t]
f = np.clip(f, 0, C)
g_final = np.zeros(n)
g_final[:-1] = np.inf

mu, v = value(f, g, g_final, np.ones(1), T)
mu, v = value_with_observed_w1(f, g, g_final, np.ones(1), T)

pass
