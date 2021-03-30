import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from scipy.stats import lognorm

# n - number of state |X|
# m - number of actions |U|
# W = W1 x W2
# p - number of noises |W|
# p1 - number of observed noises |W1|
# p2 - number of unobserved noises |W2|
# p = p1 * p2


def value(f, g, g_final, wdist, T):
    # f - n, m, p1, p2
    # g - n, p1, p2, m
    # g_final - n
    # wdist - p1 x p2

    n, m, p1, p2 = f.shape
    assert g.shape == (n, p1, p2, m)
    assert g_final.shape == (n,)
    assert wdist.shape == (p1, p2)
    wdist2 = wdist.sum(axis=0)
    v = np.zeros((T+1, n))
    mu = np.zeros((T, n, p1), dtype=int)
    v[-1] = g_final
    for t in range(T-1, -1, -1):
        for x in range(n):
            vt = v[t + 1]
            for w1 in range(p1):
                ev = vt[f[x, :, w1, :]]
                ev = ev @ wdist2
                mu[t, x, w1] = np.argmax(wdist2 @ g[x, w1] + ev)
            v_before_e = np.zeros((p1, p2))
            for w1, w2 in itertools.product(range(p1), range(p2)):
                v_before_e[w1, w2] = g[x, w1, w2, mu[t, x, w1]] + \
                                     v[t+1, f[x, mu[t, x, w1], w1, w2]]
            v[t, x] = np.sum(v_before_e * wdist)
    return mu, v


def cloop(f, g, mu, wdist):
    n, m, p1, p2 = f.shape
    T, n, p1 = mu.shape
    # n, p1, p2, m = g.shape
    # p1, p2 = wdist.shape

    fcl = np.zeros((T, n, p1, p2), dtype=int)
    gcl = np.zeros((T, n))
    for t in range(T):
        for x in range(n):
            for w1 in range(p1):
                for w2 in range(p2):
                    fcl[t, x, w1, w2] = f[x, mu[t, x, w1], w1, w2]
                    gcl[t, x] += g[x, w1, w2, mu[t, x, w1]] * wdist[w1, w2]
    fcl = fcl.reshape((T, n, -1))
    return fcl, gcl


def ftop(fcl, wdist):
    T, n, p = fcl.shape
    P = np.zeros((T, n, n))
    for t in range(T):
        for i in range(n):
            for j in range(n):
                P[t, i, j] = wdist[fcl[t, i] == j].sum()
    return P


def dist_iter(P, pi0, gcl, g_final):
    T, n = gcl.shape
    # T, n, _ = P.shape
    pi = np.zeros((T+1, n))
    pi[0] = pi0
    for t in range(1, T+1):
        pi[t] = pi[t-1] @ P[t-1]
    egcl = np.sum(gcl * pi[:-1], axis=1)
    eg_final = pi[-1] @ g_final
    return egcl, eg_final, pi

def test():
    T = 50
    p1 = 15
    p2 = 2
    m = 2
    n = 11
    w1val = np.linspace(0.6, 2, p1)
    w1dist = lognorm.pdf(w1val, s=0.2, scale=np.exp(0))
    w1dist = w1dist / w1dist.sum()
    w2dist = np.array([0.6, 0.4])
    wdist = w1dist.reshape([p1, 1]) @ w2dist.reshape([1, p2])
    f = np.zeros((n, m, p1, p2), dtype=int)
    g = np.zeros((n, p1, p2, m))
    g_final = np.zeros(n)
    pi0 = np.zeros(n)
    pi0[-1] = 1.

    for x, u, w1, w2 in itertools.product(*[range(k) for k in [n, m, p1, p2]]):
        if w2 == 1 and u == 1:
            f[x, u, w1, w2] = max([x - 1, 0])
            g[x, w1, w2, u] = w1val[w1]
        else:
            f[x, u, w1, w2] = x
            g[x, w1, w2, u] = 0
        if u == 1 and x == 0:
            g[x, w1, w2, u] = -np.inf

    mu_opt, v_opt = value(f, g, g_final, wdist, T)
    print(f'Expected cost of optimal policy: {v_opt[0, -1]}')

    mu_thr = np.zeros((T, n, p1), dtype=int)
    mu_always = np.ones((T, n, p1), dtype=int)
    mu_always[:, 0, :] = 0
    w1_mean = w1val @ w1dist
    for x in range(n):
        for w1 in range(p1):
            if w1val[w1] > w1_mean and x > 0:
                mu_thr[:, x, w1] = 1
    for mu in [mu_opt, mu_thr, mu_always]:
        fcl, gcl = cloop(f, g, mu, wdist)
        P = ftop(fcl, wdist.reshape(-1))
        egcl, eg_final, pi = dist_iter(P, pi0, gcl, g_final)
        print(f'Expected total cost: {egcl.sum() + eg_final}')
        print(f'Probability to have unsold stock: {1 - pi[-1, 0]}')

    pass

test()
