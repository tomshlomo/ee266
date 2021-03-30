import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools


def value(f, g, g_final, wdist, T):
    n, m, p = f.shape
    v = np.zeros((T+1, n))
    mu = np.zeros((T, n), dtype=int)
    v[-1] = g_final
    for t in range(T-1, -1, -1):
        for x in range(n):
            vt = v[t+1]
            ev = vt[f[x]]
            ev = ev @ wdist
            mu[t, x] = np.argmin(g[x] + ev)
            v[t, x] = np.min(g[x] + ev)
    return mu, v


def cloop(f, g, mu):
    n, m, p = f.shape
    T = mu.shape[0]
    fcl = np.zeros((T, n, p), dtype=int)
    gcl = np.zeros((T, n))
    for t in range(T):
        for x in range(n):
            for w in range(p):
                fcl[t, x, w] = f[x, mu[t, x], w]
            gcl[t, x] = g[x, mu[t, x]]
    return fcl, gcl


def ftop(fcl, wdist):
    T, n, p = fcl.shape
    P = np.zeros((T, n, n))
    for t in range(T):
        for i in range(n):
            for j in range(n):
                P[t, i, j] = wdist[fcl[t, i] == j].sum()
    return P


def test():
    T = 50
    C = 20
    D = 4
    q0 = 10
    prob_d = np.array([0.2, 0.25, 0.25, 0.2, 0.1])

    p_fixed = 4
    p_whole = 2
    p_disc = 1.6
    u_disc = 6
    s_lin = 0.1
    s_quad = 0.05
    p_rev = 3
    p_unmet = 3
    p_sal = 1.5

    n = C + 1
    m = n
    p = D + 1

    f = np.zeros((n, m, p), dtype=int)
    g_parts = {name: np.zeros((n, m, p)) for name in ['order', 'store', 'rev', 'unmet', 'constraint']}
    for q, u, d in itertools.product(*[range(x) for x in [n, m, p]]):
        f[q, u, d] = min([max([q + u - d, 0]), C])
        if u == 0:
            g_parts['order'][q, u, d] = 0
        elif 1 <= u <= u_disc:
            g_parts['order'][q, u, d] = p_fixed + p_whole * u
        else:
            g_parts['order'][q, u, d] = p_fixed + p_whole * u_disc + p_disc * (u - u_disc)

        g_parts['store'][q, u, d] = s_lin * q + s_quad * q ** 2
        g_parts['rev'][q, u, d] = -p_rev * min([q + u, d])
        g_parts['unmet'][q, u, d] = p_unmet * max([-q - u + d, 0])
        g_parts['constraint'][q, u, d] = np.inf if q+u-d > C else 0
    for part_name, g_part in g_parts.items():
        g_parts[part_name] = np.sum(g_part * prob_d, axis=-1)
    g = np.sum(list(g_parts.values()), axis=0)

    g_final = -p_sal * np.arange(n)
    mu, v = value(f, g, g_final, prob_d, T)
    print(f"J_star = {v[0, q0]}")

    fig = px.imshow(mu)
    fig.update_xaxes(title="state")
    fig.update_yaxes(title="time")
    fig.show()

    fig = px.imshow(v)
    fig.update_xaxes(title="state")
    fig.update_yaxes(title="time")
    fig.show()

    fcl, gcl = cloop(f, g, mu)
    gcl_parts = {}
    for part_name, g_part in g_parts.items():
        _, gcl_parts[part_name] = cloop(f, g_part, mu)
    P = ftop(fcl, prob_d)
    pi = np.zeros((T+1, n))
    pi[0, q0] = 1
    for t in range(1, T+1):
        pi[t] = pi[t-1] @ P[t-1]
    egcl_parts = {}
    for part_name, g_part in gcl_parts.items():
        egcl_parts[part_name] = np.sum(g_part * pi[:-1], axis=1)
    px.line(y=list(egcl_parts.values()), labels=list(egcl_parts.keys())).show()
test()
