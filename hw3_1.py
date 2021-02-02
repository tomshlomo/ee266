import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go


def mu(i):
    if i <= 15:
        return 1
    elif i <= 22:
        return 2
    else:
        return 3


op_cost = {1: 1, 2: 3.5, 3: 5.75}


def g(x, u, w):
    x_next = np.clip(x + w, 0, 25)
    proc = np.where(x_next > u, u, x_next)
    # proc = u if x_next > u else x_next
    return op_cost[u] - proc * 2


def f(x, u, w):
    return np.clip(x + w - u, 0, 25)

p = lambda w: stats.poisson.pmf(w, 2)
N = 26

# a
P = np.zeros((N, N))
for i in range(N):
    u = mu(i)
    for j in range(N):
        if j == N-1:
            P[i, j] = 1 - P[i].sum()
        elif j == 0:
            for t in range(i-u, 1):
                P[i, j] += p(t - i + u)
        else:
            w = j - i + u
            P[i, j] = p(w)

P[P<0] = 0
print((10*P).round())
fig = px.imshow(P)
fig.show()

# c
T = 100
w = stats.poisson.rvs(mu=2, size=T)
x = np.zeros(T+1)
for t in range(1, T+1):
    x[t] = f(x[t-1], mu(x[t-1]), w[t-1])

fig = go.Figure()
fig.add_trace(go.Line(y=x))
fig.add_trace(go.Line(y=w))
fig.show()

# d
g_tilde = np.zeros(N)
for i in range(N):
    g_tilde[i] = g(i, mu(i), 0) * p(0) + g(i, mu(i), 1) * (1 - p(0))

T = 100
v = np.zeros((T+1, N))
v[T] = g_tilde
for t in range(T-1, -1, -1):
    v[t] = g_tilde + P @ v[t+1]

px.line(y=v[0]).show()
px.imshow(v).show()

# e
eigval, eigvec = np.linalg.eig(P.T)
pi_ss = eigvec[:, 0] / eigvec[:, 0].sum()
px.line(pi_ss).show()
pass