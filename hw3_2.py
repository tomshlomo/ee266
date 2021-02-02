import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

i = 1
j = 0
P = np.array([[.4, .3, 0, .3, 0, 0],
              [0, .4, 0, .3, .3, 0],
              [.3, 0, .1, 0, .3 , .3],
              [.3, 0, 0, .4, 0, .3],
              [0, .3, .3, .3, .1, 0],
              [0, 0, .3, 0, .3, .4]])
n = P.shape[0]
Q = P.copy()
Q[0, 0] = 1.
Q[0, 1:] = 0


def first_passage_time(i, j, Q, T=100, plot=1):
    t_vec = range(1, T)
    f = []
    for t in t_vec:
        f.append(np.linalg.matrix_power(Q, t)[i, j] - np.linalg.matrix_power(Q, t-1)[i, j])
    if plot:
        px.line(x=t_vec, y=f).show()
    return np.array(t_vec), np.array(f)



t_vec, f = first_passage_time(i, j, Q)

Q = np.block([[P, np.zeros((n,n))],
              [np.zeros((n,n)), P]])
Q[0, :n] = 0
Q[0, n:] = P[0, :]
Q[n,n] = 1
Q[n,n+1:] = 0
t_vec, s = first_passage_time(i, j+n, Q)

p_e = [np.linalg.matrix_power(P, t)[i, j] for t in t_vec]
fig = px.line(x=t_vec, y=[s+f, p_e], )
fig.update_traces(mode="lines+markers")
fig.show()

Q = np.block([[P, np.zeros((n,1))],[np.zeros((1,n+1))]])
Q[:,n] = Q[:,0]
Q[:,0] = 0
Q[n,n] = 1
Q[n, :n] = 0
print(Q)
first_passage_time(j, n, Q)
pass
