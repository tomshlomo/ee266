import numpy as np
from plotly import express as px
import plotly.graph_objects as go

B = 10.
mu0 = 0.
mu1 = 0.1
sigma = 0.4
N = int(1e6)

p0 = np.exp(np.random.randn(N)*sigma + mu0)
p1 = np.exp(np.random.randn(N)*sigma + mu1)

# precient
alpha_precient = B * (p0 > p1)

# no knowledge
e0 = np.exp(mu0 + sigma**2/2)
e1 = np.exp(mu1 + sigma**2/2)
alpha_no = B * (e0 > e1) + np.zeros_like(alpha_precient)

# partial knowledge
alpha_partial = B * (p0 > e1)

# plot revenue
fig = go.Figure()
for alpha in [alpha_precient, alpha_no, alpha_partial]:
    rev = alpha * p0 + (B-alpha) * p1
    print(f'rev = {np.mean(rev)}')
    fig.add_trace(go.Histogram(x=rev, histnorm="probability density"))

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.5)
fig.show()