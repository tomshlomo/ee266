import math
import numpy as np
import scipy.io

mat = scipy.io.loadmat('simple_system_data.mat')

T = mat['T'][0]
U_seq = mat['U_seq'].T
W_seq = mat['W_seq'].T
a = mat['a'][0]
ct = mat['ct'][:, 0]
p = mat['p'][0]
phi_cl = mat['phi_cl'][:, 0]
pw = mat['pw'][:, 0]
x0 = mat['x0'][0]
xf = mat['xf'][0]


def cost(u, w):
    c = 0.
    x = x0
    for t in range(int(T)):
        if x == u[t] and u[t] == 1 and w[t] == 1:
            c = c - a
        elif x == 1 and u[t] == 2:
            c = c + (t+1) ** 2
        x = u[t]
    if x == 1:
        c = c + np.inf
    return c

## a
ec = []
for u in U_seq:
    c = np.array([cost(u, w) for w in W_seq])
    ec.append(np.dot(pw, c))

ec = np.array(ec)
best_u = U_seq[np.argmin(ec)]
best_ec = np.min(ec)
print(best_u)
print(best_ec)

## b
for w in W_seq:
    c = np.array([cost(u, w) for u in U_seq])
    print(f"best u for w={w} is {U_seq[np.argmin(c)]}, with cost {c.min()}")

## c
ec = []
for phi in phi_cl:
    c = []
    for w in W_seq:
        u = []
        x = x0
        for t in range(int(T)):
            u.append(phi[x-1, w[t]-1, t])
            x = u[t]
        c.append(cost(np.array(u), w))
    ec.append(np.dot(np.array(c), pw))

ec = np.array(ec)
print(f"best policy is {ec.argmin()} with expected cost of {ec.min()}")

## d

pass
