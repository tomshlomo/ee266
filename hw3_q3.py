import json
import numpy as np
import plotly.express as px


def reachable_states(A):
    n = A.shape[0]
    R = np.linalg.matrix_power(A + np.eye(n), n - 1) > 0
    return R


def communication_matrix(R):
    return R * R.T


def transient_vector(R, C):
    return np.any(R == C)


def topological_sort(A):
    A = A.copy()
    L = []
    S = {int(i) for i in np.where(np.all(A == 0, axis=0))[0]}
    while len(S):
        n = S.pop()
        L.append(n)
        outgoing, = np.where(A[n])
        for m in outgoing:
            if m == n:
                continue
            A[n, m] = 0
            if np.all(A[:, m] == 0):
                S.add(m)

    return np.array(L)


def get_calsses(R, C):
    N = R.shape[0]
    c, ind, state2class = np.unique(C, axis=0, return_index=True, return_inverse=True)
    A_class = R[np.ix_(ind, ind)]
    A_class[range(A_class.shape[0]), range(A_class.shape[0])] = False
    return state2class, A_class


def class_decomp(A):
    R = reachable_states(A)
    C = communication_matrix(R)

    state2class, A_class = get_calsses(R, C)

    is_transient = np.any(A_class, axis=1)
    transient_classes, = np.where(is_transient)
    recurrent_classes, = np.where(np.logical_not(is_transient))
    A_transient_class = A_class[np.ix_(transient_classes, transient_classes)]

    L = topological_sort(A_transient_class)
    L = transient_classes[L]
    L = np.concatenate([L, recurrent_classes])
    class2order = {l: i for i, l in enumerate(L)}

    state2order = [class2order[c] for c in state2class]
    permutation_vector = np.argsort(state2order)
    A = A[np.ix_(permutation_vector, permutation_vector)]

    return A


fp = open('class_decomposition_data.json')
d = json.load(fp)
P = np.array(d['P']['data'])
P1 = class_decomp(P)

px.imshow(P1 > 0).show()

pass
