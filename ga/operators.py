import numpy as np
from ldpc.constructors import check_rc_condition, repair_degree, repair_rc



def crossover(B1, B2, config):
    if np.random.rand() > config.CROSS_RATE:
        return B1.copy()
    
    Mb, Nb = B1.shape
    child = -np.ones((Mb, Nb), dtype=int)

    for j in range(Nb):
        if np.random.rand() < 0.5:
            child[:, j] = B1[:, j]
        else:
            child[:, j] = B2[:, j]
    return child

def crossover_row(A1, A2, config):
    if np.random.rand() > config.CROSS_RATE:
        return A1.copy()

    Mb, Nb = A1.shape
    child = -np.ones((Mb, Nb), dtype=int)

    for i in range(Mb):
        if np.random.rand() < 0.5:
            child[i, :] = A1[i, :]
        else:
            child[i, :] = A2[i, :]
    return child


def mutacao(A, config):
    r = np.random.rand()

    if r < 0.5:
        return mutacao_shift(A, config)
    elif r < 0.8:
        return mutacao_toggle(A, config)
    else:
        return mutacao_swap(A, config)

def mutacao_swap(A, config):
    A = A.copy()
    Mb, Nb = A.shape

    for j in range(Nb):
        if np.random.rand() >= config.MUT_RATE:
            continue
        i1, i2 = np.random.choice(Mb, 2, replace=False)
        A[i1, j], A[i2, j] = A[i2, j], A[i1, j]

    return A

def mutacao_shift(A, config):
    A = A.copy()
    Mb, Nb = A.shape

    for i in range(Mb):
        for j in range(Nb):
            if A[i, j] >= 0 and np.random.rand() < config.MUT_RATE:
                new_s = np.random.randint(0, config.Z)
                A[i, j] = new_s

    return A

def mutacao_toggle(A, config):
    A = A.copy()
    Mb, Nb = A.shape
    dv = config.dv

    for j in range(Nb):
        if np.random.rand() >= config.MUT_RATE:
            continue

        cw = int(np.sum(A[:, j] >= 0))

        if cw > dv:
            nonzeros = np.where(A[:, j] >= 0)[0]
            r = np.random.choice(nonzeros)
            A[r, j] = -1
        elif cw < dv:
            zeros = np.where(A[:, j] < 0)[0]
            if len(zeros) > 0:
                r = np.rand.choice(zeros)
                A[r, j] = np.random.randint(0, config.Z)
        else:
            nonzeros = np.where(A[:, j] >= 0)[0]
            zeros = np.where(A[:, j] < 0)[0]
            if len(nonzeros) > 0 and len(zeros):
                r_del = np.random.choice(nonzeros)
                r_add = np.random.choice(zeros)
                A[r_del, j] = -1
                A[r_add, j] = np.random.randint(0, config.Z)

    return A

def post_process(A, config):
    A = repair_degree(A, config.Z, config.dv, config.dc)
    A = repair_rc(A, config.Z)
    return A
