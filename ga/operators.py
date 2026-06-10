import copy
import random
import config
import numpy as np

def crossover(A1, A2, config):
    return crossover_coluna(A1, A2, config)

def crossover_coluna(A1, A2, config):
    if np.random.rand() > config.CROSS_RATE:
        return A1.copy()

    m, k = A1.shape
    child = np.zeros_like(A1)

    for col in range(k):
        if np.random.rand() < 0.5:
            child[:, col] = A1[:, col]
        else:
            child[:, col] = A2[:, col]

    return child

def crossover_corrigido(A1, A2, config):
    child = crossover_coluna(A1, A2, config)

    # corrigir grau das colunas
    for col in range(child.shape[1]):
        target = np.sum(A1[:, col])  # mantém grau do pai 1
        current = np.sum(child[:, col])

        if current > target:
            ones = np.where(child[:, col] == 1)[0]
            to_zero = np.random.choice(ones, current - target, replace=False)
            child[to_zero, col] = 0

        elif current < target:
            zeros = np.where(child[:, col] == 0)[0]
            to_one = np.random.choice(zeros, target - current, replace=False)
            child[to_one, col] = 1

    return child

def mutacao(A, config):
    r = np.random.rand()

    if r < 0.5:
        return mutacao_swap(A, config)
    elif r < 0.8:
        return mutacao_local(A, config)
    else:
        return mutacao_coluna(A, config)

def mutacao_swap(A, config):
    A = A.copy()
    m, k = A.shape

    for col in range(k):
        if np.random.rand() < config.MUT_RATE:
            ones = np.where(A[:, col] == 1)[0]
            zeros = np.where(A[:, col] == 0)[0]

            if len(ones) > 0 and len(zeros) > 0:
                i = np.random.choice(ones)
                j = np.random.choice(zeros)

                # troca
                A[i, col] = 0
                A[j, col] = 1

    return A

def mutacao_coluna(A, config):
    A = A.copy()
    m, k = A.shape

    for col in range(k):
        if np.random.rand() < config.MUT_RATE:
            A[:, col] = np.random.permutation(A[:, col])

    return A

def mutacao_local(A, config):
    A = A.copy()
    m, k = A.shape

    if np.random.rand() < config.MUT_RATE:
        col = np.random.randint(0, k)

        ones = np.where(A[:, col] == 1)[0]
        zeros = np.where(A[:, col] == 0)[0]

        if len(ones) > 0 and len(zeros) > 0:
            i = np.random.choice(ones)
            j = np.random.choice(zeros)

            A[i, col] = 0
            A[j, col] = 1

    return A


