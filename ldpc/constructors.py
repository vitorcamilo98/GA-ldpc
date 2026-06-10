import numpy as np

def gerar_H(m, n):
    H = np.zeros((m, n), dtype=int)
    for j in range(n):
        rows = np.random.choice(m, size=np.random.randint(2, 4), replace=False)
        H[rows, j] = 1
    return H

def gerar_A_aleatoria(config):
    A = np.zeros((config.m, config.k), dtype=int)

    for col in range(config.k):
        linhas = np.random.choice(config.m, size=config.DEGREE, replace=False)
        A[linhas, col] = 1

    return A

def gerar_A_peg_like(config):
    A = np.zeros((config.m, config.k), dtype=int)
    grau_linhas = np.zeros(config.m)

    for col in range(config.k):
        grau_col = config.DEGREE  

        for _ in range(grau_col):
            candidatos = np.argsort(grau_linhas)

            escolhido = None

            for lin in candidatos:
                if A[lin, col] == 1:
                    continue

                vizinhos = np.where(A[lin, :] == 1)[0]

                conflito = False
                for c in vizinhos:
                    if np.any(A[:, c] & A[:, col]):
                        conflito = True
                        break

                if not conflito:
                    escolhido = lin
                    break

            if escolhido is None:
                livres = np.where(A[:, col] == 0)[0]
                escolhido = np.random.choice(livres)

            A[escolhido, col] = 1
            grau_linhas[escolhido] += 1

    return A

def gerar_A_qc_like(config, Z=4):
    A = np.zeros((config.m, config.k), dtype=int)

    for col in range(config.k):
        blocos = np.random.choice(config.m // Z, size=config.DEGREE, replace=False)

        for bloco in blocos:
            i = bloco * Z

            shift = np.random.randint(0, Z)
            row = i + shift
            A[row, col] = 1

    return A

def derivar_G(H):
    m, n = H.shape
    k = n - m
    A = np.random.randint(0, 2, (k, m))
    G = np.concatenate([np.eye(k, dtype=int), A], axis=1) % 2
    return G

def construir_H_de_A(A):
    m, k = A.shape
    H = np.concatenate([A, np.eye(m, dtype=int)], axis=1)
    return H % 2

def construir_G_de_A(A):
    m, k = A.shape
    G = np.concatenate([np.eye(k, dtype=int), A.T], axis=1)
    return G % 2

def corrigir_A_grau_minimo(A, grau_min_linha=1, grau_min_coluna=2):
    A = A.copy()
    m, k = A.shape
    for i in range(m):
        while np.sum(A[i, :]) < grau_min_linha:
            col = np.random.randint(0, k)
            A[i, col] = 1
    for j in range(k):
        while np.sum(A[:, j]) < grau_min_coluna:
            lin = np.random.randint(0, m)
            A[lin, j] = 1

    return A

def gerar_A_qc_melhorado(config, Z=4):
    Mb = config.m // Z
    Nb = config.k // Z

    A = np.zeros((config.m, config.k), dtype=int)

    shifts = {}

    for i in range(Mb):
        for j in range(Nb):

            # garante grau médio por coluna
            if np.random.rand() < (config.DEGREE / Nb):

                tentativas = 0
                while True:
                    shift = np.random.randint(0, Z)

                    valido = True

                    for (i2, j2), shift2 in shifts.items():

                        # precisa formar um retângulo
                        if i == i2 or j == j2:
                            continue

                        # 🔥 só testa se ambos existem
                        if (i, j2) not in shifts or (i2, j) not in shifts:
                            continue

                        shift_ij2 = shifts[(i, j2)]
                        shift_i2j = shifts[(i2, j)]

                        d1 = (shift - shift_ij2) % Z
                        d2 = (shift2 - shift_i2j) % Z

                        if d1 == d2:
                            valido = False
                            break

                    if valido:
                        shifts[(i, j)] = shift
                        break

                    tentativas += 1

                    if tentativas > 20:
                        shifts[(i, j)] = shift
                        break

                for r in range(Z):
                    A[i*Z + r, j*Z + (r + shift) % Z] = 1

    return A

BASE = [
    [0, -1, 1, 2],
    [2,  0, -1, 1],
    [1,  2,  0, -1]
]

def gerar_A_qc_tabela(config, Z=4):
    Mb = len(BASE)
    Nb = len(BASE[0])

    A = np.zeros((Mb*Z, Nb*Z), dtype=int)

    for i in range(Mb):
        for j in range(Nb):
            shift = BASE[i][j]

            if shift == -1:
                continue

            for r in range(Z):
                A[i*Z + r, j*Z + (r + shift) % Z] = 1

    return A

