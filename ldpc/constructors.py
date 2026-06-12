import numpy as np

def expand_base_matrix(B, Z):
    Mb, Nb = B.shape
    H = np.zeros((Mb * Z, Nb * Z), dtype=int)
    for i in range(Mb):
        for j in range(Nb):
            s = B[i, j]
            if s < 0:
                continue
            for r in range(Z):
                H[i * Z + r, j * Z + (r + s) % Z] = 1
    return H

def construir_G_sistematico(H):
    m, n = H.shape
    k = n - m
    
    A = H.astype(int).copy()
    col_perm = np.arrange(n)

    pivot_row = 0
    for pr in range(m):
        found = False
        for c in range(pivot_row, n):
            if A[pr, c] == 1:
                if c != pivot_row:
                    A[:, [pivot_row, c]] = A[:, [c, pivot_row]]
                    col_perm[[pivot_row, c]] = col_perm[[c, pivot_row]]
                found = True
                break
        if not found:
            for r2 in range(pr + 1, m):
                for c in range(pivot_row, n):
                    if A[r2, c] == 1:
                        A[[pr, r2]] = A[[r2, pr]]
                        if c != pivot_row:
                            A[:, [pivot_row, c]] = A[:, [c, pivot_row]]
                            col_perm[[pivot_row, c]] = col_perm[[c, pivot_row]]
                        found = True
                        break
                if found: 
                    break
        if not found:
            continue

        for r in range(m):
            if r != pr and A[r, pivot_row] == 1:
                A[r] = (A[r] ^ A[pr])
        pivot_row += 1
    
    P = A[:, m:]
    G_perm = np.concatenate([P.T, np.eye(k, dtype=int)], axis=1) % 2

    inv_perm = np.argsort(col_perm)
    G = G_perm[:, inv_perm]

    return G


def check_rc_condition(B, Z, i, j, shift):
    Mb, Nb = B.shape
    for i2 in range(Mb):
        if i2 == i:
            continue
        if B[i2, j] < 0:
            continue
        for j2 in range(Nb):
            if j2 == j:
                continue
            if B[i, j2] < 0 or B[i2, j2] < 0:
                continue
            d1 = (shift - B[i, j2]) % Z
            d2 = (B[i2, j] - B[i2, j2]) % Z
            if d1 == d2:
                return False
    return True


def repair_rc(B, Z, max_attempts=50):
    Mb, Nb = B.shape 
    B = B.copy()
    for i in range(Mb):
        for j in range(Nb):
            if B[i, j] < 0:
                continue
            if not check_rc_condition(B, Z, i, j, B[i, j]):
                fixed = False
                for _ in range(max_attempts):
                    s = np.random.randint(0, Z)
                    if check_rc_condition(B, Z, i, j, s):
                        B[i, j] = s
                        fixed = True
                        break
                if not fixed:
                    B[i, j] = np.random.randint(0, Z)
    return B


def col_weights(B): 
    return np.sum(B >= 0, axis=0)


def row_weights(B):
    return np.sum(B >= 0, axis=1)

def repair_degree(B, Z, dv, dc):
    B = B.copy()
    Mb, Nb = B.shape

    for j in range(Nb):
        w = int(np.sum(B[:, j] >= 0))
        if w < dv:
            zeros = np.where(B[:, j] < 0)[0]
            rw = row_weights(B)
            zeros = sorted(zeros, key=lambda r: rw[r])
            for r in zeros[: dv - w]:
                s = np.random.randint(0, Z)
                B[r, j] = s
        elif w > dv: 
            nonzeros = np.where(B[:, j] >= 0)[0]
            rw = row_weights(B)
            nonzeros = sorted(nonzeros, key=lambda r: -rw[r])
            for r in nonzeros[: w - dv]:
                B[r, j] = -1
    return B

