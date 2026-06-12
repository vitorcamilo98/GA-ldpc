import numpy as np
from collections import deque

from ldpc.encoder import codificar
from ldpc.channel import mapear_BPSK, awgn_noise, calc_LLR
from ldpc.decoder import normalized_min_sum

def avaliar_H(H, G, snr_db, num_blocos, max_iters, alpha):
    m, n = H.shape
    k = n - m
    rate = k / n

    erros = 0

    for _ in range(num_blocos):
        msg = np.random.randint(0, 2, k)
        cw = codificar(msg, G)
        x = mapear_BPSK(cw)
        y = x + awgn_noise(snr_db, n, rate)
        llr = calc_LLR(y, snr_db, rate)
        decoded = normalized_min_sum(H, llr, max_iters, alpha)

        if not np.array_equal(decoded, cw):
            erros += 1

    return erros / num_blocos


# from ldpc.constructors import construir_H_de_A, construir_G_de_A

# def avaliar_A(A, snr_db, num_blocos, max_iters, alpha):
#     H = construir_H_de_A(A)
#     G = construir_G_de_A(A)

#     return avaliar_H(H, G, snr_db, num_blocos, max_iters, alpha)

# def avaliar_estrutura(A):
#     H = construir_H_de_A(A)
#     girth = calcular_girth(H)
#     return girth

def calcular_girth(H):
    m, n = H.shape
    min_cycle = float("inf")

    for v in range(n):
        visited = {v: 0}
        parent = {v: -1}
        queue = deque([(v, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= min_cycle:
                break

            if node < n:
                neighbours = np.where(H[:, node] == 1)[0] + n
            else:
                neighbours = np.where(H[node - n] == 1)[0]

            for nb in neighbours:
                if nb not in visited:
                    visited[nb] = depth + 1
                    parent[nb] = node
                    queue.append((nb, depth + 1))
                elif parent[node] != nb:
                    cycle_len = visited[nb] + depth + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len

    return min_cycle if min_cycle != float("inf") else 0

def contar_clicos_6(H):
    m, n = H.shape
    cn_to_vn = [set(np.where(H[c] == 1)[0]) for c in range(m)]
    count = 0

    for c1 in range(m):
        for c2 in range(c1 + 1, m):
            shared_12 = cn_to_vn[c1] & cn_to_vn[c2]
            if len(shared_12) < 2:
                continue
            for c3 in range(c2 + 1, m):
                shared_13 = cn_to_vn[c1] & cn_to_vn[c3]
                shared_23 = cn_to_vn[c2] & cn_to_vn[c3]
                if not shared_13 or not shared_23:
                    continue
                for v12 in shared_12:
                    for v23 in shared_23:
                        if v23 == v12:
                            continue
                        for v13 in shared_13:
                            if v13 == v12 or v13 == v23:
                                continue
                            count += 1
    return count

def avaliar_estrutura(H):
    return calcular_girth(H)

# def score_girth(g):
#     if g <= 4:
#         return 0
#     elif g == 6:
#         return 0.5
#     else:
#         return 1.0
    
# def score_bler(bler):
#     return 1 / (bler + 1e-6)

# def avaliar_A_completo(A, snr_db, num_blocos, max_iters, alpha):
#     num_blocos = max(num_blocos, 1000)

#     bler = avaliar_A(A, snr_db, num_blocos, max_iters, alpha)

#     if bler == 0:
#         bler = 1 / num_blocos

#     return bler