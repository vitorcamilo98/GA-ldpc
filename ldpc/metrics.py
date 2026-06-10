import numpy as np
from ldpc.encoder import codificar
from ldpc.channel import mapear_BPSK, awgn_noise, calc_LLR
from ldpc.decoder import normalized_min_sum

def avaliar_H(H, G, snr_db, num_blocos, max_iters, alpha):
    m, n = H.shape
    k = n - m

    erros = 0

    for _ in range(num_blocos):
        msg = np.random.randint(0, 2, k)
        cw = codificar(msg, G)
        x = mapear_BPSK(cw)
        y = x + awgn_noise(snr_db, n)
        llr = calc_LLR(y, snr_db)
        decoded = normalized_min_sum(H, llr, max_iters, alpha)

        if not np.array_equal(decoded, cw):
            erros += 1

    return erros / num_blocos


from ldpc.constructors import construir_H_de_A, construir_G_de_A

def avaliar_A(A, snr_db, num_blocos, max_iters, alpha):
    H = construir_H_de_A(A)
    G = construir_G_de_A(A)

    return avaliar_H(H, G, snr_db, num_blocos, max_iters, alpha)

def avaliar_estrutura(A):
    H = construir_H_de_A(A)
    girth = calcular_girth(H)
    return girth

def calcular_girth(H):
    import numpy as np

    m, n = H.shape
    min_cycle = float("inf")

    for v in range(n):
        visited = {v: 0}
        parent = {v: -1}
        queue = [(v, 0)]

        while queue:
            node, depth = queue.pop(0)

            if node < n:
                checks = np.where(H[:, node] == 1)[0] + n
                neighbors = checks
            else:
                var = node - n
                vars_ = np.where(H[var] == 1)[0]
                neighbors = vars_

            for neigh in neighbors:
                if neigh not in visited:
                    visited[neigh] = depth + 1
                    parent[neigh] = node
                    queue.append((neigh, depth + 1))

                # 🔥 correção aqui
                elif parent[node] != neigh:
                    cycle_len = visited[neigh] + depth + 1
                    if cycle_len > 2:
                        min_cycle = min(min_cycle, cycle_len)

    return min_cycle if min_cycle != float("inf") else 0

def score_girth(g):
    if g <= 4:
        return 0
    elif g == 6:
        return 0.5
    else:
        return 1.0
    
def score_bler(bler):
    return 1 / (bler + 1e-6)

def avaliar_A_completo(A, snr_db, num_blocos, max_iters, alpha):
    num_blocos = max(num_blocos, 1000)

    bler = avaliar_A(A, snr_db, num_blocos, max_iters, alpha)

    if bler == 0:
        bler = 1 / num_blocos

    return bler