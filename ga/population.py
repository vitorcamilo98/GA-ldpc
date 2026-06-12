import numpy as np
from collections import deque

from ldpc.constructors import check_rc_condition, col_weights, expand_base_matrix, repair_degree, repair_rc

def gerar_populacao_inicial(config, tipo):
    populacao = []

    return populacao

def gerar_B_aleatoria(config):
    Mb, Nb, Z, dv = config.Mb, config.Nb, config.Z, config.dv
    B = -np.ones((Mb, Nb), dtype=int)

    prob = dv / Mb
    for j in range(Nb):
        for i in range(Mb):
            if np.random.rand() < prob:
                B[i, j] = np.random.randint(0, Z)
    B = repair_degree(B, Z, dv, config.dc)
    return B

def _bfs_girth_from_var(H, v0, max_depth=20):
    m, n = H.shape
    visited = {v0: 0}
    parent = {v0: -1}
    queue = deque([(v0, 0)])
    min_cycle = float("inf")

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            break
        if node < n:
            neighbours = np.where(H[:, node] == 1)[0] +n
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
    return min_cycle

def gerar_B_peg(config):
    Mb, Nb, Z, dv = config.Mb, config.Nb, config.Z, config.dv
    B = -np.ones((Mb, Nb), dtype=int)
    row_load = np.zeros(Mb, dtype=int)

    for j in range(Nb):
        edges_placed = 0
        while edges_placed < dv:
            order = np.argsort(row_load + np.random.rand(Mb) * 0.01)
            best_row, best_shift, best_girth = -1, 0, -1

            for i in order:
                if B[i, j] >= 0:
                    continue
                for s in range(Z):
                    B[i, j] = s
                    if not check_rc_condition(B, Z, i, j, s):
                        B[i, j] = -1
                        continue

                    H = expand_base_matrix(B, Z)
                    min_g = float("inf")
                    for r in range(Z):
                        v = j * Z + r
                        g = _bfs_girth_from_var(H, v, max_depth=12)
                        min_g = min(min_g, g)

                    if min_g > best_girth:
                        best_girth = min_g
                        best_row = i
                        best_shift = s
                    
                    B[i, j] = -1
                    
                if best_girth == float("inf"):
                    break
            
            if best_row < 0:
                free = [i for i in order if B[i, j] < 0]
                if not free:
                    break
                best_row = free[0]
                best_shift = np.random.randint(0, Z)
            
            B[best_row, j] = best_shift
            row_load[best_row] += 1
            edges_placed +=1

    return B


def gerar_B_qc_structured(config):
    Mb, Nb, Z, dv = config.Mb, config.Nb, config.Z, config.dv
    B = -np.ones((Mb, Nb), dtype=int)

    for j in range(Nb):
        rw = np.sum(B >= 0, axis=1).astype(float)
        rw += np.random.rand(Mb) * 0.01
        rows = np.argsort(rw)[:dv]

        for i in rows:
            placed = False
            candidates = np.random.permutation(Z)
            for s in candidates:
                if check_rc_condition(B, Z, int(i), j, int(s)):
                    B[int(i), j] = int(s)
                    placed = True
                    break
            if not placed:
                B[int(i), j] = int(candidates[0])

    return B


def gerar_populacao_inicial(config, tipo):
    populacao = []

    generators = {
        "aleatoria": gerar_B_aleatoria,
        "peg": gerar_B_peg,
        "qc": gerar_B_qc_structured,
    }

    for idx in range(config.POP_SIZE):
        if tipo == "hibrida":
            r = idx % 3
            if r == 0:
                B =  gerar_B_aleatoria(config)
            elif r == 1:
                B = gerar_B_peg(config)
            else: 
                B = gerar_B_qc_structured(config)
        else:
            gen_func = generators[tipo]
            B = gen_func(config)

        populacao.append(B)
        
    return populacao