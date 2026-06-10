import numpy as np

def normalized_min_sum(H, LLRs, max_iters, alpha):
    m, n = H.shape
    Q = np.tile(LLRs, (m, 1))

    for _ in range(max_iters):
        R = np.zeros_like(Q)
        for c in range(m):
            v_idx = np.where(H[c] == 1)[0]
            for v in v_idx:
                outros = np.delete(v_idx, np.where(v_idx == v))
                if len(outros) == 0:
                    continue
                sinais = np.prod(np.sign(Q[c, outros]))
                minabs = np.min(np.abs(Q[c, outros]))
                R[c, v] = alpha * sinais * minabs

        for v in range(n):
            c_idx = np.where(H[:, v] == 1)[0]
            for c in c_idx:
                outros = np.delete(c_idx, np.where(c_idx == c))
                soma = np.sum(R[outros, v])
                Q[c, v] = LLRs[v] + soma

        LLR_apost = np.array([LLRs[v] + np.sum(R[np.where(H[:, v] == 1)[0], v]) for v in range(n)])
        hard = (LLR_apost < 0).astype(int)

        hard_col_vector = hard.reshape(-1, 1)
        syndrome = (H @ hard_col_vector) % 2
        if np.all(syndrome == 0):
            return hard

    return hard