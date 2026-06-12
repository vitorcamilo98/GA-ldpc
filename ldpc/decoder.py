import numpy as np

def normalized_min_sum(H, LLRs, max_iters, alpha):
    m, n = H.shape
    cn_to_vn = [np.where(H[c] == 1)[0] for c in range(m)]
    vn_to_cn = [np.where(H[:, v] == 1)[0] for v in range(n)]

    R = np.zeros((m, n), dtype=float)

    for _ in range(max_iters):
        R_new = np.zeros((m, n), dtype=float)
        for c in range(m):
            vns = cn_to_vn[c]
            if len(vns) < 2:
                continue
            Q_cv = np.array([
                LLRs[v] + np.sum(R[vn_to_cn[v], v]) - R[c, v]
                for v in vns
            ])
            signs = np.sign(Q_cv)
            abs_vals = np.abs(Q_cv)
            total_sign = np.prod(signs)

            for idx, v in enumerate(vns):
                others_abs = np.delete(abs_vals, idx)
                min_abs = np.min(others_abs) if len(others_abs) > 0 else 0.0
                R_new[c, v] = alpha * total_sign * signs[idx] * min_abs
        R = R_new


        LLR_apost = np.array([
            LLRs[v] + np.sum(R[vn_to_cn[v], v]) 
            for v in range(n)
            ])
        hard = (LLR_apost < 0).astype(int)

        syndrome = np.zeros(m, dtype=int)
        for c in range(m):
            syndrome[c] = np.sum(hard[cn_to_vn[c]]) % 2
        if np.all(syndrome == 0):
            return hard
    return hard