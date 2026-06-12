import numpy as np

from ldpc.metrics import avaliar_H, calcular_girth, contar_clicos_6
from ldpc.constructors import expand_base_matrix, construir_G_sistematico


def score_bler(bler):
    return max(0.0, 1.0 - bler)


def score_girth(girth, girth_max=12):
    if girth <= 4:
        return 0.0
    return min(1.0, (girth - 4) / max(1, girth_max - 4))

def score_cycles(n_cycles_6):
    return 1.0 / (1.0 + n_cycles_6)

def fitness_B(B, config):
    Z = config.Z
    H = expand_base_matrix(B, Z)
    G = construir_G_sistematico(H)

    m, n = H.shape
    k = n - m

    bler = avaliar_H(H, G, config.SNR_FITNESS, config.NUM_BLOCKS, config.MAX_ITERS, config.ALPHA)

    girth = calcular_girth(H)
    n_c6 = contar_clicos_6(H)

    f_bler = score_bler(bler)
    f_girth = score_girth(girth, config.GIRTH_MAX)
    f_c6 = score_cycles(n_c6)

    fitness = (
        config.W_BLER * f_bler
        + config.W_GIRTH * f_girth
        + config.W_CYCLES6 * f_c6
    )

    return fitness, bler, girth, n_c6

# def fitness_A(A, config):
#     bler = avaliar_A(
#         A,
#         config.SNR_FITNESS,
#         config.NUM_BLOCKS,
#         config.MAX_ITERS,
#         config.ALPHA
#     )

#     H = construir_H_de_A(A)
#     girth = calcular_girth(H)

#     s_bler = score_bler(bler)
#     s_girth = score_girth(girth)

#     fitness = 0.7 * s_bler + 0.3 * s_girth

#     return fitness, bler, girth