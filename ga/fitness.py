from ldpc.metrics import avaliar_A, calcular_girth
from ldpc.constructors import construir_H_de_A


def score_bler(bler):
    return 1 / (bler + 1e-3)


def score_girth(girth):
    return girth


def fitness_A(A, config):
    bler = avaliar_A(
        A,
        config.SNR_FITNESS,
        config.NUM_BLOCKS,
        config.MAX_ITERS,
        config.ALPHA
    )

    H = construir_H_de_A(A)
    girth = calcular_girth(H)

    s_bler = score_bler(bler)
    s_girth = score_girth(girth)

    fitness = 0.7 * s_bler + 0.3 * s_girth

    return fitness, bler, girth