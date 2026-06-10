from ldpc.constructors import corrigir_A_grau_minimo, gerar_A_aleatoria, gerar_A_peg_like, gerar_A_qc_like, gerar_A_qc_melhorado, gerar_A_qc_tabela
import numpy as np

def gerar_populacao_inicial(config, tipo):
    populacao = []

    for _ in range(config.POP_SIZE):
        if tipo == "aleatoria":
            A = gerar_A_aleatoria(config)
        elif tipo == "peg":
            A = gerar_A_peg_like(config)
        elif tipo == "qc":
            A = gerar_A_qc_melhorado(config)
        elif tipo == "hibrida":
            if np.random.rand() < 0.5:
                A = gerar_A_peg_like(config)
            else:
                A = gerar_A_qc_melhorado(config)

        A = corrigir_A_grau_minimo(A)

        populacao.append(A)

    return populacao