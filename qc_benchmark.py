import numpy as np
import time

from config import Config
from ldpc.constructors import construir_H_de_A, construir_G_de_A
from ldpc.metrics import avaliar_H, calcular_girth

from ldpc.constructors import gerar_A_qc_like
from ldpc.constructors import gerar_A_qc_melhorado
from ldpc.constructors import gerar_A_qc_tabela

def testar(config, gerar_func, snr_db, num_blocos):
    A = gerar_func(config)

    H = construir_H_de_A(A)
    G = construir_G_de_A(A)

    girth = calcular_girth(H)

    start = time.time()

    bler = avaliar_H(
        H,
        G,
        snr_db,
        num_blocos,
        config.MAX_ITERS,
        config.ALPHA
    )

    tempo = time.time() - start

    if bler == 0:
        bler = 1 / num_blocos

    return bler, girth, tempo


def benchmark_duplo(n_execucoes=10, snr_db=2.0, num_blocos=500):
    config = Config()

    metodos = {
        "qc_original": gerar_A_qc_like,
        "qc_melhorado": gerar_A_qc_melhorado,
        "qc_tabela": gerar_A_qc_tabela
    }

    resultados = {}

    for nome, func in metodos.items():
        print(f"\n=== {nome.upper()} ===")

        blers = []
        girths = []
        tempos = []

        for i in range(n_execucoes):
            bler, girth, tempo = testar(
                config, func, snr_db, num_blocos
            )

            blers.append(bler)
            girths.append(girth)
            tempos.append(tempo)

            print(
                f"[{i+1}] BLER={bler:.6f} | girth={girth} | tempo={tempo:.2f}s"
            )

        resultados[nome] = {
            "bler_mean": np.mean(blers),
            "bler_std": np.std(blers),
            "girth_mean": np.mean(girths),
            "tempo_mean": np.mean(tempos)
        }

        print("\n--- RESUMO ---")
        print(f"BLER médio: {resultados[nome]['bler_mean']:.6f}")
        print(f"BLER std:   {resultados[nome]['bler_std']:.6f}")
        print(f"Girth médio: {resultados[nome]['girth_mean']:.2f}")
        print(f"Tempo médio: {resultados[nome]['tempo_mean']:.2f}s")

    return resultados


if __name__ == "__main__":
    benchmark_duplo(n_execucoes=10)