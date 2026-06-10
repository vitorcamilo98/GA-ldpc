import numpy as np
import time

from config import Config
from ldpc.constructors import construir_H_de_A, construir_G_de_A
from ldpc.metrics import avaliar_H, calcular_girth

from ga.population import (
    gerar_A_aleatoria,
    gerar_A_peg_like,
    gerar_A_qc_like
)

# 🔹 mapeamento
GERADORES = {
    "aleatoria": gerar_A_aleatoria,
    "peg": gerar_A_peg_like,
    "qc": gerar_A_qc_like
}


def testar_uma_matriz(config, gerar_func, snr_db, num_blocos):
    A = gerar_func(config)

    H = construir_H_de_A(A)
    G = construir_G_de_A(A)

    # métricas estruturais
    grau_linhas = np.mean(np.sum(A, axis=1))
    grau_colunas = np.mean(np.sum(A, axis=0))
    girth = calcular_girth(H)

    # BLER
    bler = avaliar_H(
        H,
        G,
        snr_db,
        num_blocos,
        config.MAX_ITERS,
        config.ALPHA
    )

    # evitar zero artificial
    if bler == 0:
        bler = 1 / num_blocos

    return bler, girth, grau_linhas, grau_colunas


def benchmark(tipo="aleatoria", n_execucoes=2, snr_db=2.0, num_blocos=300):
    config = Config()

    if tipo not in GERADORES:
        raise ValueError(f"Tipo inválido: {tipo}")

    gerar_func = GERADORES[tipo]

    print(f"\n=== BENCHMARK: {tipo} ===")
    print(f"Execuções: {n_execucoes} | SNR: {snr_db} dB | Blocos: {num_blocos}\n")

    blers = []
    girths = []
    graus_linha = []
    graus_coluna = []
    tempos = []

    for i in range(n_execucoes):
        start = time.time()

        bler, girth, gl, gc = testar_uma_matriz(
            config, gerar_func, snr_db, num_blocos
        )

        tempo = time.time() - start

        blers.append(bler)
        girths.append(girth)
        graus_linha.append(gl)
        graus_coluna.append(gc)
        tempos.append(tempo)

        print(
            f"[{i+1}] BLER={bler:.6f} | girth={girth} | tempo={tempo:.2f}s"
        )

    print("\n--- RESULTADO FINAL ---")

    print(f"BLER médio: {np.mean(blers):.6f}")
    print(f"BLER std:   {np.std(blers):.6f}")

    print(f"Girth médio: {np.mean(girths):.2f}")
    print(f"Grau linhas: {np.mean(graus_linha):.2f}")
    print(f"Grau colunas: {np.mean(graus_coluna):.2f}")

    print(f"Tempo médio: {np.mean(tempos):.2f}s")

    return {
        "bler_mean": np.mean(blers),
        "bler_std": np.std(blers),
        "girth_mean": np.mean(girths),
        "tempo_mean": np.mean(tempos)
    }


if __name__ == "__main__":
    for tipo in ["aleatoria", "peg", "qc"]:
        benchmark(tipo, n_execucoes=10)