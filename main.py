"""
Main experiment runner for QC-LDPC base-matrix GA.

Compares four initialisation strategies:
    aleatoria  - random base matrix
    peg        - PEG-inspired (BFS girth maximisation)
    qc         - QC-structured (cycle-4-free by construction)
    hibrida    - 1/3 of each
"""

import csv
import os
import time

import numpy as np

from config import Config
from ga.ga import algoritmo_genetico
from ga.fitness import fitness_B
from ldpc.constructors import expand_base_matrix, construir_G_sistematico
from ldpc.metrics import avaliar_H, calcular_girth, contar_ciclos_6
from utils.plot import (
    plot_convergencia,
    plot_convergencia_girth,
    plot_bler_vs_snr,
    plot_girth_histogram,
    plot_bler_boxplot,
    plot_girth_vs_bler,
    gerar_tabela_comparativa,
)

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

def rodar_experimento(config, tipo, n_execucoes=3):
    resultados = []
    historico_global = []

    os.makedirs("matrizes", exist_ok=True)

    for i in range(n_execucoes):
        print(f"\n{'='*60}")
        print(f"  Execucao {i+1}/{n_execucoes} | Tipo: {tipo}")
        print(f"{'='*60}")

        # use different seed per run
        cfg = Config()
        cfg.SEED = config.SEED + i

        start = time.time()
        melhor_B, historico = algoritmo_genetico(cfg, tipo)
        tempo_exec = time.time() - start

        np.save(f"matrizes/B_{tipo}_{i+1}.npy", melhor_B)

        for gen, h in enumerate(historico):
            historico_global.append({
                "tipo": tipo,
                "execucao": i + 1,
                "geracao": gen,
                "bler": h["bler"],
                "fitness": h["fitness"],
                "girth": h["girth"],
                "n_c6": h["n_c6"],
            })

        # --- full SNR sweep on best code ---
        H = expand_base_matrix(melhor_B, config.Z)
        G = construir_G_sistematico(H)
        girth = calcular_girth(H)
        n_c6 = contar_ciclos_6(H)

        for snr in config.SNR_dBs:
            bler = avaliar_H(
                H,
                G,
                snr,
                max(config.NUM_BLOCKS, 2000),
                config.MAX_ITERS,
                config.ALPHA,
            )

            resultados.append({
                "tipo": tipo,
                "execucao": i + 1,
                "snr": snr,
                "bler": bler if bler > 0 else 1 / max(config.NUM_BLOCKS, 2000),
                "girth": girth,
                "n_c6": n_c6,
                "tempo": tempo_exec,
            })

    return resultados, historico_global


# ------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------

def salvar_csv(rows, filename):
    if not rows:
        return

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    config = Config()

    tipos = ["aleatoria", "peg", "qc", "hibrida"]

    todos_resultados = []
    todos_historicos = []

    for tipo in tipos:
        res, hist = rodar_experimento(config, tipo, n_execucoes=2)
        todos_resultados.extend(res)
        todos_historicos.extend(hist)

    os.makedirs("resultados", exist_ok=True)

    salvar_csv(
        todos_resultados,
        "resultados/resultados.csv"
    )

    salvar_csv(
        todos_historicos,
        "resultados/convergencia.csv"
    )

    print("\nExperimentos finalizados! Gerando figuras...")

    conv_csv = "resultados/convergencia.csv"
    res_csv = "resultados/resultados.csv"

    # Fig. 1 - Convergencia (BLER + Fitness)
    plot_convergencia(conv_csv)

    # Fig. 1b - Convergencia do girth
    plot_convergencia_girth(conv_csv)

    # Fig. 2 - BLER vs Eb/N0
    plot_bler_vs_snr(res_csv)

    # Fig. 3 - Histograma de distribuicao de girth
    plot_girth_histogram(conv_csv)

    # Fig. 4 - Boxplot de BLER (robustez)
    plot_bler_boxplot(res_csv)

    # Fig. 5 - Scatter girth vs BLER (tradeoff)
    plot_girth_vs_bler(conv_csv)

    # Tab. 1 - Tabela comparativa
    gerar_tabela_comparativa(res_csv, conv_csv)


if __name__ == "__main__":
    main()