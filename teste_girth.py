import numpy as np
import matplotlib.pyplot as plt

from config import Config
from ldpc.constructors import construir_H_de_A
from ldpc.metrics import calcular_girth

# 🔹 seus geradores
from ldpc.constructors import (
    gerar_A_aleatoria,
    gerar_A_peg_like,
    gerar_A_qc_like,
    gerar_A_qc_melhorado,
    gerar_A_qc_tabela
)


GERADORES = {
    "aleatoria": gerar_A_aleatoria,
    "peg": gerar_A_peg_like,
    "qc": gerar_A_qc_like,
    "qc_melhorado": gerar_A_qc_melhorado,
    "qc_tabela": gerar_A_qc_tabela
}


def coletar_girths(config, gerar_func, n=100):
    girths = []

    for _ in range(n):
        A = gerar_func(config)
        H = construir_H_de_A(A)

        g = calcular_girth(H)
        girths.append(g)

    return girths


def plot_histogramas(resultados):
    plt.figure(figsize=(10, 6))

    bins = range(2, 12)  # ajustar conforme necessário

    for nome, girths in resultados.items():
        plt.hist(
            girths,
            bins=bins,
            alpha=0.5,
            label=nome,
            edgecolor="black"
        )

    plt.xlabel("Girth")
    plt.ylabel("Frequência")
    plt.title("Distribuição do Girth por Gerador")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    config = Config()

    resultados = {}

    for nome, gerar_func in GERADORES.items():
        print(f"\nTestando: {nome}")

        girths = coletar_girths(config, gerar_func, n=100)

        print(f"Média: {np.mean(girths):.2f}")
        print(f"Valores únicos: {sorted(set(girths))}")

        resultados[nome] = girths

    plot_histogramas(resultados)


if __name__ == "__main__":
    main()