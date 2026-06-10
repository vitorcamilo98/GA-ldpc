import numpy as np
import matplotlib.pyplot as plt


def media_por_geracao(resultados):
    historicos = [r["historico"] for r in resultados]
    return np.mean(historicos, axis=0)


def bler_medio(resultados):
    return np.mean([r["melhor_bler"] for r in resultados])


def bler_std(resultados):
    return np.std([r["melhor_bler"] for r in resultados])


def plot_convergencia(csv_path):
    import pandas as pd 
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    plt.figure()

    for tipo in df["tipo"].unique():
        sub = df[df["tipo"] == tipo]
        media = sub.groupby("geracao")["bler"].mean()

        plt.plot(media.index, media.values, label=tipo)

    plt.xlabel("Geração")
    plt.ylabel("Melhor BLER")
    plt.title("Convergência do Algoritmo Genético")
    plt.legend()
    plt.grid()
    plt.show()


def imprimir_resumo(todos_resultados):
    print("\n===== RESUMO DOS RESULTADOS =====\n")

    for tipo, resultados in todos_resultados.items():
        media = bler_medio(resultados)
        std = bler_std(resultados)

        print(f"{tipo}:")
        print(f"  BLER médio = {media:.6f}")
        print(f"  Desvio padrão = {std:.6f}\n")