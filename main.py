import numpy as np
import csv
import time
import os

from ga.fitness import fitness_A
from ga.ga import algoritmo_genetico
from ldpc.constructors import construir_H_de_A, construir_G_de_A
from ldpc.metrics import avaliar_A_completo, avaliar_H, avaliar_estrutura
from utils.plot import plot_convergencia

def rodar_experimento(config, tipo, n_execucoes=3):
    resultados = []
    historico_global = []

    os.makedirs("matrizes", exist_ok=True)

    for i in range(n_execucoes):
        print(f"\n--- Execução {i+1} | Tipo: {tipo} ---")

        start = time.time()

        melhor_A, historico = algoritmo_genetico(config, tipo)

        tempo_exec = time.time() - start

        np.save(f"matrizes/A_{tipo}_{i+1}.npy", melhor_A)

        for gen, bler in enumerate(historico):
            historico_global.append({
                "tipo": tipo,
                "execucao": i + 1,
                "geracao": gen,
                "bler": bler
            })


        for snr in config.SNR_dBs:
            bler = avaliar_A_completo(
                melhor_A,
                snr,
                config.NUM_BLOCKS,
                config.MAX_ITERS,
                config.ALPHA
            )
            girth = avaliar_estrutura(melhor_A)
            fitness = fitness_A(melhor_A, config)

            resultados.append({
                "tipo": tipo,
                "execucao": i + 1,
                "snr": snr,
                "bler": bler,
                "girt": girth,
                "fitness": fitness,
                "tempo_execucao": tempo_exec
            })

    return resultados, historico_global


def salvar_csv(resultados, filename="resultados.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)


def main():
    from config import Config
    config = Config()

    tipos = ["aleatoria", "peg", "qc", "hibrida"]

    todos_resultados = []
    todos_historicos = []

    for tipo in tipos:
        res, hist = rodar_experimento(config, tipo, n_execucoes=2)
        todos_resultados.extend(res)
        todos_historicos.extend(hist)

    os.makedirs("resultados", exist_ok=True)
    salvar_csv(todos_resultados, "resultados/resultados.csv")
    salvar_csv(todos_historicos, "resultados/convergencia.csv")

    print("\nExperimentos finalizados!")
    plot_convergencia("resultados/convergencia.csv")

if __name__ == "__main__":
    main()