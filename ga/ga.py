import random
import numpy as np

from ga.operators import crossover, mutacao
from ga.population import gerar_populacao_inicial
from ldpc.constructors import construir_H_de_A, corrigir_A_grau_minimo
from ldpc.metrics import calcular_girth
from ga.fitness import fitness_A


def algoritmo_genetico(config, tipo_pop):
    populacao = gerar_populacao_inicial(config, tipo_pop)

    historico = []

    girths = []
    graus_linha = []
    graus_coluna = []

    for A in populacao:
        H = construir_H_de_A(A)

        girths.append(calcular_girth(H))
        graus_linha.append(np.mean(np.sum(A, axis=1)))
        graus_coluna.append(np.mean(np.sum(A, axis=0)))

    print("\n--- Análise população inicial ---")
    print(f"Tipo: {tipo_pop}")
    print(f"Girth médio: {np.mean(girths):.2f}")
    print(f"Grau médio linhas: {np.mean(graus_linha):.2f}")
    print(f"Grau médio colunas: {np.mean(graus_coluna):.2f}")
    print("-------------------------------\n")

    for gen in range(config.MAX_GEN):

        girth_pop = []
        for A in populacao:
            H = construir_H_de_A(A)
            g = calcular_girth(H)
            girth_pop.append((g, A))

        top_k = config.POP_SIZE // 2
        girth_pop.sort(reverse=True, key=lambda x: x[0])
        candidatos = girth_pop[:top_k]

        fitness = []

        for i, (girth, A) in enumerate(candidatos):
            fit, bler, _ = fitness_A(A, config)

            fitness.append((fit, bler, girth, A))

            print(
                f"Geração {gen+1}, Indivíduo {i+1}, "
                f"BLER={bler:.6f}, girth={girth}"
            )

        for girth, A in girth_pop[top_k:]:
            fit = 0.3 * girth
            fitness.append((fit, None, girth, A))

        fitness.sort(reverse=True, key=lambda x: x[0])

        melhor_fit, melhor_bler, melhor_girth, melhor_A = fitness[0]

        if melhor_bler is None:
            _, melhor_bler, _ = fitness_A(melhor_A, config)

        historico.append(melhor_bler)

        print(
            f"==> Melhor da geração {gen+1}: "
            f"BLER={melhor_bler:.6f}, girth={melhor_girth}\n"
        )

        novos = [melhor_A.copy()] 

        while len(novos) < config.POP_SIZE:
            p1, p2 = random.sample(fitness[:top_k], 2)

            filho = crossover(p1[3], p2[3], config)
            filho = mutacao(filho, config)
            filho = corrigir_A_grau_minimo(filho)

            novos.append(filho)

        populacao = novos

    return melhor_A, historico