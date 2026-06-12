import random
import numpy as np

from ga.operators import crossover, mutacao, post_process
from ga.population import gerar_populacao_inicial
from ga.fitness import fitness_B

def _tournament_select(pop_fitness, k):
    indices = random.sample(range(len(pop_fitness)), k)
    best = max(indices, key=lambda i: pop_fitness[i][0])
    return best


def algoritmo_genetico(config, tipo_pop):
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    populacao = gerar_populacao_inicial(config, tipo_pop)

    n_elite = max(1, int(config.ELITE_FRAC * config.POP_SIZE))
    historico = []
    
    for gen in range(config.MAX_GEN):
        pop_fitness = []
        for B in populacao:
            fit, bler, girth, n_c6 = fitness_B(B, config)
            pop_fitness.append((fit, bler, girth, n_c6, B))

        pop_fitness.sort(reverse=True, key=lambda x: x[0])

        best_fit, best_bler, best_girth, best_c6, best_B = pop_fitness[0]
        historico.append({
            "fitness": best_fit,
            "bler": best_bler,
            "girth": best_girth,
            "n_c6": best_c6,
        })

        print(
            f"Gen {gen+1:3d}/{config.MAX_GEN} | "
            f"fitness={best_fit:.4f} BLER={best_bler:.6f}  "
            f"girth={best_girth} cycles6={best_c6}"
        )

        novos = [pf[4].copy() for pf in pop_fitness[:n_elite]]
        while len(novos) < config.POP_SIZE:
            i1 = _tournament_select(pop_fitness, config.TOURNAMENT_K)
            i2 = _tournament_select(pop_fitness, config.TOURNAMENT_K)

            child = crossover(pop_fitness[i1][4], pop_fitness[i2][4], config)
            child = mutacao(child, config)
            child = post_process(child, config)

            novos.append(child)
        
        populacao = novos

    return best_B, historico