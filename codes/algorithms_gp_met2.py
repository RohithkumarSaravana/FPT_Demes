import random
from deap import tools, gp
import math
import numpy as np


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring


def behavioural_diversity(population):
    """
    Behavioural diversity = number of unique prediction vectors / population size.
    Each individual's prediction vector is stored as individual.behaviour after
    fitness evaluation. Returns NaN if no individual has been evaluated yet.
    """
    behaviours = []
    for ind in population:
        if hasattr(ind, 'behaviour'):
            behaviours.append(tuple(ind.behaviour))
    if len(behaviours) == 0:
        return float('nan')
    return len(set(behaviours)) / len(population)


def eaSimple(population, toolbox, cxpb, mutpb, ngen, points_train,
             points_test=None, report_items=None, stats=None, halloffame=None):
    """
    Single-population GP.
    Records every generation: gen, nevals, best_train_fitness, best_ind_nodes,
                               behavioural_diversity
    Records last generation only: accuracy_test, best_phenotype
    """
    logbook = tools.Logbook()
    logbook.header = report_items if report_items else [
        'gen', 'nevals', 'best_train_fitness', 'best_ind_nodes',
        'behavioural_diversity', 'accuracy_test', 'best_phenotype'
    ]

    # Generation 0
    new_inds = [ind for ind in population if not ind.fitness.valid]
    n_evals = len(new_inds)
    for ind in new_inds:
        ind.fitness.values = toolbox.evaluate(ind, points_train)

    if halloffame is not None:
        halloffame.update(population)
        best_train_fitness = halloffame.items[0].fitness.values[0]
        best_ind_nodes = len(halloffame.items[0])
        print("gen =", 0, ", fitness =", best_train_fitness, ", evals =", n_evals)

    bd = behavioural_diversity(population)

    logbook.record(gen=0, nevals=n_evals,
                   best_train_fitness=best_train_fitness,
                   best_ind_nodes=best_ind_nodes,
                   behavioural_diversity=bd,
                   accuracy_test=float('nan'),
                   best_phenotype=float('nan'))

    # Generational loop
    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        new_inds = [ind for ind in offspring if not ind.fitness.valid]
        n_evals = len(new_inds)
        for ind in new_inds:
            ind.fitness.values = toolbox.evaluate(ind, points_train)

        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)
            best_train_fitness = halloffame.items[0].fitness.values[0]
            best_ind_nodes = len(halloffame.items[0])
            print("gen =", gen, ", fitness =", best_train_fitness, ", evals =", n_evals)

        bd = behavioural_diversity(population)

        if points_test and gen == ngen:
            _ = toolbox.evaluate(halloffame.items[0], points_test)
            accuracy_test = 1.0 - halloffame.items[0].mce
            best_phenotype = str(halloffame.items[0])
        else:
            accuracy_test = float('nan')
            best_phenotype = float('nan')

        logbook.record(gen=gen, nevals=n_evals,
                       best_train_fitness=best_train_fitness,
                       best_ind_nodes=best_ind_nodes,
                       behavioural_diversity=bd,
                       accuracy_test=accuracy_test,
                       best_phenotype=best_phenotype)

    return population, logbook


def eaSimple_demes(population, toolbox, cxpb, mutpb, ngen, points_train,
                   points_test=None, report_items=None, halloffame=None,
                   n_demes=10, migration_rate=0.05, migration_topology='circle'):
    """
    Island-model GP with circle migration topology.
    Behavioural diversity is measured over ALL demes combined (global population)
    after migration — not within each deme separately.
    """
    logbook = tools.Logbook()
    logbook.header = report_items if report_items else [
        'gen', 'nevals', 'best_train_fitness', 'best_ind_nodes',
        'behavioural_diversity', 'accuracy_test', 'best_phenotype'
    ]

    # Split population into demes
    pop_size = len(population)
    deme_size = pop_size // n_demes
    demes = []
    for d in range(n_demes):
        start = d * deme_size
        end = start + deme_size if d < n_demes - 1 else pop_size
        demes.append(population[start:end])

    n_migrants = max(1, round(deme_size * migration_rate))

    # Generation 0
    all_inds = [ind for deme in demes for ind in deme]
    new_inds = [ind for ind in all_inds if not ind.fitness.valid]
    n_evals = len(new_inds)
    for ind in new_inds:
        ind.fitness.values = toolbox.evaluate(ind, points_train)

    if halloffame is not None:
        halloffame.update(all_inds)
        best_train_fitness = halloffame.items[0].fitness.values[0]
        best_ind_nodes = len(halloffame.items[0])
        print("gen =", 0, ", fitness =", best_train_fitness, ", evals =", n_evals,
              f"[{n_demes} demes, mr={migration_rate}]")

    bd = behavioural_diversity(all_inds)

    logbook.record(gen=0, nevals=n_evals,
                   best_train_fitness=best_train_fitness,
                   best_ind_nodes=best_ind_nodes,
                   behavioural_diversity=bd,
                   accuracy_test=float('nan'),
                   best_phenotype=float('nan'))

    # Generational loop
    for gen in range(1, ngen + 1):
        n_evals = 0

        # Evolve each deme independently
        for d in range(n_demes):
            deme = demes[d]
            offspring = toolbox.select(deme, len(deme))
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            new_inds = [ind for ind in offspring if not ind.fitness.valid]
            n_evals += len(new_inds)
            for ind in new_inds:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
            demes[d] = offspring

        # Migration — circle topology
        if migration_topology == 'circle':
            emigrants = []
            for d in range(n_demes):
                sorted_deme = sorted(demes[d], key=lambda ind: ind.fitness.values[0])
                emigrants.append([toolbox.clone(ind) for ind in sorted_deme[:n_migrants]])
            for d in range(n_demes):
                receiver = (d + 1) % n_demes
                demes[receiver].sort(key=lambda ind: ind.fitness.values[0], reverse=True)
                for k, migrant in enumerate(emigrants[d]):
                    demes[receiver][k] = migrant
        else:
            raise ValueError(f"Unknown migration_topology: {migration_topology}")

        # Global HoF + diversity over full combined population
        all_inds = [ind for deme in demes for ind in deme]

        if halloffame is not None:
            halloffame.update(all_inds)
            best_train_fitness = halloffame.items[0].fitness.values[0]
            best_ind_nodes = len(halloffame.items[0])
            print("gen =", gen, ", fitness =", best_train_fitness, ", evals =", n_evals,
                  f"[{n_demes} demes, mr={migration_rate}]")

        bd = behavioural_diversity(all_inds)

        if points_test and gen == ngen:
            _ = toolbox.evaluate(halloffame.items[0], points_test)
            accuracy_test = 1.0 - halloffame.items[0].mce
            best_phenotype = str(halloffame.items[0])
        else:
            accuracy_test = float('nan')
            best_phenotype = float('nan')

        logbook.record(gen=gen, nevals=n_evals,
                       best_train_fitness=best_train_fitness,
                       best_ind_nodes=best_ind_nodes,
                       behavioural_diversity=bd,
                       accuracy_test=accuracy_test,
                       best_phenotype=best_phenotype)

    population[:] = [ind for deme in demes for ind in deme]
    return population, logbook
