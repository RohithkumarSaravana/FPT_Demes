"""
algorithms_gp_final.py — GP evolutionary loops.

Provides the two generational algorithms used across the experiments:
  - eaSimple: a standard single-population, tournament-style GP loop.
  - eaSimple_demes: an island-model (demes) variant with periodic circle
    migration between sub-populations.

Both record per-generation statistics (best training fitness, best
individual's node count, behavioural diversity) plus final-generation
test accuracy and best phenotype, via a DEAP Logbook.
"""

import random
from deap import tools, gp
import math
import numpy as np


def varAnd(population, toolbox, cxpb, mutpb):
    """
    Apply crossover and mutation to a cloned copy of the population
    (standard DEAP varAnd): for each adjacent pair of individuals, mate
    them with probability cxpb; independently, mutate each individual
    with probability mutpb. Fitness is invalidated (deleted) on any
    individual that was mated or mutated, so it gets re-evaluated by the
    caller.

    Parameters:
        population -- list of individuals to vary.
        toolbox    -- DEAP toolbox providing clone/mate/mutate operators.
        cxpb       -- probability of mating each adjacent pair.
        mutpb      -- probability of mutating each individual.

    Returns:
        New list of (possibly varied) individuals, same length as population.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring


def behavioural_diversity(population):
    """
    Behavioural diversity = unique prediction vectors / population size.
    Each individual stores individual.behaviour (list of predicted class labels)
    after fitness evaluation. Returns NaN if no individual has been evaluated.
    """
    behaviours = []
    for ind in population:
        if hasattr(ind, 'behaviour'):
            behaviours.append(tuple(ind.behaviour))
    if not behaviours:
        return float('nan')
    return len(set(behaviours)) / len(population)


def eaSimple(population, toolbox, cxpb, mutpb, ngen, points_train,
             points_test=None, report_items=None, stats=None, halloffame=None):
    """
    Single-population tournament GP.
    Records every generation : gen, nevals, best_train_fitness,
                               best_ind_nodes, behavioural_diversity
    Records last gen only    : accuracy_test, best_phenotype

    Parameters:
        population    -- initial population of individuals.
        toolbox       -- DEAP toolbox with select/mate/mutate/evaluate/clone
                         registered.
        cxpb, mutpb   -- crossover and mutation probabilities (see varAnd).
        ngen          -- number of generations to run.
        points_train  -- training data passed through to toolbox.evaluate.
        points_test   -- optional test data; if given, the best individual
                         (from the hall of fame) is evaluated on it after
                         the final generation to record accuracy_test.
        report_items  -- optional list of column names for the logbook
                         header (defaults to the standard set below).
        stats         -- unused (accepted for interface compatibility).
        halloffame    -- DEAP HallOfFame (or similar) updated each
                         generation; required for best_train_fitness /
                         best_ind_nodes / accuracy_test to be recorded.

    Returns:
        Tuple of (final population, DEAP Logbook of per-generation stats).
    """
    logbook = tools.Logbook()
    logbook.header = report_items if report_items else [
        'gen', 'nevals', 'best_train_fitness', 'best_ind_nodes',
        'behavioural_diversity', 'accuracy_test', 'best_phenotype'
    ]

    # ── Gen 0 ─────────────────────────────────────────────────────────────────
    new_inds = [ind for ind in population if not ind.fitness.valid]
    n_evals  = len(new_inds)
    for ind in new_inds:
        ind.fitness.values = toolbox.evaluate(ind, points_train)

    if halloffame is not None:
        halloffame.update(population)
        best_train_fitness = halloffame.items[0].fitness.values[0]
        best_ind_nodes     = len(halloffame.items[0])
        print("gen =", 0, ", fitness =", best_train_fitness, ", evals =", n_evals)

    bd = behavioural_diversity(population)
    logbook.record(gen=0, nevals=n_evals,
                   best_train_fitness=best_train_fitness,
                   best_ind_nodes=best_ind_nodes,
                   behavioural_diversity=bd,
                   accuracy_test=float('nan'),
                   best_phenotype=float('nan'))

    # ── Loop ──────────────────────────────────────────────────────────────────
    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        new_inds  = [ind for ind in offspring if not ind.fitness.valid]
        n_evals   = len(new_inds)
        for ind in new_inds:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)
            best_train_fitness = halloffame.items[0].fitness.values[0]
            best_ind_nodes     = len(halloffame.items[0])
            print("gen =", gen, ", fitness =", best_train_fitness, ", evals =", n_evals)

        bd = behavioural_diversity(population)

        if points_test and gen == ngen:
            _ = toolbox.evaluate(halloffame.items[0], points_test)
            accuracy_test  = 1.0 - halloffame.items[0].mce
            best_phenotype = str(halloffame.items[0])
        else:
            accuracy_test  = float('nan')
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
                   n_demes=10, migration_rate=0.05, migration_topology='circle',
                   frequency_parameter=1):
    """
    Island-model GP with circle migration.

    Parameters
    ----------
    n_demes              : number of islands
    migration_rate       : fraction of each deme that migrates when migration fires
    migration_topology   : 'circle' — deme d sends migrants to deme (d+1) % n_demes
    frequency_parameter  : migration fires every `frequency_parameter` generations.
                           1 = every generation (original behaviour)
                           5 = every 5 generations
                          10 = every 10 generations
                          25 = every 25 generations

    Behavioural diversity is measured over ALL demes merged (global population)
    AFTER migration fires (or at every generation when migration does not fire,
    still measured on the full merged population).

    Migration logic:
    - Emigrants are selected from the ORIGINAL demes before any receiving deme
      is modified — so deme B's emigrants are from the original B, not the B
      that already received from A.
    - Best n_migrants individuals (lowest RMSE) are sent.
    - They replace the worst n_migrants in the receiving deme (elitist replacement).

    Parameters:
        population           -- initial population, split evenly into
                                n_demes sub-populations.
        toolbox               -- DEAP toolbox with select/mate/mutate/
                                evaluate/clone registered.
        cxpb, mutpb            -- crossover and mutation probabilities
                                (see varAnd), applied within each deme.
        ngen                   -- number of generations to run.
        points_train           -- training data passed through to
                                toolbox.evaluate.
        points_test            -- optional test data; if given, the best
                                individual (from the hall of fame) is
                                evaluated on it after the final
                                generation to record accuracy_test.
        report_items           -- optional list of column names for the
                                logbook header (defaults to the standard
                                set below).
        halloffame             -- DEAP HallOfFame (or similar), updated
                                each generation from all demes merged.
        n_demes                -- number of islands to split the
                                population into.
        migration_rate         -- fraction of a deme's individuals sent
                                as emigrants when migration fires.
        migration_topology     -- migration graph; only 'circle' is
                                implemented.
        frequency_parameter    -- migration fires every this many
                                generations.

    Returns:
        Tuple of (final population, DEAP Logbook of per-generation stats).
    """
    logbook = tools.Logbook()
    logbook.header = report_items if report_items else [
        'gen', 'nevals', 'best_train_fitness', 'best_ind_nodes',
        'behavioural_diversity', 'accuracy_test', 'best_phenotype'
    ]

    # ── Split into demes ───────────────────────────────────────────────────────
    pop_size  = len(population)
    deme_size = pop_size // n_demes
    demes = []
    for d in range(n_demes):
        start = d * deme_size
        end   = start + deme_size if d < n_demes - 1 else pop_size
        demes.append(population[start:end])

    n_migrants = max(1, round(deme_size * migration_rate))
    print(f"  deme_size={deme_size}, n_migrants={n_migrants} "
          f"(mr={migration_rate*100:.0f}%), migration every {frequency_parameter} gen(s)")

    # ── Gen 0 ─────────────────────────────────────────────────────────────────
    all_inds = [ind for deme in demes for ind in deme]
    new_inds = [ind for ind in all_inds if not ind.fitness.valid]
    n_evals  = len(new_inds)
    for ind in new_inds:
        ind.fitness.values = toolbox.evaluate(ind, points_train)

    if halloffame is not None:
        halloffame.update(all_inds)
        best_train_fitness = halloffame.items[0].fitness.values[0]
        best_ind_nodes     = len(halloffame.items[0])
        print(f"gen = 0 , fitness = {best_train_fitness} , evals = {n_evals} "
              f"[{n_demes} demes, mr={migration_rate}, freq={frequency_parameter}]")

    bd = behavioural_diversity(all_inds)
    logbook.record(gen=0, nevals=n_evals,
                   best_train_fitness=best_train_fitness,
                   best_ind_nodes=best_ind_nodes,
                   behavioural_diversity=bd,
                   accuracy_test=float('nan'),
                   best_phenotype=float('nan'))

    # ── Loop ──────────────────────────────────────────────────────────────────
    for gen in range(1, ngen + 1):
        n_evals = 0

        # ── Evolve each deme independently ────────────────────────────────────
        for d in range(n_demes):
            offspring = toolbox.select(demes[d], len(demes[d]))
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            new_inds  = [ind for ind in offspring if not ind.fitness.valid]
            n_evals  += len(new_inds)
            for ind in new_inds:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
            demes[d] = offspring

        # ── Migration — fires every frequency_parameter generations ───────────
        if gen % frequency_parameter == 0:
            if migration_topology == 'circle':
                # Collect emigrants from ORIGINAL demes (before any modification)
                # so deme B sends its own best, not those already received from A
                emigrants = []
                for d in range(n_demes):
                    sorted_deme = sorted(demes[d],
                                         key=lambda ind: ind.fitness.values[0])
                    emigrants.append(
                        [toolbox.clone(ind) for ind in sorted_deme[:n_migrants]]
                    )
                # Send: deme d → deme (d+1) % n_demes
                # Replace worst n_migrants in receiving deme (elitist)
                for d in range(n_demes):
                    receiver = (d + 1) % n_demes
                    demes[receiver].sort(key=lambda ind: ind.fitness.values[0],
                                         reverse=True)
                    for k, migrant in enumerate(emigrants[d]):
                        demes[receiver][k] = migrant
            else:
                raise ValueError(f"Unknown topology: {migration_topology}")

        # ── Global HoF + diversity (all demes merged, after migration) ─────────
        all_inds = [ind for deme in demes for ind in deme]

        if halloffame is not None:
            halloffame.update(all_inds)
            best_train_fitness = halloffame.items[0].fitness.values[0]
            best_ind_nodes     = len(halloffame.items[0])
            print(f"gen = {gen} , fitness = {best_train_fitness} , evals = {n_evals} "
                  f"[{n_demes} demes, mr={migration_rate}, freq={frequency_parameter}]")

        bd = behavioural_diversity(all_inds)

        if points_test and gen == ngen:
            _ = toolbox.evaluate(halloffame.items[0], points_test)
            accuracy_test  = 1.0 - halloffame.items[0].mce
            best_phenotype = str(halloffame.items[0])
        else:
            accuracy_test  = float('nan')
            best_phenotype = float('nan')

        logbook.record(gen=gen, nevals=n_evals,
                       best_train_fitness=best_train_fitness,
                       best_ind_nodes=best_ind_nodes,
                       behavioural_diversity=bd,
                       accuracy_test=accuracy_test,
                       best_phenotype=best_phenotype)

    population[:] = [ind for deme in demes for ind in deme]
    return population, logbook
