"""
selection.py — Lexicase selection variants for GP-based classification.

Implements the family of selection operators used by the island-model and
tournament GP experiments: standard lexicase selection, epsilon-lexicase
(static and dynamic thresholding), batch-lexicase (cases grouped into
batches to reduce the number of filtering steps), and node-count
tie-breaking variants that prefer smaller trees among individuals tied on
behaviour. Several variants additionally record diagnostic attributes on
the selected individuals (e.g. number of cases/batches used, number of
ties, fraction of unique individuals selected) for downstream analysis of
selection pressure and diversity.

Most individuals are expected to carry a `fitness_each_sample` attribute
(a per-case error/correctness vector) and a `nodes` attribute (tree size)
set elsewhere in the GP evaluation pipeline; these functions read and, in
several cases, annotate those attributes but do not compute them.
"""

import re
import math
from operator import attrgetter
import numpy as np
import random
import copy
import statistics

from codes.functions import shuffle_rows_except_first, remove_row, add_index_column, remove_columns, aggregate_rows, represent_matrix_behaviour, remove_equal_rows, remove_equal_columns, find_equal_columns, remove_columns_with_different_value, aggregate_rows_sum, count_zeros_except_first_row, count_zeros

def median_abs_deviation(arr, axis=0):
    """
    Compute the median absolute deviation (MAD) of a NumPy array along the
    given axis: the median of the absolute deviations of each element from
    the array's median. Used as a robust (outlier-resistant) spread measure
    for setting epsilon thresholds in the epsilon-lexicase variants below.

    This is a local copy of the same helper defined in functions.py —
    kept independent here rather than imported, so it is duplicated
    intentionally, not accidentally.

    Parameters:
        arr  -- NumPy array of values (individuals x fitness cases).
        axis -- axis along which to compute the median/MAD (default 0,
                i.e. per fitness case, down the rows).

    Returns:
        NumPy array of MAD values, one per index along the given axis.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate the median along axis 0
    median = np.median(arr, axis=0)

    # Calculate the absolute deviations from the median along axis 0
    abs_deviations = np.abs(arr - median)

    # Calculate the median of the absolute deviations along axis 0
    mad = np.median(abs_deviations, axis=0)

    return mad

def selLexicaseFilter(individuals, k):
    """
    Standard lexicase selection with a duplicate-behaviour prefilter.

    Individuals are first grouped by identical `fitness_each_sample` error
    vectors, and one representative is drawn at random from each group to
    form the candidate pool for each of the k selections; this avoids
    wasting filtering steps on behaviourally-identical clones. Within a
    pool, fitness cases are shuffled and applied one at a time, keeping
    only individuals that tie for the best value on each case, until a
    single individual remains or all cases are exhausted (ties broken
    randomly). If any individual already has fitness 0 (a perfect score),
    selection short-circuits to a random choice among those individuals
    for all k slots.

    Distinguishing feature vs. similarly-named variants: no node-count
    tie-breaking and no case-count instrumentation (see
    selLexicaseFilterCount for the latter, and selLexi2_nodesCountTies
    for the former).

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        while len(cases) > 0 and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexicaseFilterCount(individuals, k):
    """
    Same as selLexicaseFilter (duplicate-behaviour prefilter + standard
    lexicase filtering by shuffled cases), but additionally records on the
    selected individual how many fitness cases were consumed before the
    pool was reduced to a single candidate, via the `n_cases` attribute.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals, each with `n_cases` set (except
        when the fitness-zero shortcut applies).
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexi2_nodesCountTies(individuals, k):
    """
    Lexicase selection ("Lexi^2") with node-count tie-breaking and tie
    counting. Individuals are grouped by identical error vectors; within
    each group, only the individuals with the smallest tree size (`nodes`)
    are kept as candidates, and every individual in the group has its
    `ties` attribute set to the group size. The candidate pool (one
    representative per group) is then filtered by shuffled fitness cases
    as in standard lexicase, and the number of cases used is recorded on
    the selected individual via `n_cases`.

    As a shortcut, if any individuals already have a perfect score on
    every fitness case, selection is made randomly among the smallest of
    those perfect individuals for all k slots.

    Distinguishing feature vs. similarly-named variants: same as
    selLexi2_nodesCount (not shown here) plus the `ties` bookkeeping;
    unlike selEpsilonLexi2_nodesCountTies, ties require an exact match on
    the error vector rather than an epsilon-thresholded one.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals, with `ties` and `n_cases` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals
   
def selEpsilonLexi2_nodesCountTies(individuals, k, alpha):
    """
    Epsilon-lexicase selection ("Lexi^2") with node-count tie-breaking and
    tie counting. Each fitness case's error values are discretised to 0
    (within epsilon of the best value on that case) or 1 (outside epsilon)
    using a per-case threshold of alpha * MAD; the discretised vector is
    stored as `fitness_each_sample_discrete`. Individuals are then grouped
    by identical discretised vectors, filtered down to the smallest tree
    size within each group, and the group size is recorded as `ties` on
    every individual in it. Selection then proceeds as lexicase over the
    discretised vectors (minimising rather than maximising, since 0 = pass
    within epsilon). Also records `n_cases` (cases used), `avg_zeros`
    (fraction of discretised entries that are 0, i.e. within epsilon) and
    `avg_epsilon` (mean epsilon across cases) on the selected individuals,
    and `unique_selected` (fraction of unique individuals chosen across
    all k selections) on the first selected individual.

    As a shortcut, if any individuals already have a perfect score on
    every fitness case, selection is made randomly among the smallest of
    those perfect individuals for all k slots.

    Distinguishing feature vs. similarly-named variants: adds the `ties`
    bookkeeping on top of the epsilon-thresholded grouping used by
    selEpsilonLexi2_nodesCount; unlike selLexi2_nodesCountTies, matching
    is epsilon-tolerant rather than requiring an exact error-vector match.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        alpha       -- multiplier applied to the per-case MAD to obtain
                       the epsilon threshold for that case.

    Returns:
        List of k selected individuals, annotated as described above.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    #Check if we have found a perfect score already
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)

    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    epsilon = alpha * mad
    avg_epsilon = np.mean(epsilon)
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + epsilon[j]:
                fitness_cases_matrix[i][j] = 0
            else:
                fitness_cases_matrix[i][j] = 1
        candidates[i].fitness_each_sample_discrete = list(fitness_cases_matrix[i,:])
    
    n_zeros = count_zeros(fitness_cases_matrix) #number of zeros in the matrix with discrete fitness cases
    avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
    avg_zeros = avg_zeros / l_samples #represent as a percentage of the number of samples

    error_vectors = list(fitness_cases_matrix)
    
    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample_discrete == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    indexes = []
    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = min
            best_val_for_case = f(map(lambda x: x.fitness_each_sample_discrete[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample_discrete[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        pool[0].avg_zeros = avg_zeros
        pool[0].avg_epsilon = avg_epsilon
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases
        
        index = individuals.index(pool[0])
        indexes.append(index)
        
    selected_individuals[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected

    return selected_individuals
    
def selDynEpsilonLexicase(individuals, k):
    """
    Dynamic epsilon-lexicase selection.

    Unlike the static epsilon variants, the epsilon threshold here is
    recomputed at each filtering step from only the individuals still in
    the pool, rather than once upfront from the whole population. Fitness
    cases are represented as a matrix (samples x individuals, with an
    index row/column tracking original positions) and shuffled; on each
    iteration the current case's minimum plus MAD-over-the-remaining-pool
    defines the threshold, individuals above it are dropped, and the case
    is removed, continuing until one individual remains or all cases are
    used. Also records `n_cases` (cases used) and `ties` (size of the
    final tied group) on each selected individual.

    Distinguishing feature vs. similarly-named variants: "dynamic" means
    epsilon is recomputed per case from the shrinking pool (as opposed to
    the static per-generation epsilon used by selEpsilonLexi2_nodesCount
    and selEpsilonLexi2_nodesCountTies); this variant has no node-count
    tie-break (see selDynEpsilonLexi2_nodesCountTies for that).

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals, with `n_cases` and `ties` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        l, c = fitness_cases_matrix_original.shape
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            min_ = np.nanmin(fitness_cases_matrix[1])    
            mad = median_abs_deviation(fitness_cases_matrix[1]) #mad for the second row
            
            fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_ + mad) #filter individuals
            
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        remaining_candidates = fitness_cases_matrix[0].astype(int) #indexes of the remaining candidates
        selected_ind = candidates[random.choice(remaining_candidates)]
        selected_ind.n_cases = l_samples - l #number of testcases used in the filtering process
        selected_ind.ties = len(remaining_candidates)
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals

def selDynEpsilonLexi2_nodesCountTies(individuals, k):
    """
    Dynamic epsilon-lexicase selection ("Lexi^2") with node-count
    tie-breaking. Same case-by-case dynamic epsilon filtering as
    selDynEpsilonLexicase (threshold recomputed per case from the
    remaining pool; cases with no variation across the pool are skipped
    since they cannot filter anything), but once the case-based filtering
    ends with more than one candidate remaining, the tie is broken by
    picking randomly among the individuals with the smallest tree size
    (`nodes`). Records `n_cases` (cases used) and `ties` (size of the
    final tied group before node-count tie-breaking) on each selected
    individual.

    Distinguishing feature vs. similarly-named variants: adds node-count
    tie-breaking on top of selDynEpsilonLexicase's dynamic-epsilon
    filtering.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals, with `n_cases` and `ties` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        l, c = fitness_cases_matrix_original.shape
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            if np.all(fitness_cases_matrix[1] == fitness_cases_matrix[1, 0]): #if all individuals have the same fitness value for this case, we won't be able to filter anything
                pass
            else:
                min_ = np.nanmin(fitness_cases_matrix[1])    
                mad = median_abs_deviation(fitness_cases_matrix[1]) #mad for the second row
                fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_ + mad) #filter individuals
            
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        remaining_candidates = [candidates[j] for j in fitness_cases_matrix[0].astype(int)] #indexes of the remaining candidates
        if len(remaining_candidates) > 1:
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, remaining_candidates))
            smallest_size_candidates = [ind for ind in remaining_candidates if ind.nodes == best_val_for_nodes]
            selected_ind = random.choice(smallest_size_candidates) #if there are still more than one candidate with the same size, we choose randomly
        else:
            selected_ind = remaining_candidates[0]
        selected_ind.n_cases = l_samples - l #number of testcases used in the filtering process
        selected_ind.ties = len(remaining_candidates)
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals
     
def selEpsilonLexi2_nodesCount(individuals, k):
    """
    Static epsilon-lexicase selection ("Lexi^2") with node-count
    filtering but no tie-count bookkeeping. Each fitness case is
    discretised in place (`fitness_each_sample` is overwritten) to 1 if
    within a per-case MAD-based threshold of the best value, else 0;
    individuals are grouped by identical discretised vectors and each
    group is filtered down to its smallest-tree-size members before being
    used as the candidate pool for lexicase filtering by shuffled cases.
    Records `n_cases` (cases used) on the selected individual.

    As a shortcut, if any individuals already have a perfect score on
    every fitness case, selection is made randomly among the smallest of
    those perfect individuals for all k slots.

    Distinguishing feature vs. similarly-named variants: like
    selEpsilonLexi2_nodesCountTies but without recording the `ties`
    attribute, and epsilon is a fixed MAD (not alpha-scaled).

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals, with `n_cases` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)

    mad = median_abs_deviation(fitness_cases_matrix, axis=0)

    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
                
    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selBatchLexicase(individuals, k, batch_size=20):
    """
    Batch lexicase selection: fitness cases are grouped into batches of
    `batch_size`, and each individual's fitness within a batch is the sum
    of its per-case values over that batch. Filtering proceeds batch by
    batch (analogous to case by case in standard lexicase), keeping only
    individuals tying for the best summed value in each batch, which
    reduces the number of filtering steps from l_samples to
    ceil(l_samples / batch_size). Candidates are prefiltered by unique
    error vector before batching, as in selLexicaseFilter. If more than
    one candidate remains after exhausting all batches (possible because
    prefiltering is on raw per-case vectors, while filtering happens on
    batch sums), the final choice is random. Records `n_cases` (number of
    batches used) on the selected individual.

    Distinguishing feature vs. similarly-named variants: cases are
    aggregated into batches rather than filtered individually, and there
    is no epsilon thresholding or node-count tie-breaking here (see the
    selBatchEpsilonLexi2_nodesCount* variants for those).

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        batch_size  -- number of fitness cases summed per batch (default 20).

    Returns:
        List of k selected individuals, with `n_cases` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors
	
    n_batches = math.ceil(l_samples / batch_size)
    
    for i in range(len(candidates)):
        candidates[i].fitness_each_batch = [0] * n_batches

    for _ in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            candidate = random.choice(list_)
            candidate.fitness_each_batch = [0] * n_batches
            pool.append(candidate) 
        random.shuffle(cases)
        batch_ = 0
        while batch_ < n_batches - 1 and len(pool) > 1:
            #Build batch of batch_size cases
            for _ in range(batch_size):
                for i in range(len(pool)):
                    pool[i].fitness_each_batch[batch_] += pool[i].fitness_each_sample[cases[0]]
                del cases[0]
            f = max
            best_val_for_batch = f(map(lambda x: x.fitness_each_batch[batch_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batch_] == best_val_for_batch]
            batch_ += 1
        if batch_ == n_batches - 1 and len(pool) > 1:
            #Build batch with the remaining cases
            for case in cases:
                for i in range(len(pool)):
                    pool[i].fitness_each_batch[batch_] += pool[i].fitness_each_sample[case]
            f = max
            best_val_for_batch = f(map(lambda x: x.fitness_each_batch[batch_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batch_] == best_val_for_batch]
            batch_ += 1
        
        #Despite filtering the individuals initially, we can have more than one remaining in the pool after checking the batches, because inds with different behaviours can have the same batch fitness
        if len(pool) == 1:
            selected_individual = pool[0]
        else:
            selected_individual = random.choice(pool)
        selected_individual.n_cases = batch_
        selected_individuals.append(selected_individual)
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selBatchEpsilonLexi2_nodesCountTies(individuals, k, batch_size=20):
    """
    Batch epsilon-lexicase selection ("Lexi^2") with node-count
    tie-breaking and tie counting. Epsilon (per-case MAD) and the
    resulting discretised error vectors are computed once for the whole
    generation (not per selection), and individuals are prefiltered by
    unique discretised vector and smallest tree size, with `ties` set to
    each group's size. The unique vectors are then batched and filtered
    batch by batch (matrix-based, via aggregate_rows), analogous to
    selBatchLexicase but on epsilon-thresholded values. Because batching
    happens after deduplication, it is still possible for two prefiltered
    groups to end up tied after batching; the final choice among tied
    groups, and within the winning group, is random. Records `n_cases`
    (batches used) and `ties` on the selected individual.

    Distinguishing feature vs. similarly-named variants: epsilon/MAD is
    computed once per generation (cheaper) as opposed to
    selBatchEpsilonLexi2_nodesCountTies_MADafter, which recomputes it
    after batching (more accurate per-batch statistics, more expensive).

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        batch_size  -- number of fitness cases aggregated per batch
                       (default 20).

    Returns:
        List of k selected individuals, with `n_cases` and `ties` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    error_vectors = [ind.fitness_each_sample for ind in individuals] #real values
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    epsilon = median_abs_deviation(fitness_cases_matrix, axis=0) #mad

    candidates = individuals
    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + epsilon[j]:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
            else:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
                
    error_vectors = list(fitness_cases_matrix)
    
    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        for ind in cands:
            ind.ties = len(cands)
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    n_batches = math.ceil(l_samples / batch_size)

    fitness_cases_matrix_original = np.array(unique_error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning; these indexes match to candidates_prefiltered_set
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        #Create batches
        fitness_cases_matrix = aggregate_rows(fitness_cases_matrix, batch_size)
        
        l, c = fitness_cases_matrix.shape
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            if np.all(fitness_cases_matrix[1] == fitness_cases_matrix[1, 0]): #if all individuals have the same fitness value for this case, we won't be able to filter anything
                pass #we preprocessing the data to have unique vectors, but while creating batches, it's possible to have same vector again
            else: #we do Lexicase as normal
                min_ = np.nanmin(fitness_cases_matrix[1])    
                fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_) #filter individuals
                
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        winning_indexes = list(fitness_cases_matrix[0].astype(int)) #indexes of the remaining candidates
        if len(winning_indexes) > 1:
            selected_index = random.choice(winning_indexes)
        else:
            selected_index = winning_indexes[0]
        selected_ind = random.choice(candidates_prefiltered_set[selected_index])
        selected_ind.n_cases = n_batches - l #number of batches used in the filtering process
        selected_ind.ties = len(candidates_prefiltered_set[selected_index])
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals
    
def selBatchEpsilonLexi2_nodesCountTies_MADafter(individuals, k, batch_size=20):
    """
    Same as selBatchEpsilonLexi2_nodesCountTies, but MAD/epsilon is
    computed after batching (per selection, from the shuffled+aggregated
    batch matrix) rather than once per generation on raw cases. This is
    more expensive since MAD is recomputed at the start of every one of
    the k selections, but it can save time afterwards because
    deduplication of unique vectors happens after batching+discretisation
    rather than before, giving fewer, more meaningful groups. Also
    records `avg_zeros`, `avg_epsilon` (per selection) and
    `unique_selected` (fraction of unique individuals chosen across all k
    selections, set on the first selected individual), in addition to
    `n_cases` (batches used) and `ties`.

    Distinguishing feature vs. similarly-named variants: MAD/epsilon
    thresholding is recomputed per-selection after batching, instead of
    once per generation before batching (selBatchEpsilonLexi2_nodesCountTies).

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        batch_size  -- number of fitness cases aggregated per batch
                       (default 20).

    Returns:
        List of k selected individuals, annotated as described above.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    error_vectors = [ind.fitness_each_sample for ind in individuals] #real values
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning; these indexes match to candidates_prefiltered_set
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    candidates = individuals
    
    indexes = [] #indexes of the selected inds
    
    for _ in range(k):
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        #Create batches. aggregate_rows is used rather than aggregate_rows_sum because MAD
        #is computed per batch on the raw matrix and every row must stay independent, even
        #though the last batch may have a different size.
        fitness_cases_matrix = aggregate_rows(fitness_cases_matrix, batch_size)

        fitness_cases_matrix = fitness_cases_matrix.transpose()
        min_ = np.nanmin(fitness_cases_matrix[:,1:], axis=0)
        epsilon = median_abs_deviation(fitness_cases_matrix[:,1:], axis=0) #mad
        fitness_cases_matrix = fitness_cases_matrix.transpose()

        fitness_cases_matrix[1:] = represent_matrix_behaviour(fitness_cases_matrix[1:], min_ + epsilon)
        
        n_zeros = count_zeros_except_first_row(fitness_cases_matrix) #number of zeros in the matrix with dicrete fitness cases
        
        fitness_cases_matrix_reserved = fitness_cases_matrix.copy()
        
        fitness_cases_matrix = remove_equal_columns(fitness_cases_matrix)
        
        l, c = fitness_cases_matrix.shape
        n_batches = l - 1
                
        avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
        avg_zeros = avg_zeros / n_batches #represent as a percentage of the number of batches
        
        avg_epsilon = np.mean(epsilon)
    
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            min_ = np.nanmin(fitness_cases_matrix[1])    
            fitness_cases_matrix = remove_columns_with_different_value(fitness_cases_matrix, min_) #filter individuals
                
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        selected_index = int(fitness_cases_matrix[0])
        candidates_indexes = find_equal_columns(fitness_cases_matrix_reserved, selected_index) #indexes of the candidates with the best vector

        tied_candidates = []
        for idx in candidates_indexes:
            tied_candidates.append(candidates[idx])

        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, tied_candidates))
        smallest_candidates = [ind for ind in tied_candidates if ind.nodes == best_val_for_nodes]

        selected_ind = random.choice(smallest_candidates)
        index = candidates_indexes[tied_candidates.index(selected_ind)]
        indexes.append(index)
        selected_ind.ties = len(tied_candidates)
        selected_ind.n_cases = n_batches - l #number of batches used in the filtering process
        selected_individuals.append(selected_ind) #Select the remaining candidate
        selected_ind.fitness_each_sample_discrete = list(fitness_cases_matrix_reserved[1:,selected_index])
        selected_ind.avg_zeros = avg_zeros
        selected_ind.avg_epsilon = avg_epsilon
        
    selected_individuals[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected

    return selected_individuals

def selTournamentExtra(individuals, k, tournsize, fit_attr="fitness"):
    """
    Standard tournament selection (draw `tournsize` random individuals,
    keep the best by `fit_attr`, repeat k times), augmented with the same
    post-hoc behavioural diagnostics used by the epsilon-lexicase
    variants for comparison purposes: each chosen individual gets an
    epsilon-discretised behaviour vector (`fitness_each_sample_discrete`,
    using a per-case MAD threshold computed over just the k chosen
    individuals), `avg_zeros` and `avg_epsilon`, and the first chosen
    individual gets `unique_selected` (fraction of unique individuals
    among the k chosen). These diagnostics are computed only to make
    tournament-selection runs directly comparable to the lexicase
    variants in later analysis; they play no part in the tournament
    selection itself.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        tournsize   -- number of individuals drawn per tournament.
        fit_attr    -- attribute name used to compare fitness (default
                       "fitness").

    Returns:
        List of k selected individuals, annotated as described above.
    """
    chosen = []
    for i in range(k):
        aspirants = [random.choice(individuals) for i in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    
    error_vectors = [ind.fitness_each_sample for ind in chosen] #real values
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix[:,:], axis=0)
    epsilon = median_abs_deviation(fitness_cases_matrix[:,:], axis=0) #mad
    fitness_cases_matrix = fitness_cases_matrix.transpose() # samples (rows) x inds (cols)
    fitness_cases_matrix[:] = represent_matrix_behaviour(fitness_cases_matrix[:], min_ + epsilon)

    n_zeros = count_zeros(fitness_cases_matrix) #number of zeros in the matrix with unique fitness cases
    avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
    avg_zeros = avg_zeros / len(fitness_cases_matrix[:,0]) #represent as a percentage of the number of samples
    
    avg_epsilon = np.mean(epsilon)
    
    indexes = []
    for i in range(k):
        chosen[i].fitness_each_sample_discrete = list(fitness_cases_matrix[:,i])
        chosen[i].avg_zeros = avg_zeros
        chosen[i].avg_epsilon = avg_epsilon

        index = individuals.index(chosen[i])
        indexes.append(index)
        
    chosen[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected
    
    return chosen

def selBatchEpsilonLexi2_nodesCountTiesOld(individuals, k, batch_size=20):
    """
    Earlier, explicit-loop implementation of batch epsilon-lexicase with
    node-count tie-breaking and tie counting (superseded by
    selBatchEpsilonLexi2_nodesCountTies, which does the same thing using
    matrix helper functions). Cases are shuffled once for the whole
    generation and manually summed into `n_batches` batches on
    `fitness_each_batch`; each batch is then discretised against a
    per-batch MAD threshold, individuals are grouped by identical
    discretised batch vectors and filtered to the smallest tree size
    (recording `ties`), and lexicase filtering proceeds batch by batch
    (with a fresh random batch order per selection) until one candidate
    remains. Records `n_cases` (batches used) and `ties`.

    Kept for reference/comparison against the current implementation
    rather than for active use.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        batch_size  -- number of fitness cases aggregated per batch
                       (default 20).

    Returns:
        List of k selected individuals, with `n_cases` and `ties` set.
    """
    selected_individuals = []
    error_vectors = [ind.fitness_each_sample for ind in individuals]
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
   
    pop_size = len(individuals)
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    n_batches = math.ceil(l_samples / batch_size)
    
    cases = list(range(0,l_samples))
    random.shuffle(cases)
    fitness_batches_matrix = np.zeros([pop_size, n_batches], dtype=float) # inds (rows) x samples (cols)
    #partitions
    for i in range(n_batches-1):
        for _ in range(batch_size):
            fitness_batches_matrix[:,i] += fitness_cases_matrix[:,cases[0]]
            del cases[0]
    for case in cases:
        fitness_batches_matrix[:,n_batches-1] += fitness_cases_matrix[:,case]

    min_ = np.nanmin(fitness_batches_matrix, axis=0)
    mad = median_abs_deviation(fitness_batches_matrix, axis=0)

    candidates = individuals
    for i in range(len(candidates)):
        candidates[i].fitness_each_batch = [0] * n_batches
        for j in range(n_batches):
            if fitness_batches_matrix[i][j] <= min_[j] + mad[j]:
                fitness_batches_matrix[i][j] = 1
                candidates[i].fitness_each_batch[j] = 1
            else:
                fitness_batches_matrix[i][j] = 0
                candidates[i].fitness_each_batch[j] = 0
            
    error_vectors = list(fitness_batches_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_batch == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    batches = list(range(0,n_batches))
    
    for _ in range(k):
        random.shuffle(batches)
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        count_ = 0
        while len(batches) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_batch[batches[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batches[0]] == best_val_for_case]
            del batches[0]
            
        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        
        batches = list(range(0,n_batches))

    return selected_individuals

def selDynBatchEpsilonLexi2_nodesCountTies(individuals, k, batch_size):
    """
    Dynamic version of batch epsilon-lexicase ("Lexi^2") with node-count
    tie-breaking. Fitness cases are shuffled and aggregated into batches
    (via aggregate_rows), then filtered batch by batch with epsilon/MAD
    recomputed at each step from only the individuals still in the pool
    (as in selDynEpsilonLexicase, but operating on batches instead of
    individual cases); batches with no variation across the remaining
    pool are skipped since they cannot filter anything. Once batch-based
    filtering ends, ties are broken by picking randomly among the
    remaining individuals with the smallest tree size (`nodes`). Records
    `n_cases` (batches used) and `ties` (size of the final tied group
    before node-count tie-breaking).

    Distinguishing feature vs. similarly-named variants: epsilon is
    recomputed dynamically per batch from the shrinking pool, rather than
    once per generation (selBatchEpsilonLexi2_nodesCountTies) or once per
    selection after batching (selBatchEpsilonLexi2_nodesCountTies_MADafter).
    Unlike those two, `batch_size` has no default here.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        batch_size  -- number of fitness cases aggregated per batch.

    Returns:
        List of k selected individuals, with `n_cases` and `ties` set.
    """
    selected_individuals = []

    l_batches = math.ceil(np.shape(individuals[0].fitness_each_sample)[0] / batch_size)
    
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        #Create batches
        fitness_cases_matrix = aggregate_rows(fitness_cases_matrix, batch_size)
        
        l, c = fitness_cases_matrix.shape
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            if np.all(fitness_cases_matrix[1] == fitness_cases_matrix[1, 0]): #if all individuals have the same fitness value for this case, we won't be able to filter anything
                pass
            else:
                min_ = np.nanmin(fitness_cases_matrix[1])    
                mad = median_abs_deviation(fitness_cases_matrix[1]) #mad for the second row
                fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_ + mad) #filter individuals
            
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        remaining_candidates = [candidates[j] for j in fitness_cases_matrix[0].astype(int)] #indexes of the remaining candidates
        if len(remaining_candidates) > 1:
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, remaining_candidates))
            smallest_size_candidates = [ind for ind in remaining_candidates if ind.nodes == best_val_for_nodes]
            selected_ind = random.choice(smallest_size_candidates) #if there are still more than one candidate with the same size, we choose randomly
        else:
            selected_ind = remaining_candidates[0]
        selected_ind.n_cases = l_batches - l #number of batches used in the filtering process
        selected_ind.ties = len(remaining_candidates)
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals
        
def selBatchEpsilonLexi2_nodesCountOld(individuals, k, batch_size=2):
    """
    Earlier, more expensive batch epsilon-lexicase variant with
    node-count filtering (no tie counting). Unlike the other batch
    variants, which build one shared batching for the whole generation
    (or one per selection but reusing shuffled cases), this version
    reshuffles cases and rebuilds the batches, MAD thresholds, and
    discretised vectors independently for every one of the k selections
    ("different batches for selecting each individual"). Individuals are
    discretised per batch against a MAD threshold, grouped by identical
    discretised batch vectors, filtered to the smallest tree size, and
    then lexicase-filtered batch by batch. Records `n_cases` (batches
    used) on the selected individual.

    Kept for reference/comparison; selBatchEpsilonLexi2_nodesCountTies
    is the cheaper, generation-level-batching equivalent.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.
        batch_size  -- number of fitness cases aggregated per batch
                       (default 2).

    Returns:
        List of k selected individuals, with `n_cases` set.
    """
    selected_individuals = []
    error_vectors = [ind.fitness_each_sample for ind in individuals]
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    pop_size = len(individuals)
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    n_batches = math.ceil(l_samples / batch_size)
    
    candidates = individuals
    
    for _ in range(k):
        cases = list(range(0,l_samples))
        random.shuffle(cases)
        fitness_batches_matrix = np.zeros([pop_size, n_batches], dtype=float) # inds (rows) x samples (cols)
        #partitions
        for i in range(n_batches-1):
            for _ in range(batch_size):
                fitness_batches_matrix[:,i] += fitness_cases_matrix[:,cases[0]]
                del cases[0]
        for case in cases:
            fitness_batches_matrix[:,n_batches-1] += fitness_cases_matrix[:,case]

        min_ = np.nanmin(fitness_batches_matrix, axis=0)
        mad = median_abs_deviation(fitness_batches_matrix, axis=0)

        for i in range(len(candidates)):
            candidates[i].fitness_each_batch = [0] * n_batches
            for j in range(n_batches):
                if fitness_batches_matrix[i][j] <= min_[j] + mad[j]:
                    fitness_batches_matrix[i][j] = 1
                    candidates[i].fitness_each_batch[j] = 1
                else:
                    fitness_batches_matrix[i][j] = 0
                    candidates[i].fitness_each_batch[j] = 0
                
        error_vectors = list(fitness_batches_matrix)

        unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
        unique_error_vectors = [list(i) for i in unique_error_vectors]
        
        candidates_prefiltered_set = []
        for i in range(len(unique_error_vectors)):
            cands = [ind for ind in candidates if ind.fitness_each_batch == unique_error_vectors[i]]
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, cands))
            cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
            candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        count_ = 0
        while count_ < n_batches and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_batch[count_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[count_] == best_val_for_case]
            count_ += 1

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate

    return selected_individuals

def selEpsilonLexicaseCount(individuals, k):
    """
    Static epsilon-lexicase selection with case-count instrumentation but
    no node-count filtering. Each fitness case is discretised in place
    (`fitness_each_sample` overwritten) to 1 if within a per-case MAD
    threshold of the best value, else 0; individuals are grouped by
    identical discretised vectors (no size-based prefiltering within a
    group, unlike the *_nodesCount variants), and lexicase filtering by
    shuffled cases picks one representative per group. Records `n_cases`
    (cases used) on the selected individual.

    As a shortcut, if any individuals already have a perfect score on
    every fitness case, selection is made randomly among the smallest of
    those perfect individuals for all k slots.

    Distinguishing feature vs. similarly-named variants: like
    selEpsilonLexi2_nodesCount but without the node-count tie-break
    within each duplicate-vector group.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals, with `n_cases` set.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    mad = median_abs_deviation(fitness_cases_matrix, axis=0)

    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0

    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]

    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_))
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexicase(individuals, k):
    """
    Reference lexicase selection using a boolean per-case correctness
    vector (`fitness_each_sample` entries are True/False) rather than a
    continuous error vector: on each shuffled case, only individuals with
    a True value survive, unless doing so would eliminate the whole pool
    (in which case that case is skipped and the pool is left unchanged).
    Excludes individuals flagged invalid (`i.invalid`) before selection
    begins. No case-count instrumentation, node-count tie-breaking, or
    duplicate-vector prefiltering — this is the simplest lexicase
    implementation in this module, used as a baseline/reference; see
    selLexicaseCount for the instrumented version.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        List of k selected individuals.
    """
    selected_individuals = []
    valid_individuals = [i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = valid_individuals
    
    for i in range(k):
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update    
            del cases[0]                    

        #If there is only one candidate remaining, it will be selected
        #If there are more than one, the choice will be made randomly
        selected_individuals.append(random.choice(candidates))
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals

def selLexicaseCount(individuals, k):
    """Same as selLexicase (boolean per-case correctness vector, one case
    filtered at a time), but instrumented to analyse how the selection
    process played out across the population, and returns extra
    diagnostic data alongside the selected individuals:
      - samples_attempted        -- per-case count of how many times that
                                     case was used as a filtering step.
      - samples_used              -- per-case count of how many times
                                     filtering on that case actually
                                     narrowed the candidate pool (without
                                     eliminating it entirely).
      - samples_unsuccessful1     -- per-case count of times filtering on
                                     that case changed nothing (all
                                     candidates already agreed).
      - samples_unsuccessful2     -- per-case count of times filtering on
                                     that case would have eliminated all
                                     candidates (so it was skipped).
      - inds_to_choose             -- per-selection size of the final
                                     candidate pool the choice was made
                                     from.
      - times_chosen               -- length-4 tally of how each of the k
                                     selections was resolved: index 0 =
                                     resolved by a unique fitness winner
                                     (or the fitness-zero shortcut with a
                                     single candidate), index 3 = resolved
                                     by random tie-break; indices 1-2 are
                                     unused placeholders.

    If some individuals already have fitness equal to zero (a perfect
    score), selection short-circuits to a random choice among those
    individuals for all k slots, and the counters above are updated
    accordingly without running the per-case filtering loop.

    Parameters:
        individuals -- pool of individuals to select from.
        k           -- number of individuals to select.

    Returns:
        Tuple of (selected_individuals, samples_attempted, samples_used,
        samples_unsuccessful1, samples_unsuccessful2, inds_to_choose,
        times_chosen).
    """
    selected_individuals = []
    valid_individuals = [i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    
    #For analysing Lexicase selection
    samples_attempted = [0]*l_samples
    samples_used = [0]*l_samples
    samples_unsuccessful1 = [0]*l_samples
    samples_unsuccessful2 = [0]*l_samples
    inds_to_choose = [0]*k
    times_chosen = [0]*4
    
    cases = list(range(0,l_samples))
    candidates = valid_individuals
    
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
            inds_to_choose[i] = len(inds_fitness_zero)
            if len(inds_fitness_zero) == 1:
                times_chosen[0] += 1 #The choise was made by error
            else:
                times_chosen[3] += 1 #The choise was made by randomly
        samples_attempted = [x+k for x in samples_attempted]
        samples_used = [x+1 for x in samples_used]
        samples_unsuccessful1 = [x+k-1 for x in samples_unsuccessful1]
        
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen

    for i in range(k):
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]

            samples_attempted[cases[0]] += 1
            if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                samples_used[cases[0]] += 1
            if (len(candidates_update) == len(candidates)):
                samples_unsuccessful1[cases[0]] += 1
            if len(candidates_update) == 0:
                samples_unsuccessful2[cases[0]] += 1
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update
            del cases[0]

        #If there is only one candidate remaining, it will be selected
        if len(candidates) == 1:
            selected_individuals.append(candidates[0])
            inds_to_choose[i] = 1
            times_chosen[0] += 1 #The choise was made by fitness
        else: #If there are more than one, the choice will be made randomly
            selected_individuals.append(random.choice(candidates))
            inds_to_choose[i] = len(candidates)
            times_chosen[3] += 1 #The choise was made by randomly
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen
