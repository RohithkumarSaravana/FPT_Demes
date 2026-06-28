"""
paper_complete_script_met2.py
Tournament selection baseline — 5 datasets, 30 seeds, 7 metrics.

Metrics per generation : gen, nevals, best_train_fitness, best_ind_nodes,
                         behavioural_diversity
Last generation only   : accuracy_test, best_phenotype

Usage (run from inside the met2/ folder):
    python codes/paper_complete_script_met2.py wine        1 30
    python codes/paper_complete_script_met2.py iris        1 30
    python codes/paper_complete_script_met2.py australian  1 30
    python codes/paper_complete_script_met2.py pima        1 30
    python codes/paper_complete_script_met2.py heart       1 30
"""

import sys, os, csv, random, itertools, math, warnings
import numpy as np
import pandas as pd
from deap import creator, base, tools, gp
import operator
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithms_gp_met2 as algorithms
from functions import WA, OWA, minimum, maximum, dilator, concentrator, complement
from fuzzify import matrixDomain, fuzzifyDataFrame

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
problem = sys.argv[1] if len(sys.argv) > 1 else 'wine'
run     = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_RUNS  = int(sys.argv[3]) if len(sys.argv) > 3 else 30

# ── GP parameters ─────────────────────────────────────────────────────────────
FUZZY_SETS      = 3
MAX_DEPTH       = 17
POPULATION_SIZE = 500
GENERATIONS     = 50
TOURNSIZE       = 7
MIN_INIT_DEPTH  = 2
MIN_MUT_DEPTH   = 0
MAX_MUT_DEPTH   = 3
MAX_INIT_DEPTH  = 6
P_CROSSOVER     = 0.8
P_MUTATION      = 0.05

REPORT_ITEMS = ['gen', 'nevals', 'best_train_fitness', 'best_ind_nodes',
                'behavioural_diversity', 'accuracy_test', 'best_phenotype']

N_CLASSES = {'wine': 3, 'iris': 3, 'australian': 2, 'pima': 2, 'heart': 2}

# ── Dataset loader ────────────────────────────────────────────────────────────
def setDataSet(problem, seed, fuzzy_sets):
    if problem == 'wine':
        data = pd.read_csv("datasets/wine.data", sep=",")
        Ypd = data['class']; data = data.drop(['class'], axis=1)
        num_cols = list(data.columns); cat_cols = []

    elif problem == 'iris':
        data = pd.read_csv("datasets/iris.data", sep=",")
        Ypd = data['class']; data = data.drop(['class'], axis=1)
        num_cols = list(data.columns); cat_cols = []

    elif problem == 'australian':
        data = pd.read_csv("datasets/australian.data", sep=" ", header=None)
        data.columns = ['A1','A2','A3','A4','A5','A6','A7',
                        'A8','A9','A10','A11','A12','A13','A14','class']
        Ypd = data['class']; data = data.drop(['class'], axis=1)
        num_cols = ['A2','A3','A7','A10','A13','A14']
        cat_cols = ['A1','A4','A5','A6','A8','A9','A11','A12']

    elif problem == 'pima':
        data = pd.read_csv("datasets/pima.data", sep=",", header=None)
        data.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                        'Insulin','BMI','DiabetesPedigree','Age','class']
        Ypd = data['class']; data = data.drop(['class'], axis=1)
        num_cols = list(data.columns); cat_cols = []

    elif problem == 'heart':
        data = pd.read_csv("datasets/heart.data", sep=" ", header=None)
        data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
                        'thalach','exang','oldpeak','slope','ca','thal','class']
        Ypd = data['class']; data = data.drop(['class'], axis=1)
        Ypd = (Ypd > 0).astype(int)
        num_cols = ['age','trestbps','chol','thalach','oldpeak']
        cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

    else:
        raise ValueError(f"Unknown problem '{problem}'. "
                         "Choose: wine, iris, australian, pima, heart.")

    Y = pd.get_dummies(Ypd).to_numpy()
    X = data.to_numpy()
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=seed)
    df_tr = pd.DataFrame(X_tr, columns=data.columns)
    df_te = pd.DataFrame(X_te, columns=data.columns)
    domain = matrixDomain(df_tr, num_cols, cat_cols) if cat_cols else matrixDomain(df_tr, num_cols)
    X_tr = fuzzifyDataFrame(df_tr, fuzzy_sets, domain).to_numpy()
    X_te = fuzzifyDataFrame(df_te, fuzzy_sets, domain).to_numpy()
    return X_tr, Y_tr, X_te, Y_te

# ── Helpers ───────────────────────────────────────────────────────────────────
def extend(f): return f
def WTA(a, b): return [a, b]
def WTA3(a, b, c): return [a, b, c]

def mutUniform(individual, expr, pset):
    index = random.randrange(1, len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,

def fitness_eval0(string_features, individual, points):
    X = points[0]; y = points[1]
    exec(string_features)
    try:
        pred = np.array(eval(str(individual))).transpose()
    except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError):
        return np.NaN,
    assert np.isrealobj(pred)
    fitness = math.sqrt(np.mean(np.square(np.subtract(y, pred))))
    labels = np.argmax(y, axis=1)
    labels_pred = np.argmax(pred, axis=1)
    individual.mce = 1 - np.mean(np.equal(labels, labels_pred))
    individual.behaviour = list(labels_pred)
    expr_tmp = toolbox.individual()
    nodes, _, _ = gp.graph(expr_tmp)
    individual.nodes = len(nodes)
    return fitness,

def create_toolbox(problem, n_features):
    n_classes = N_CLASSES.get(problem, 3)
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
    if n_classes == 3:
        pset.addPrimitive(WTA3, [float, float, float], list)
    else:
        pset.addPrimitive(WTA, [float, float], list)
    pset.addPrimitive(WA,           [float, float, str], float)
    pset.addPrimitive(OWA,          [float, float, str], float)
    pset.addPrimitive(minimum,      [float, float],      float)
    pset.addPrimitive(maximum,      [float, float],      float)
    pset.addPrimitive(dilator,      [float],             float)
    pset.addPrimitive(concentrator, [float],             float)
    pset.addPrimitive(complement,   [float],             float)
    pset.addPrimitive(extend,       [str],               str)
    for t in ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']:
        pset.addTerminal(t, str)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    tb = base.Toolbox()
    tb.register("expr",       gp.genHalfAndHalf, pset=pset, min_=MIN_INIT_DEPTH, max_=MAX_INIT_DEPTH)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat,  list, tb.individual)
    tb.register("compile",    gp.compile, pset=pset)
    tb.register("mate",       gp.cxOnePoint)
    tb.register("expr_mut",   gp.genGrow, min_=MIN_MUT_DEPTH, max_=MAX_MUT_DEPTH)
    tb.register("mutate",     mutUniform, expr=tb.expr_mut, pset=pset)
    tb.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    tb.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    tb.register("select", tools.selTournament, tournsize=TOURNSIZE)
    string_features = ''.join(f"IN{i} = X[:,{i}]; " for i in range(n_features))
    tb.register("evaluate", fitness_eval0, string_features)
    return tb

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in range(N_RUNS):
    RANDOM_SEED = i + run
    print(f"\nRun: {RANDOM_SEED}  [{problem} | tournament]")
    np.random.seed(RANDOM_SEED)
    X_train, Y_train, X_test, Y_test = setDataSet(problem, RANDOM_SEED, FUZZY_SETS)
    print(f"n_features = {X_train.shape[1]}")
    toolbox = create_toolbox(problem, X_train.shape[1])
    random.seed(RANDOM_SEED)
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    population, logbook = algorithms.eaSimple(
        population=pop, toolbox=toolbox,
        cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=GENERATIONS,
        points_train=[X_train, Y_train], points_test=[X_test, Y_test],
        report_items=REPORT_ITEMS, halloffame=hof
    )

    address = f"results_met2/GP/{problem}/tournament/"
    os.makedirs(address, exist_ok=True)
    gen  = logbook.select("gen");  nevals = logbook.select("nevals")
    btf  = logbook.select("best_train_fitness")
    bin_ = logbook.select("best_ind_nodes")
    bd   = logbook.select("behavioural_diversity")
    acc  = logbook.select("accuracy_test")
    bph  = logbook.select("best_phenotype")

    with open(address + f"{RANDOM_SEED}.csv", "w", encoding='UTF8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(REPORT_ITEMS)
        for v in range(len(gen)):
            w.writerow([gen[v], nevals[v], btf[v], bin_[v], bd[v], acc[v], bph[v]])

    print(f"  accuracy_test={acc[-1]:.4f}  bd_final={bd[-1]:.4f}  -> {address}{RANDOM_SEED}.csv")
