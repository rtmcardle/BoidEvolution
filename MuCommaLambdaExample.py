
from deap import creator, base, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import random, array, math, os, datetime, textwrap


def initACK(ind_cls, strg_cls, dim, dmin, dmax, smin, smax):
    """
    An initializer to create individuals for the Ackley 
    Function Evolution.
    """

    ind = ind_cls((np.random.uniform(dmin,dmax) for _ in range(dim)))
    ind.strategy = strg_cls((np.random.uniform(smin,smax) for _ in range(dim)))
    return ind

def evaluateAck(ind):
    """
    The fitness function for probelm 2, i.e. Ackley's function.
    """

    n = len(ind)
    listsq = [x**2 for x in ind]
    listcos = [math.cos(2*math.pi*x) for x in ind]
    t1 = -20*math.exp(-0.2*math.sqrt((1/n)*np.sum(listsq)))
    t2 = -math.exp((1/n)*np.sum(listcos))
    t3 = math.e + 20
    return t1+t2+t3,

def checkBounds(xmin, xmax, ymin, ymax):
    """
    A wrapper to correct for parameters that go outside of the 
    parameter's desired bounds.
    """

    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                if child[0] < xmin:
                    child[0] = xmin
                elif child[0] > xmax:
                    child[0] = xmax
                if child[1] < ymin:
                    child[1] = ymin
                elif child[1] > ymax:
                    child[1] = ymax
            return offspring
        return wrapper
    return decorator

def checkStrategy(minstrategy):
    """
    A wrapper to correct for strategy variables that go below
    a minimum value.
    """

    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def solve2():
    """
    The function to solve for the optimum to the function
    described in problem 2, i.e. Ackley's function.
    """

    mu = 100
    lambda_ = 7 * mu
    CXPB = 0.75
    MUTPB = 0.2
    eval_limit = 200000

    ## Specifies the number of dimensions for Ackley's
    dim = 30

    ## Specifies the bounds for our parameters; pass to the 
    ## toolbox.decorate methods with mate and mutate
    dmin,dmax = -30,30
    smin, smax = 5e-16, 1.0

    ## Create Fitness, Individuals, and Strategy variables
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    ## Sets up our evolutionary approach
    toolbox = base.Toolbox()

    toolbox.register('individual', initACK, creator.Individual, creator.Strategy, dim, dmin, dmax, smin, smax)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxESBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutESLogNormal, c=0.01, indpb=0.25)
    toolbox.register('select', tools.selRandom)
    toolbox.register('evaluate', evaluateAck)

    toolbox.decorate('mate', checkBounds(dmin,dmax,dmin,dmax))
    toolbox.decorate('mutate', checkBounds(dmin,dmax,dmin,dmax))

    toolbox.decorate('mate', checkStrategy(smin))
    toolbox.decorate('mutate', checkStrategy(smin))


    ## Records statistics about the population during evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    ## Records the 1 best individual seen in evolution
    hof = tools.HallOfFame(1)

    ## Initializes the population with n individuals
    pop = toolbox.population(n=mu)

    ## Evaluate the entire population for fitness
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    ## Record initial population in stats logbook
    g = 0
    record = stats.compile(pop)
    total_evals = mu
    logbook = tools.Logbook()
    logbook.header = 'gen','evals','min','max','avg','std'
    logbook.record(gen=0, evals=mu, **record)

    ## Loop for each generation until stopping criteria
    while True:
        if total_evals < (eval_limit - lambda_):
            g+=1
            # Select the next generation individuals
            offspring = toolbox.select(pop, lambda_)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Sort the offspring by fitness
            offspring.sort(key=lambda x: x.fitness.values[0])

            # Replace population with top mu offspring
            pop = list(map(toolbox.clone, offspring[:mu]))

            # Record the new generation
            hof.update(pop)
            record = stats.compile(pop)
            total_evals += len(invalid_ind)
            logbook.record(gen=g, evals=total_evals, **record)
            print(logbook.stream)
        else:
            break


    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    size_avgs = logbook.select("avg")

    print(f'Best Score: {hof[0].fitness.values[0]}')
    print(f'Best Individual: {hof[0]}')

    ## Plot evolution
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Fitness")
    ax2.set_ylabel("Avg. Fit.", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    ## Record run
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
    cur_dir = os.getcwd()
    prob_dir = os.path.join(cur_dir,'HW5\\Prob2\\')
    if not os.path.exists(prob_dir):
        os.mkdir(prob_dir)
    time_dir = os.path.join(prob_dir,f'{timestamp}\\')
    os.mkdir(time_dir)
    plt.savefig(os.path.join(time_dir,'plot.png'))

    ## Log the run
    log_file = os.path.join(time_dir,'log.txt')
    with open(log_file, 'w') as f:
        print(f'Best Score: {hof[0].fitness.values[0]}', file=f)
        print(f'Best Individual: {hof[0]}\n', file=f)
        print(f'History: ', file=f)
        print(logbook, file=f)

    return hof[0]


if __name__=='__main__':

    solve2()