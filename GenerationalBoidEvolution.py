from BoidSim import Flock

import random
import math
import os
import multiprocessing
import numpy as np
from deap import algorithms, base, creator, tools
from joblib import load
import sklearn


class GenerationalGA:
    def __init__(self):
        self.loadClassifiers()
        # Initialize fitness goal and individual type
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize individuals, populations, and evolution operators
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.createSpecies)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutGaussian, indpb=0.5, mu=25.5,
                         sigma=12.5)  # 50% chance for each value to mutate
        toolbox.register("mutate2", tools.mutShuffleIndexes, indpb=0.5)  # 50% chance for each value to mutate
        toolbox.register("select", tools.selTournament, tournsize=5)

        # Evolve population of 100 indivduals for 2000 generations
        # 200,000 fitness evaluations
        pop = toolbox.population(n=15)
        ngen = 100

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # CXPB  is the probability with which two individuals are crossed
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.75, 0.6

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Gather best values
        best = [0.0]

        # Begin the evolution
        while max(best) < 1.0 and g < ngen:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values for ind in pop]

            print("  Max %s" % max(fits)[0])
            best.append(max(fits)[0])

            ##############################################################################
            # Everything enclosed in hash was altered by Zach from Gianni's code for
            # reference.

            # Select the next generation individuals
            bestIndv = tools.selBest(pop, k=2)

            # Elitism: put top two individuals in next generation
            nextGeneration = []
            nextGeneration += bestIndv

            # perform crossover on pairs until nextGeneration is equal in size to pop
            while len(nextGeneration) < (len(pop)-2):
                parents = tools.selRoulette(pop, k=2)
                children = toolbox.mate(parents[0], parents[1])

                # mutation of children with 'MUTPB' chance
                for child in children:
                    if random.random() < MUTPB:
                        toolbox.mutate(child)

                # add the pair of children to nextGeneration
                nextGeneration += children

                # adds a single child in the case that len(pop) is odd
                if len(nextGeneration) == (len(pop)-1):
                    parents = tools.selRoulette(pop, k=2)
                    child = toolbox.mate(parents[0], parents[1])[0]

                    # mutation of child
                    if random.random() < MUTPB:
                        toolbox.mutate(child)

                    # add the child to nextGeneration
                    nextGeneration += [child]

            ##print(f'Pop: {pop}')
            print(f'Population: {fits}')
            pop[:] = nextGeneration
            ##print(f'Offspring: {pop}')

            ##############################################################################

            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # Extracting all the fitnesses of
            fits = [ind.fitness.values[0] for ind in pop]
            print(f'Offspring: {fits}')

        print(f"----Best solution----")
        t = tools.selBest(pop, k=1)[0]
        fit = t.fitness.values
        print(f"Fitness {fit}")

    # Fitness evalutation
    def evaluate(self, individual):
        value = self.boidFitness(individual)
        return (value,)

    def loadClassifiers(self,
                        location='C:\\Users\\ZachPC\\PycharmProjects\\EvolProg Term Project\\BoidEvolution\\TreeModels'):
        ## Load the saved models
        self.alignedClass = load(os.path.join(location, 'alignedClass.joblib'))
        self.flockingClass = load(os.path.join(location, 'flockingClass.joblib'))
        self.groupedClass = load(os.path.join(location, 'groupedClass.joblib'))

        ## Save list of classifiers as an accessible class variable
        self.classifiers = [self.alignedClass, self.flockingClass, self.groupedClass]

    # def boidFitness(self,species=[1.0, 1.5, 1.35, 200, 75, 2.5]):
    # 	## Run the boid simulation
    #     count=150
    #     screen_width = 3000
    #     screen_height = screen_width
    #     num_cores = multiprocessing.cpu_count()
    #     num_processes = num_cores//2

    #     swarm = Flock(num_processes, count, screen_width, screen_height, *species)

    #     saved_data = swarm.simulate()

    #     ## Classify the instances and calculate fitness
    #     max_fit = 4 * len(saved_data.index)
    #     fit_weights = [1,2,1]

    #     classes = [classifier.predict(saved_data) for classifier in self.classifiers]

    #     ## Am hoping to provide a 1-point bonus to 'perfect' instances
    # 	#bonus = [1  if (np.sum(classes[:][i]) == 4) else 0 for i in range(100)]

    #     fits = np.dot(fit_weights,classes)

    #     fitness = np.sum(fits)/max_fit

    #     return fitness

    def boidFitness(self, species=[1.0, 1.5, 1.35, 200, 75, 2.5], seed=0, lock=None, detail=False):
        """
        Simulate the species of boid and return the fitness
        valuation.

		:param species: A list which specifies the parameters
			that define a given species of boid.
		:return: the evaluated fitness value
		"""

        ## Run the boid simulation
        count = 150
        screen_width = 3000
        screen_height = screen_width
        # seed = random.randint(1,1e10)
        # seed = 10

        swarm = Flock(seed, count, screen_width, screen_height, *species)

        saved_data = swarm.simulate()

        ## Classify the instances and calculate fitness
        max_fit = 4 * len(saved_data.index)
        fit_weights = [1, 2, 1]

        classes = [classifier.predict(saved_data) for classifier in self.classifiers]

        ## Am hoping to provide a 1-point bonus to 'perfect' instances
        # bonus = [1  if (np.sum(classes[:][i]) == 4) else 0 for i in range(100)]

        fits = np.dot(fit_weights, classes)

        fitness = np.sum(fits) / max_fit

        if detail:
            detail_fits = [np.sum(classes[i]) / len(classes[i]) for i in range(len(classes))]
            return fitness, detail_fits

        else:
            return fitness

    def createSpecies(self):
        return [round(random.uniform(0.10, 10.00), 2),  ## Weight of the alignment force
                round(random.uniform(0.10, 10.00), 2),  ## Weight of the separation force
                round(random.uniform(0.10, 10.00), 2),  ## Weight of the cohesion force
                round(random.uniform(0.10, 10.00), 2),  ##random.randint(1, 500), ## Radius for alignment/cohesion
                round(random.uniform(0.10, 10.00), 2),  ##random.randint(1, 200), ## Radius for separation
                round(random.uniform(0.10, 10.00), 2)  ## Maximum acceleration
                ]


def main():
    GenerationalGA()


if __name__ == '__main__':
    main()

