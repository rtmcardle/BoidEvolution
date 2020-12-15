from typing import List
from itertools import repeat
from BoidEvolution import BoidEvolution
import random
import time
import math


try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


Genome = List[float]
Population = List[Genome]


def generate_genome() -> Genome:
    genome = []

    genome += [random.uniform(0, 5)]
    genome += [random.uniform(0, 5)]
    genome += [random.uniform(0, 1)]
    genome += [random.uniform(100, 500)]
    genome += [random.uniform(0, 5)]
    genome += [random.uniform(0, 5)]

    return genome


def generate_population(pop_size: int) -> Population:
    return [generate_genome() for i in range(pop_size)]


def fitness(genome: Genome, evolution: BoidEvolution) -> float:
    print("reached fitness")
    return evolution.boidFitness(genome, detail=False)


def selection(population: Population) -> Population:
    return random.choices(
        population=population,
        weights=[(1/(fitness(genome, evolution))) for genome in population],
        k=2
    )


def gaussian_mutation(genome: Genome, mu, sigma, indpb):
    size = len(genome)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            genome[i] += random.gauss(m, s)

    return genome


def whole_arithmetic_cx(parent1: Genome, parent2: Genome):
    rand_num = random.random()
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        child1 += [(rand_num * parent1[i] + (1-rand_num) * parent2[i])]
        child2 += [((1-rand_num) * parent1[0] + rand_num * parent2[0])]

    return [child1, child2]


def run_algorithm(evolution: BoidEvolution):
    optimal_fitness = 0
    pop_size = 6
    fitness_calls = 0
    max_fitness_calls = 200000
    num_generations = 0
    mutation_prob = .5

    population = generate_population(pop_size)
    population = sorted(    # sort the init population
        population,
        key=lambda genome: fitness(genome, evolution),
        reverse=False
    )
    fitness_calls += pop_size

    while True:
        # if the optimal solution is already present, then break
        if fitness(population[0], evolution) == optimal_fitness:
            break
        fitness_calls += 1

        # if we have exceeded the max num of fitness calls, then break
        if fitness_calls >= max_fitness_calls:
            break

        # select two parents (Roulette)
        parents = selection(population)
        # perform recombination on the parents
        children = whole_arithmetic_cx(parents[0], parents[1])
        # mutate the children (Gaussian)
        for i in range(len(children)):
            children[i] = gaussian_mutation(children[i], 0, 7, mutation_prob)

        # add children to the population
        population += children

        # sort to put the children in the correct order
        population = sorted(
            population,
            key=lambda genome: fitness(genome, evolution),
            reverse=False
        )
        fitness_calls += pop_size

        # the worst 2 individuals do not live to the next generation
        next_generation = population[0:pop_size]

        population = next_generation
        num_generations += 1

    population = sorted(
        population,
        key=lambda genome: fitness(genome, evolution),
        reverse=False
    )
    fitness_calls += pop_size

    print(f"Number of generations: {num_generations}")
    print(f"Fitness function calls: {fitness_calls}")
    print(f"Individual with fitness 0: {population[0]}")
    print(f"Fitness of this individual: {fitness(population[0], evolution)}")

    return population


if __name__ == "__main__":
    start = time.time()
    evolution = BoidEvolution()
    evolution.loadClassifiers()
    test = [1.4, 1.2, 0.0148, 150, 1.0, 1.0]
    # run_algorithm(evolution) #
    end = time.time()
    print(f"Time: {end - start}s")
