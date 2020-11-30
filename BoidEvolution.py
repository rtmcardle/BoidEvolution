###############################################################
####
####	Ryan McArdle 2020
####
####	A framework for the Evolutionary Computation Boid
####	Evolution Group Project. Each group member should 
####	select one of the evolve******() methods to work with 
####	to explore the best parameters for evolution (good 
####	values for mu, lambda, population size, selection 
####	pressure methods, etc.). This compilation for 
####	submission can then run each of these codes with the
####	optimized values and we can discuss the results.
####
####	Notes:
####	A species of boid should be a list with the following
####	elements:
####	[alignWeight,		## Weight of the alignment force
####	 sepWeight,			## Weight of the separation force
####	 cohWeight,			## Weight of the cohesion force
####	 alignCohRadius,	## Radius for alignment/cohesion
####	 sepRadius,			## Radius for separation
####	 maxAccel,			## Maximum acceleration
####	]
####
###############################################################
from BoidSim import Flock

from joblib import load, Parallel, delayed
import os
import sklearn
import multiprocessing as mp
import numpy as np
import random
from deap import creator, base, tools, algorithms
import array, datetime
import matplotlib.pyplot as plt





class BoidEvolution():
	"""
	A class for evolving a species of boid that satisfies the 
	BoidSwarmBehavior classifier.
	"""

	def __init__(self):
		"""
		Initial/main method.
		"""


	def evolve(self,method):
		"""
		Method for evolving a species of boid using the 
		provided method.

		:param method: the method to be used
		:returns: the evolved species
		"""

		## If method == any of the types (steady, generational, mucomma, muplus)
		#		run the method for that type

		## Return the evolved species


	def loadClassifiers(self,location='./ClassifierModels/'):
		"""
		Loads the pre-trained classification model for use in
		the fitness function.

		:param location: the path to the saved model
		"""

		## Load the saved models
		self.alignedClass = load(os.path.join(location,'alignedClass.joblib'))
		self.flockingClass = load(os.path.join(location,'flockingClass.joblib'))
		self.groupedClass = load(os.path.join(location,'groupedClass.joblib'))

		## Save list of classifiers as an accessible class variable
		self.classifiers = [self.alignedClass, self.flockingClass, self.groupedClass]


	def boidFitness(self, species=[1.0, 1.5, 1.35, 200, 75, 2.5], seed=0, lock=None):
		"""
		Simulate the species of boid and return the fitness 
		valuation.

		:param species: A list which specifies the parameters 
			that define a given species of boid.
		:return: the evaluated fitness value
		"""

		## Run the boid simulation
		count=150
		screen_width = 3000
		screen_height = screen_width
		#seed = random.randint(1,1e10)		
		#seed = 10

		swarm = Flock(seed, count, screen_width, screen_height, *species)

		saved_data = swarm.simulate()


		## Classify the instances and calculate fitness
		max_fit = 4 * len(saved_data.index)
		fit_weights = [1,2,1]

		classes = [classifier.predict(saved_data) for classifier in self.classifiers]
		

		## Am hoping to provide a 1-point bonus to 'perfect' instances 
		#bonus = [1  if (np.sum(classes[:][i]) == 4) else 0 for i in range(100)]

		fits = np.dot(fit_weights,classes)

		fitness = np.sum(fits)/max_fit

		return fitness

	def listFitness(self,species_list=[[1.0, 1.5, 1.35, 200.0, 75.0, 2.5],[1.2, 1.1, 1.5, 100.0, 150.0, 5.0]]*10):
		"""
		Parallelizes the evaluation of multiple fitness functions 
		for each of the species of boid included in species_list

		:param species_list: the list of boid species whose 
			fitness is to be evaluated
		:return: the coresponding list of fitnesses
		"""

		## Define a seed to evaluate new species on same 
		#	initial conditions
		seed = random.randint(1,1e10)

		## Determine the number of processors to use
		num_cores = mp.cpu_count()
		num_processes = num_cores-2 if num_cores-2>=2 else 1 

		if num_processes != 1:
			## Run in parallel
			if __name__=="__main__":
				## Pool option
				pool = mp.Pool(num_processes)
				fitnesses = pool.starmap(self.boidFitness, [(boid,seed) for boid in species_list])
				pool.close()
				## Parallel option
				#fitnesses = Parallel(n_jobs=num_processes)(delayed(self.boidFitness)(boid) for boid in species_list)
		else:
			## Run in linear
			fitnesses = [self.boidFitness(boid) for boid in species_list]

		return fitnesses



	def evolveSteady(self):
		"""
		A method to evolve a species of boid using a 
		steady-state genetic algorithm.

		:returns: the evolved species, and its fitness
		"""

		## Run the evolution

		## Record the statistics about the evolution
		########
		#	Note: It may be helpful to encode the parameters
		#	that resulted in the statistic as well. I.e. a log
		#	that reports the parameters used and the evolution
		#	that took place.
		########

		## Return the evolved species and its fitness


	def evolveGeneration(self):
		"""
		A method to evolve a species of boid using a 
		generational genetic algorithm.

		:returns: the evolved species, and its fitness
		"""
		
		## Run the evolution

		## Record the statistics about the evolution
		########
		#	Note: It may be helpful to encode the parameters
		#	that resulted in the statistic as well. I.e. a log
		#	that reports the parameters used and the evolution
		#	that took place.
		########

		## Return the evolved species and its fitness


	def evolveMuCommaLambda(self,mu,lambda_):
		"""
		A method to evolve a species of boid using a 
		(Mu,Lambda) evolutionary strategy.

		:param mu: the parent population size
		:param lambda_: the number of children generated
		:returns: the evolved species, and its fitness
		"""
		
		## Run the evolution

		## Record the statistics about the evolution
		########
		#	Note: It may be helpful to encode the parameters
		#	that resulted in the statistic as well. I.e. a log
		#	that reports the parameters used and the evolution
		#	that took place.
		########

		## Return the evolved species and its fitness


	def evolveMuPlusLambda(self,mu,lambda_):
		"""
		A method to evolve a species of boid using a 
		(Mu + Lambda) evolutionary strategy.

		:param mu: the parent population size
		:param lambda_: the number of children generated
		:returns: the evolved species, and its fitness
		"""
		
		## Run the evolution

		## Record the statistics about the evolution
		########
		#	Note: It may be helpful to encode the parameters
		#	that resulted in the statistic as well. I.e. a log
		#	that reports the parameters used and the evolution
		#	that took place.
		########

		## Return the evolved species and its fitness

	def myEvolve(self):
		"""
		Test function to see what kind of behavior is evolved.
		"""

		mu = 100
		lambda_ = 7 * mu
		CXPB = 0.95
		MUTPB = 0.15
		eval_limit = 2000

		## Specifies the number of dimensions for Ackley's
		#dim = 30

		## Specifies the bounds for our parameters; pass to the 
		## toolbox.decorate methods with mate and mutate
		wMin= 0.0
		wMax= 2.0
		rMin= 0.0
		rMax= 300.0
		aMin= 0.0
		aMax= 5.0
		smin= 1e-2
		smax= 5

		## Create Fitness, Individuals, and Strategy variables
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax, strategy=None)
		creator.create("Strategy", array.array, typecode="d")

		## Sets up our evolutionary approach
		toolbox = base.Toolbox()

		toolbox.register('individual', initBoids, creator.Individual, creator.Strategy, wMin, wMax, rMin, rMax, aMin, aMax, smin, smax)
		toolbox.register('population', tools.initRepeat, list, toolbox.individual)

		toolbox.register("mate", tools.cxESBlend, alpha=0.333)
		toolbox.register("mutate", tools.mutESLogNormal, c=0.01, indpb=0.15)
		toolbox.register('select', tools.selRandom)
		toolbox.register('evaluate', self.boidFitness)

		toolbox.decorate('mate', checkBounds(wMin, wMax, rMin, rMax, aMin, aMax))
		toolbox.decorate('mutate', checkBounds(wMin, wMax, rMin, rMax, aMin, aMax))

		toolbox.decorate('mate', checkStrategy(smin))
		toolbox.decorate('mutate', checkStrategy(smin))

		num_cores = mp.cpu_count()
		num_processes = num_cores-4 if num_cores-4>=2 else 1 
		

		
		## Initializes the population with n individuals
		pop = toolbox.population(n=mu)
		seed = random.randint(1,1e10)

		## Evaluate the entire population for fitness
		if __name__=="__main__":
			#lock = mp.Lock()
			#pool = mp.Pool(num_processes)
			with mp.Pool(num_processes) as pool:
				fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed) for boid in pop])
		#fitnesses = map(toolbox.evaluate, pop)
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit,

		## Records statistics about the population during evolution
		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		## Records the 1 best individual seen in evolution
		hof = tools.HallOfFame(1)

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
				seed = random.randint(1,1e10)
				#fitnesses = map(toolbox.evaluate, invalid_ind)
				if __name__=="__main__":
					#lock = mp.Lock()
					#pool = mp.Pool(num_processes)
					with mp.Pool(num_processes) as pool:
						fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed) for boid in invalid_ind])
				for ind, fit in zip(invalid_ind, fitnesses):
					ind.fitness.values = fit,

				# Sort the offspring by fitness
				offspring.sort(key=lambda x: x.fitness.values[0])

				# Replace population with top mu offspring
				pop = list(map(toolbox.clone, offspring[mu:]))

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
		prob_dir = os.path.join(cur_dir,'runs_record\\')
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

	def myEvolve2():
		# Test built-in DEAP algos
		return

def initBoids(ind_cls, strg_cls, wMin, wMax, rMin, rMax, aMin, aMax, smin, smax):
    #"""
    #An initializer to create individuals for the Ackley 
    #Function Evolution.
    #"""

	nWeights = 3
	nRadii = 2

	ind = ind_cls((*[np.random.uniform(wMin,wMax) for _ in range(nWeights)],*[np.random.uniform(rMin,rMax) for _ in range(nRadii)],*[np.random.uniform(aMin,aMax)]))
	#ind = ind_cls(	(	np.random.uniform(	dmin,dmax	) for _ in range(	dim	)	)	)
	ind.strategy = strg_cls((np.random.uniform(smin,smax) for _ in range(nWeights+nRadii+1)))
	return ind

def checkBounds(wMin, wMax, rMin, rMax, aMin, aMax):
    #"""
    #A wrapper to correct for parameters that go outside of the 
    #parameter's desired bounds.
    #"""
	def decorator(func):
		def wrapper(*args, **kargs):
			offspring = func(*args, **kargs)
			for child in offspring:
				for i in range(3):
					if child[i] < wMin:
						child[i] = wMin
					elif child[i] > wMax:
						child[i] = wMax
				for i in range(3,6):
					if child[i] < rMin:
						child[i] = rMin
					elif child[i] > rMax:
						child[i] = rMax
				if child[-1] < aMin:
					child[-1] = aMin
				elif child[-1] > aMax:
					child[-1] = aMax
			return offspring
		return wrapper
	return decorator

def checkStrategy(smin):
    """
    A wrapper to correct for strategy variables that go below
    a minimum value.
    """

    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < smin:
                        child.strategy[i] = smin
            return children
        return wrappper
    return decorator



def main():
	evolution = BoidEvolution()
	evolution.loadClassifiers()
	#fit = evolution.listFitness()
	best_ind = evolution.myEvolve()

	count=150
	screen_width = 3000
	screen_height = screen_width
	swarm = Flock(random.randint(1,1e10), count, screen_width, screen_height, *best_ind.tolist())
	swarm.animate()
	print(f"Sample Species Fitnesses: {best_ind.fitness}")


if __name__ == '__main__':
	main()