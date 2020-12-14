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


	def loadClassifiers(self,location='./TreeModels/'):
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


	def boidFitness(self, species, seed=0, current_evals=3000, eval_limit=3000, detail=False):
		"""
		Simulate the species of boid and return the fitness 
		valuation.

		:param species: A list which specifies the parameters 
			that define a given species of boid.
		:return: the evaluated fitness value
		"""

		## Run the boid simulation
		count=200
		screen_width = 3000
		screen_height = screen_width
		#seed = random.randint(1,1e10)		
		#seed = 10

		swarm = Flock(seed, count, screen_width, screen_height, *species)

		saved_data = swarm.simulate()


		## Classify the instances and calculate fitness
		progress = current_evals/eval_limit
		if progress <= 1/3:
			alignWeight = 1.0
			flockWeight = 0.0 
			groupWeight = 0.0
			fit_weights = [alignWeight,flockWeight,groupWeight]

			classes = [classifier.predict(saved_data) for classifier in self.classifiers]
			detail_fits = [np.sum(classes[i])/len(classes[i]) for i in range(len(classes))]
			weighting = 1
			
		elif progress <= 2/3:
			alignWeight = 2/3
			flockWeight = 1/6 
			groupWeight = 1/6
			fit_weights = [alignWeight,flockWeight,groupWeight]
			
			classes = [classifier.predict(saved_data) for classifier in self.classifiers]
			detail_fits = [np.sum(classes[i])/len(classes[i]) for i in range(len(classes))]
			weighting = self.fitDifferences(detail_fits)

		else:
			alignWeight = 1/3
			flockWeight = 1/3
			groupWeight = 1/3
			fit_weights = [alignWeight,flockWeight,groupWeight]

			classes = [classifier.predict(saved_data) for classifier in self.classifiers]
			detail_fits = [np.sum(classes[i])/len(classes[i]) for i in range(len(classes))]
			weighting = self.fitDifferences(detail_fits)
			
		fitness = np.dot(fit_weights,detail_fits)
		final_fit = fitness*weighting

		if detail:
			return final_fit, detail_fits, weighting
		else:
			return final_fit


	def fitDifferences(self,detail_fits):
		"""
		Returns a weight in the range [1/9,1] that rewards 
		detail_fits which balance the behaviors rather than 
		favor/neglect any.

		:param detail_fits: a triple of fitness values for a 
			single boid with respect to aligned, flocking, 
			grouped
		:return: a weight in range [1,3]
		"""

		difference_measure = np.sum([abs(first-second) for i,first in enumerate(detail_fits) for second in detail_fits[i+1:]])

		return_val = 1/(1+4*difference_measure)

		return return_val


	#def listFitness(self,species_list=[[1.0, 1.5, 1.35, 200.0, 75.0, 2.5],[1.2, 1.1, 1.5, 100.0, 150.0, 5.0]]*10):
	#	"""
	#	Parallelizes the evaluation of multiple fitness functions 
	#	for each of the species of boid included in species_list

	#	:param species_list: the list of boid species whose 
	#		fitness is to be evaluated
	#	:return: the coresponding list of fitnesses
	#	"""

	#	## Define a seed to evaluate new species on same 
	#	#	initial conditions
	#	seed = random.randint(1,1e10)

	#	## Determine the number of processors to use
	#	num_cores = mp.cpu_count()
	#	num_processes = num_cores-2 if num_cores-2>=2 else 1 

	#	if num_processes != 1:
	#		## Run in parallel
	#		if __name__=="__main__":
	#			with mp.Pool(num_processes) as pool:
	#				fitnesses = pool.starmap(self.boidFitness, [(boid,seed) for boid in species_list])
	#	else:
	#		## Run in linear
	#		fitnesses = [self.boidFitness(boid) for boid in species_list]

	#	return fitnesses



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

	def ryanEvolve(self):
		"""
		Test function to see what kind of behavior is evolved.
		Uses a Mu+Lambda approach.
		"""

		mu = 50
		lambda_ = 5 * mu
		CXPB = 0.9
		MUTPB = 0.1
		self.eval_limit = 3000

		## Specifies the bounds for our parameters; pass to the 
		## toolbox.decorate methods with mate and mutate
		wMin= 1e-2
		wMax= 2.0
		rMin= 10.0
		rMax= 300.0
		aMin= 1e-2
		aMax= 5.0
		sWMin = 0.5 * wMin
		sWMax = 0.2 * wMax
		sRMin = 0.5 * rMin		
		sRMax = 0.2 * rMax		
		sAMin = 0.5 * aMin		
		sAMax = 0.2 * aMax

		parameter_bounds = [wMin, wMax, rMin, rMax, aMin, aMax]
		strategy_bounds = [sWMin, sWMax, sRMin, sRMax, sAMin, sAMax]

		## Create Fitness, Individuals, and Strategy variables
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax, strategy=None)
		creator.create("Strategy", array.array, typecode="d")

		## Sets up our evolutionary approach
		toolbox = base.Toolbox()

		toolbox.register('individual', self.initBoids, creator.Individual, creator.Strategy, *parameter_bounds, *strategy_bounds)
		toolbox.register('population', tools.initRepeat, list, toolbox.individual)

		toolbox.register("mate", tools.cxESBlend, alpha=0.333)
		toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=1/6)
		toolbox.register('select', tools.selRandom)
		toolbox.register('evaluate', self.boidFitness)

		toolbox.decorate('mate', self.checkBounds(parameter_bounds))
		toolbox.decorate('mutate', self.checkBounds(parameter_bounds))

		toolbox.decorate('mate', self.checkStrategy(strategy_bounds))
		toolbox.decorate('mutate', self.checkStrategy(strategy_bounds))

		## Prepares multiprocessing variables
		num_cores = mp.cpu_count()
		num_processes = num_cores-2 if num_cores-2>=2 else 1 
		

		
		## Initializes the population with n individuals
		g = 0
		total_evals = 0
		pop = toolbox.population(n=lambda_)  # Start with lambda_ initial individuals to explore more of the space
		seed = random.randint(1,1e10)

		## Evaluate the entire population for fitness in parallel
		if __name__=="__main__":
			with mp.Pool(num_processes) as pool:
				fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,total_evals,self.eval_limit) for boid in pop])

		## Apply fitness values
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
		pop = list(map(toolbox.clone, pop[-mu:])) #Ensures top mu are initial population
		record = stats.compile(pop)
		total_evals = lambda_
		logbook = tools.Logbook()
		logbook.header = 'gen','evals','min','max','avg','std'
		logbook.record(gen=0, evals=total_evals, **record)
		print(logbook.stream)
		hof.update(pop)
		print(hof[0])
		print(hof[0].fitness.values[0])

		## Loop for each generation until stopping criteria
		while total_evals < (self.eval_limit - lambda_):
			g+=1
			## Select the next generation individuals
			offspring = toolbox.select(pop, lambda_)

			## Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			## Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < CXPB:
					toolbox.mate(child1, child2)
					del child1.fitness.values
					del child2.fitness.values

			for mutant in offspring:
				if random.random() < MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values

			## Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			seed = random.randint(1,1e10)
			if __name__=="__main__":
				with mp.Pool(num_processes) as pool:
					fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,total_evals,self.eval_limit) for boid in invalid_ind])

			## Apply found fitness values
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit,


			#### MuPlusLambda
			## New generation
			new_gen = list(map(toolbox.clone, [*pop,*offspring]))

			## Sort the new generation by fitness
			new_gen.sort(key=lambda x: x.fitness.values[0])

			## Replace population with top mu of new generation
			pop = list(map(toolbox.clone, new_gen[-mu:]))
			####


			##### MuCommaLambda
			### Sort the new generation by fitness
			#offspring.sort(key=lambda x: x.fitness.values[0])

			### Replace population with top mu of new generation
			#pop = list(map(toolbox.clone, offspring[-mu:]))
			#####


			## Record the new generation
			hof.update(pop)
			record = stats.compile(pop)
			total_evals += len(invalid_ind)
			logbook.record(gen=g, evals=total_evals, **record)
			print(logbook.stream)
			print(hof[0])
			print(hof[0].fitness.values[0])
			bestFit, bestDetailFit, bestFitWeight = self.boidFitness(hof[0].tolist(),seed,total_evals-len(invalid_ind),self.eval_limit,detail=True)
			print(bestDetailFit)

		gen = logbook.select("gen")
		fit_maxes = logbook.select("max")
		size_avgs = logbook.select("avg")

		print(f'Best Score: {hof[0].fitness.values[0]}')
		print(f'Best Individual: {hof[0]}')

		## Plot evolution
		fig, ax1 = plt.subplots()
		line1 = ax1.plot(gen, fit_maxes, "b-", label="Maximum Fitness")
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

	def initBoids(self,ind_cls, strg_cls, wMin, wMax, rMin, rMax, aMin, aMax, sWMin, sWMax, sRMin, sRMax, sAMin, sAMax):
		#"""
		#An initializer to create individuals for the Ackley 
		#Function Evolution.
		#"""

		nWeights = 3
		nRadii = 2

		ind = ind_cls((*[np.random.uniform(wMin,wMax) for _ in range(nWeights)],*[np.random.uniform(rMin,rMax) for _ in range(nRadii)],*[np.random.uniform(aMin,aMax)]))
		ind.strategy = strg_cls((*[np.random.uniform(sWMin,sWMax) for _ in range(nWeights)],*[np.random.uniform(sRMin,sRMax) for _ in range(nRadii)],np.random.uniform(sAMin,sAMax)))
		return ind

	def checkBounds(self,bounds):
		#"""
		#A wrapper to correct for parameters that go outside of the 
		#parameter's desired bounds.
		#"""
		def decorator(func):
			def wrapper(*args, **kargs):
				offspring = func(*args, **kargs)
				for child in offspring:
					for i,p in enumerate(child):
						## Identify the appropriate bounds for the strategy
						if 0 <= i <= 2:
							min = bounds[0]
							max = bounds[1]
						elif 3 <= i <= 4:
							min = bounds[2]
							max = bounds[3]
						elif i == 5:
							min = bounds[4]
							max = bounds[5]
						## Check bounds and correct if necessary
						if p < min:
							child[i] = min
						elif p > max:
							child[i] = max
					#for i in range(3):
					#	if child[i] < wMin:
					#		child[i] = wMin
					#	elif child[i] > wMax:
					#		child[i] = wMax
					#for i in range(3,6):
					#	if child[i] < rMin:
					#		child[i] = rMin
					#	elif child[i] > rMax:
					#		child[i] = rMax
					#if child[-1] < aMin:
					#	child[-1] = aMin
					#elif child[-1] > aMax:
					#	child[-1] = aMax
				return offspring
			return wrapper
		return decorator

	def checkStrategy(self,bounds):
		"""
		A wrapper to correct for strategy variables that go below
		a minimum value.
		"""

		def decorator(func):
			def wrappper(*args, **kargs):
				children = func(*args, **kargs)
				for child in children:
					for i, s in enumerate(child.strategy):
						## Identify the appropriate bounds for the strategy
						if 0 <= i <= 2:
							min = bounds[0]
							max = bounds[1]
						elif 3 <= i <= 4:
							min = bounds[2]
							max = bounds[3]
						elif i == 5:
							min = bounds[4]
							max = bounds[5]
						## Check bounds and correct if necessary
						if s < min:
							child.strategy[i] = min
						elif s > max:
							child.strategy[i] = max
				return children
			return wrappper
		return decorator

	

	

def main():
	evolution = BoidEvolution()
	evolution.loadClassifiers(location='./GBCModels/')
	#test_ind = [1.4615518242421464, 1.2793818826191314, 0.014844554893147854, 188.9352702321177, 1.0, 1.0]
	#testFit, testDetailFit, testFitWeight = evolution.boidFitness([0.9023762847051067, 0.30433293175946874, 0.02953153397374242, 137.40745609213968, 83.98444330294794, 1.2727552672946598],detail=True)
	#print(f"Test Individual Fitness: {testFit}")
	#print(f"Test Detail Fitness: {testDetailFit}")

	best_ind = evolution.ryanEvolve()
	bestFit, bestDetailFit, bestFitWeight = evolution.boidFitness(best_ind.tolist(),detail=True)
	print(f"Best Individual Fitness: {bestFit}")
	print(f"Best Detail Fitness: {bestDetailFit}")

	count=200
	screen_width = 3000
	screen_height = screen_width
	swarm = Flock(random.randint(1,1e10), count, screen_width, screen_height, *best_ind.tolist())
	swarm.animate()

def test():
	evolution = BoidEvolution()
	evolution.loadClassifiers(location='./GBCModels/')
	test_ind = [1.75, 0.5, 0.5, 200, 10, 2.5]
	testFit, testDetailFit, testFitWeight = evolution.boidFitness(test_ind,detail=True)
	print(f"Test Individual Fitness: {testFit}")
	print(f"Test Detail Fitness: {testDetailFit}")
	
	count=200
	screen_width = 3000
	screen_height = screen_width
	swarm = Flock(random.randint(1,1e10), count, screen_width, screen_height, *test_ind)
	swarm.animate()



if __name__ == '__main__':
	main()
	#test()