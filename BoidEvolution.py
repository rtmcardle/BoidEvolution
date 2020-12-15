###############################################################
####
####	Evolving Boid Flocking Behavior
####
####	Code produced by: Ryan McArdle, Gianni Orlando, Olusade
####		Calhoun, and Zach Sipper
####	Term Project
####	CSCI 4560/6560: Evolutionary Computation
####	Rasheed, K.
####	17 December 2020
####
####	A framework for the Evolutionary Computation Boid
####	Evolution Group Project. Each group member was tasked 
####	with selecting one of the evolve******() methods to 
####	develop and explore the performance of the given method
####	on the probelm. This file compiles for submission each 
####	of these codes with uniform plotting codes so results 
####	are easily compared.
####
####	Each method credits its primary author with additional
####	acknowledgement provided.
####
####	Compilation and final edits of methods performed by 
####	Ryan McArdle, Dec. 2020.
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

		self.loadClassifiers()


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


	def prepareEvolution(self):
		"""
		Establishes parameters that can be applied uniformly 
		for the evolutionary methods.
		"""

		## Evolutionary parameters and limits
		self.mu = 100  ### Also defines population size for generational and steady GAs
		self.lambda_ = 7 * self.mu

		self.current_evals = 0
		self.eval_limit = 10000
		self.eval_limit_steady = self.eval_limit/10  ### Lower limit for steady
		


		self.CXPB = 0.9
		self.MUTPB = 0.3

		## Parameter bounds for all methods
		wMin= 1e-2
		wMax= 2.0
		rMin= 100.0
		rMax= 300.0
		aMin= 1e-2
		aMax= 5.0
		self.parameter_bounds = [wMin, wMax, rMin, rMax, aMin, aMax]

		## Strategy bounds for evolutionary strategy methods
		sWMin = 0.5 * wMin
		sWMax = 0.2 * wMax
		sRMin = 0.5 * rMin		
		sRMax = 0.2 * rMax		
		sAMin = 0.5 * aMin		
		sAMax = 0.2 * aMax
		self.strategy_bounds = [sWMin, sWMax, sRMin, sRMax, sAMin, sAMax]

		## Simulation parameters
		self.count=200
		self.screen_width = 3000
		self.screen_height = self.screen_width

		
		## Multiprocessing parameters
		num_cores = mp.cpu_count()
		self.num_processes = num_cores-2 if num_cores-2>=2 else 1 


	def plotAndRecord(self,logbook,hof,name):
		### Plotting Code
		gen = logbook.select("gen")
		fit_maxes = logbook.select("max")
		size_avgs = logbook.select("avg")
		evals = logbook.select("evals")
		first_gen_break = [n for n,eval in enumerate(evals) if eval>=self.eval_limit/3][0] + 0.5
		second_gen_break = [n for n,eval in enumerate(evals) if eval>=2*self.eval_limit/3][0] + 0.5


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
		
		ax1.axvline(x=first_gen_break)
		ax1.axvline(x=second_gen_break)


		## Record run
		timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
		cur_dir = os.getcwd()
		prob_dir = os.path.join(cur_dir,'runs_record\\')
		if not os.path.exists(prob_dir):
			os.mkdir(prob_dir)
		time_dir = os.path.join(prob_dir,f'{timestamp}{name}\\')
		os.mkdir(time_dir)
		plt.savefig(os.path.join(time_dir,'plot.png'))

		## Log the run
		log_file = os.path.join(time_dir,'log.txt')
		with open(log_file, 'w') as f:
			print(f'Best Score: {hof[0].fitness.values[0]}', file=f)
			print(f'Best Individual: {hof[0]}\n', file=f)
			print(f'History: ', file=f)
			print(logbook, file=f)


	def boidFitness(self, species, seed=0, current_evals=3000, eval_limit=3000, detail=False):
		"""
		Simulate the species of boid and return the fitness 
		valuation.

		:param species: A list which specifies the parameters 
			that define a given species of boid.
		:return: the evaluated fitness value
		"""

		## Run the boid simulation
		self.count=200
		self.screen_width = 3000
		self.screen_height = self.screen_width
		
		swarm = Flock(seed, self.count, self.screen_width, self.screen_height, *species)

		saved_data = swarm.simulate()


		## Classify the instances and calculate fitness
		progress = self.current_evals/self.eval_limit
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


	def evolveSteady(self):
		"""
		A method to evolve a species of boid using a 
		steady-state genetic algorithm. Method authored by 
		Gianni Orlando. Edited by Ryan McArdle.

		:returns: the evolved species, and its fitness
		"""
		name = "Steady"

		self.prepareEvolution()

		self.eval_limit = self.eval_limit_steady

		# Initialize fitness goal and individual type
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", array.array, typecode='d',fitness=creator.FitnessMax)

        # Initialize individuals, populations, and evolution operators
		toolbox = base.Toolbox()
		toolbox.register("individual", self.initBoids, creator.Individual, *self.parameter_bounds)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.boidFitness)
		toolbox.register("mate", tools.cxOnePoint)
		toolbox.register("mutate", tools.mutGaussian, indpb=0.5, mu=25.5, sigma=12.5) # 50% chance for each value to mutate
		toolbox.register("mutate2", tools.mutShuffleIndexes, indpb=0.5) # 50% chance for each value to mutate
		toolbox.register("select", tools.selTournament, tournsize= 5)

		toolbox.decorate('mate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mutate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mutate2', self.checkBounds(self.parameter_bounds))

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)
		self.logbook = tools.Logbook()
		self.logbook.header = 'gen','evals','min','max','avg','std'
		hof = tools.HallOfFame(1)

		pop = toolbox.population(n=self.mu)

        # Evaluate the entire population
		seed = random.randint(1,1e10)
		## Evaluate the entire population for fitness in parallel
		if __name__=="__main__":
			with mp.Pool(self.num_processes) as pool:
				fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in pop])

		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit,

        # Extracting all the fitnesses of 
		fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
		g = 0

		## Record the initial population
		self.current_evals += len(pop)
		record = stats.compile(pop)
		logbook = tools.Logbook()
		logbook.header = 'gen','evals','min','max','avg','std'
		logbook.record(gen=0, evals=self.current_evals, **record)
		print(logbook.stream)
		hof.update(pop)
		print(hof[0])
		print(hof[0].fitness.values[0])
		
		# Begin the evolution
		while hof[0].fitness.values[0] < 1.0 and self.current_evals < self.eval_limit:
            # A new generation
			g = g + 1

             # Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values for ind in pop]

            # Select the next generation individuals
			offspring = pop
			bestIndv = tools.selBest(offspring,k=2)
			worstIndv = tools.selWorst(offspring,k=2)
            # Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

            # Apply crossover the two individuals (tournmanet 
            # selection)
			parents = toolbox.select(offspring,2)
			parent1, parent2 = parents[0], parents[1]
			replace1 = offspring[offspring.index(worstIndv[0])]
			replace2 = offspring[offspring.index(worstIndv[1])]
			for child1, child2 in zip([parent1], [parent2]):
				if random.random() < self.CXPB:
					toolbox.mate(child1, child2)
					if random.random() < self.MUTPB:
						toolbox.mutate(child1)
						del child1.fitness.values
					if random.random() < self.MUTPB:
						toolbox.mutate(child2)
						del child2.fitness.values
					if random.random() < self.MUTPB:
						toolbox.mutate2(child1)
						del child1.fitness.values
					if random.random() < self.MUTPB:
						toolbox.mutate2(child2)
						del child1.fitness.values
					offspring[offspring.index(replace1)] = parent1
					offspring[offspring.index(replace2)] = parent2

			pop[:] = offspring

			# Evaluate the population with invalid fitnesses
			invalid_ind = [ind for ind in pop if not ind.fitness.valid]
			seed = random.randint(1,1e10)
			if __name__=="__main__":
				with mp.Pool(self.num_processes) as pool:
					fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in invalid_ind])

			## Apply found fitness values
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit,

            # Extracting all the fitnesses of 
			fits = [ind.fitness.values[0] for ind in pop]

			##################################################
			## Record the new generation
			hof.update(pop)
			record = stats.compile(pop)
			self.current_evals += len(invalid_ind)
			logbook.record(gen=g, evals=self.current_evals, **record)
			print(logbook.stream)
			print(hof[0])
			print(hof[0].fitness.values[0])
			#bestFit, bestDetailFit, bestFitWeight = self.boidFitness(hof[0].tolist(),seed,self.current_evals-len(invalid_ind),self.eval_limit,detail=True)
			#print(bestDetailFit)
			##################################################

		self.plotAndRecord(logbook,hof,name)

		return hof[0]


	def createSpeciesSS(self):
		"""
		A function to create the species for Steady State 
		evolution. Authored by Gianni Orlando.
		"""

		return [round(random.uniform(0.10, 10.00), 2), ## Weight of the alignment force
        round(random.uniform(0.10, 10.00), 2), ## Weight of the separation force
        round(random.uniform(0.10, 10.00), 2), ## Weight of the cohesion force
        round(random.uniform(0.10, 10.00), 2), ##random.randint(1, 500), ## Radius for alignment/cohesion
        round(random.uniform(0.10, 10.00), 2), ##random.randint(1, 200), ## Radius for separation
        round(random.uniform(0.10, 10.00), 2)  ## Maximum acceleration
        ]


	def evolveGeneration(self):
		"""
		A method to evolve a species of boid using a 
		generational genetic algorithm. Authored by Zach Sipper,
		modified from evolveSteady() by Gianni Orlando. Edited
		by Ryan McArdle.

		:returns: the evolved species, and its fitness
		"""
		
		name = "Steady"

		self.prepareEvolution()
		
		# Initialize fitness goal and individual type
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", array.array, typecode='d',fitness=creator.FitnessMax)
		
        #Initialize individuals, populations, and evolution operators
		toolbox = base.Toolbox()
		toolbox.register("individual", self.initBoids, creator.Individual, *self.parameter_bounds)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.boidFitness)
		toolbox.register("mate", tools.cxOnePoint)
		toolbox.register("mutate", tools.mutGaussian, indpb=0.5, mu=25.5, sigma=12.5) # 50% chance for each value to mutate
		toolbox.register("mutate2", tools.mutShuffleIndexes, indpb=0.5) # 50% chance for each value to mutate
		toolbox.register("select", tools.selTournament, tournsize= 5)

		toolbox.decorate('mate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mutate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mutate2', self.checkBounds(self.parameter_bounds))

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)
		self.logbook = tools.Logbook()
		self.logbook.header = 'gen','evals','min','max','avg','std'
		hof = tools.HallOfFame(1)

		
		pop = toolbox.population(n=self.mu)

        #Evaluate the entire population
		seed = random.randint(1,1e10)
		## Evaluate the entire population for fitness in parallel
		if __name__=="__main__":
			with mp.Pool(self.num_processes) as pool:
				fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in pop])

		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit,

		#Extracting all the fitnesses of 
		fits = [ind.fitness.values[0] for ind in pop]

		#Variable keeping track of the number of generations
		g = 0

		## Record the initial population
		self.current_evals += len(pop)
		record = stats.compile(pop)
		logbook = tools.Logbook()
		logbook.header = 'gen','evals','min','max','avg','std'
		logbook.record(gen=0, evals=self.current_evals, **record)
		print(logbook.stream)
		hof.update(pop)
		print(hof[0])
		print(hof[0].fitness.values[0])
		
		#Begin the evolution
		while hof[0].fitness.values[0] < 1.0 and self.current_evals < self.eval_limit:
            #A new generation
			g = g + 1

			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values for ind in pop]
			
			############################################################
			####	Code between hash lines produced by Zach Sipper and 
			####	edited by Ryan McArdle.
			####
			# Select the next generation individuals
			bestIndv = tools.selBest(pop, k=2)

            # Elitism: put top two individuals in next generation
			nextGeneration = [*bestIndv]
			#nextGeneration += bestIndv

            # perform crossover on pairs until nextGeneration is equal in size to pop
			while len(nextGeneration) < (len(pop)-2):
				parents = toolbox.select(pop,2)
				parents = list(map(toolbox.clone,parents))
				toolbox.mate(parents[0], parents[1])
				for parent in parents:
					del parent.fitness.values

                # mutation of children with 'MUTPB' chance
				for parent in parents:
					if random.random() < self.MUTPB:
						toolbox.mutate(parent)
						del parent.fitness.values

                # add the pair of children to nextGeneration
				nextGeneration.extend(parents)

                # adds a single child in the case that len(pop) is odd
				if len(nextGeneration) == (len(pop)-1):
					parents = toolbox.select(pop,2)
					toolbox.mate(parents[0], parents[1])
					child = parents[0]
					del child.fitness.values

                    # mutation of child
					if random.random() < self.MUTPB:
						toolbox.mutate(child)
						del child.fitness.values

                    # add the child to nextGeneration
					nextGeneration.extend(list(map(toolbox.clone,child)))

			pop = list(map(toolbox.clone,nextGeneration))
			####
			####
			############################################################

			#Evaluate the population with invalid fitnesses
			invalid_ind = [ind for ind in pop if not ind.fitness.valid]
			seed = random.randint(1,1e10)
			if __name__=="__main__":
				with mp.Pool(self.num_processes) as pool:
					fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in invalid_ind])

			## Apply found fitness values
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit,

            # Extracting all the fitnesses of 
			fits = [ind.fitness.values[0] for ind in pop]

			##################################################
			## Record the new generation
			hof.update(pop)
			record = stats.compile(pop)
			self.current_evals += len(invalid_ind)
			logbook.record(gen=g, evals=self.current_evals, **record)
			print(logbook.stream)
			print(hof[0])
			print(hof[0].fitness.values[0])
			#bestFit, bestDetailFit, bestFitWeight = self.boidFitness(hof[0].tolist(),seed,self.current_evals-len(invalid_ind),self.eval_limit,detail=True)
			#print(bestDetailFit)
			##################################################

		self.plotAndRecord(logbook,hof,name)

		return hof[0]


	def evolveMuCommaLambda(self):
		"""
		A method to evolve a species of boid using a 
		(Mu,Lambda) evolutionary strategy. Authored by Ryan 
		McArdle.

		:returns: the best individual found by the evolution
		"""
		
		name = "muCommaLambda"

		self.prepareEvolution()

		## Create Fitness, Individuals, and Strategy variables
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax, strategy=None)
		creator.create("Strategy", array.array, typecode="d")

		## Sets up our evolutionary approach
		toolbox = base.Toolbox()

		toolbox.register('individual', self.initBoidsStrat, creator.Individual, creator.Strategy, *self.parameter_bounds, *self.strategy_bounds)
		toolbox.register('population', tools.initRepeat, list, toolbox.individual)

		toolbox.register("mate", tools.cxESBlend, alpha=0.333)
		toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=1/6)
		toolbox.register('select', tools.selRandom)
		toolbox.register('evaluate', self.boidFitness)

		toolbox.decorate('mate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mutate', self.checkBounds(self.parameter_bounds))

		toolbox.decorate('mate', self.checkStrategy(self.strategy_bounds))
		toolbox.decorate('mutate', self.checkStrategy(self.strategy_bounds))

		
		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)
		self.logbook = tools.Logbook()
		self.logbook.header = 'gen','evals','min','max','avg','std'
		hof = tools.HallOfFame(1)
		
		## Initializes the population with n individuals
		g = 0
		pop = toolbox.population(n=self.mu)
		seed = random.randint(1,1e10)

		## Evaluate the entire population for fitness in parallel
		if __name__=="__main__":
			with mp.Pool(self.num_processes) as pool:
				fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in pop])

		## Apply fitness values
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit,


		## Record initial population in stats logbook
		self.current_evals += len(pop)
		record = stats.compile(pop)
		logbook = tools.Logbook()
		logbook.header = 'gen','evals','min','max','avg','std'
		logbook.record(gen=0, evals=self.current_evals, **record)
		print(logbook.stream)
		hof.update(pop)
		print(hof[0])
		print(hof[0].fitness.values[0])

		## Loop for each generation until stopping criteria
		while self.current_evals < (self.eval_limit - self.lambda_):
			g+=1
			## Select the next generation individuals
			offspring = toolbox.select(pop, self.lambda_)

			## Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			## Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < self.CXPB:
					toolbox.mate(child1, child2)
					del child1.fitness.values
					del child2.fitness.values

			for mutant in offspring:
				if random.random() < self.MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values

			## Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			seed = random.randint(1,1e10)
			if __name__=="__main__":
				with mp.Pool(self.num_processes) as pool:
					fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in invalid_ind])

			## Apply found fitness values
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit,


			#### MuPlusLambda
			## New generation
			new_gen = list(map(toolbox.clone, [*pop,*offspring]))

			## Sort the new generation by fitness
			new_gen.sort(key=lambda x: x.fitness.values[0])

			## Replace population with top mu of new generation
			pop = list(map(toolbox.clone, new_gen[-self.mu:]))
			####


			##### MuCommaLambda
			### Sort the new generation by fitness
			#offspring.sort(key=lambda x: x.fitness.values[0])

			### Replace population with top mu of new generation
			#pop = list(map(toolbox.clone, offspring[-self.mu:]))
			#####

			##################################################
			## Record the new generation
			hof.update(pop)
			record = stats.compile(pop)
			self.current_evals += len(invalid_ind)
			logbook.record(gen=g, evals=self.current_evals, **record)
			print(logbook.stream)
			print(hof[0])
			print(hof[0].fitness.values[0])
			#bestFit, bestDetailFit, bestFitWeight = self.boidFitness(hof[0].tolist(),seed,self.current_evals-len(invalid_ind),self.eval_limit,detail=True)
			#print(bestDetailFit)
			##################################################

		self.plotAndRecord(logbook,hof,name)

		return hof[0]


	def evolveMuPlusLambda(self):
		"""
		A method to evolve a species of boid using a 
		(Mu + Lambda) evolutionary strategy. Method authored
		by Olusade Calhoun. Edited by Ryan McArdle.

		:returns: the best individual found by the evolution
		"""

		name = "MuPlusLambda"

		self.prepareEvolution()


		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
		creator.create("Strategy", array.array, typecode="d")

		toolbox = base.Toolbox()
		toolbox.register("individual", self.initBoidsStrat, creator.Individual, creator.Strategy, *self.parameter_bounds, *self.strategy_bounds) 
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mate", tools.cxESBlend, alpha=0.333)
		toolbox.register("mutate", tools.mutESLogNormal, c=0.01, indpb=0.1)
		toolbox.register("select", tools.selRandom)
		toolbox.register("evaluate", self.boidFitness)

		toolbox.decorate('mate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mutate', self.checkBounds(self.parameter_bounds))
		toolbox.decorate('mate', self.checkStrategy(self.strategy_bounds))
		toolbox.decorate('mutate', self.checkStrategy(self.strategy_bounds))

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)
		self.logbook = tools.Logbook()
		self.logbook.header = 'gen','evals','min','max','avg','std'
		hof = tools.HallOfFame(1)

		pop = toolbox.population(n=self.mu)
		seed = random.randint(1,1e10)

		## Evaluate the entire population for fitness in parallel
		if __name__=="__main__":
			with mp.Pool(self.num_processes) as pool:
				fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in pop])

		for ind, fit in zip(pop, fitnesses):
		    ind.fitness.values = fit,

		g = 0

		self.current_evals += len(pop)
		record = stats.compile(pop)
		logbook = tools.Logbook()
		logbook.header = 'gen','evals','min','max','avg','std'
		logbook.record(gen=0, evals=self.current_evals, **record)
		print(logbook.stream)
		hof.update(pop)
		print(hof[0])
		print(hof[0].fitness.values[0])

		## Main loop
		while True:
			if self.current_evals < self.eval_limit:
				g+=1
				offspring = toolbox.select(pop, self.lambda_)
				offspring = list(map(toolbox.clone, offspring))

				#crossover and mutation
				for child1, child2 in zip(offspring[::2], offspring[1:2]):
					if random.random() < self.CXPB:
						toolbox.mate(child1, child2)
						del child1.fitness.values
						del child2.fitness.values

				for mutant in offspring:
					if random.random() < self.MUTPB:
						toolbox.mutate(mutant)
						del mutant.fitness.values

				invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
				seed = random.randint(1,1e10)
				if __name__=="__main__":
					with mp.Pool(self.num_processes) as pool:
						fitnesses = pool.starmap(self.boidFitness, [(boid.tolist(),seed,self.current_evals,self.eval_limit) for boid in invalid_ind])

				for ind, fit in zip(invalid_ind, fitnesses):
					ind.fitness.values = fit,

				#replace population with top mu+lam offspring
				newGen = list(map(toolbox.clone, [*pop, *offspring]))

				newGen.sort(key = lambda x: x.fitness.values[0])

				pop = list(map(toolbox.clone, newGen[-self.mu:])) #selects mu highest fitness in population

				##################################################
				## Record the new generation
				hof.update(pop)
				record = stats.compile(pop)
				self.current_evals += len(invalid_ind)
				logbook.record(gen=g, evals=self.current_evals, **record)
				print(logbook.stream)
				print(hof[0])
				print(hof[0].fitness.values[0])
				#bestFit, bestDetailFit, bestFitWeight = self.boidFitness(hof[0].tolist(),seed,self.current_evals-len(invalid_ind),self.eval_limit,detail=True)
				#print(bestDetailFit)
				##################################################

			else:
				break

		self.plotAndRecord(logbook,hof,name)

		return hof[0]


	def initBoidsStrat(self,ind_cls, strg_cls, wMin, wMax, rMin, rMax, aMin, aMax, sWMin, sWMax, sRMin, sRMax, sAMin, sAMax):
		"""
		An initializer to create individuals for the Boid 
		Behavior Evolution using strategy vairables. Receives 
		the type for both individual and strategy, as well as 
		the bounds for various parameters and their associated
		strategies. Authored by Ryan McArdle.

		:return: a DEAP individual with strategy variables for 
			boid evolution
		"""

		nWeights = 3
		nRadii = 2

		ind = ind_cls((*[np.random.uniform(wMin,wMax) for _ in range(nWeights)],*[np.random.uniform(rMin,rMax) for _ in range(nRadii)],*[np.random.uniform(aMin,aMax)]))
		ind.strategy = strg_cls((*[np.random.uniform(sWMin,sWMax) for _ in range(nWeights)],*[np.random.uniform(sRMin,sRMax) for _ in range(nRadii)],np.random.uniform(sAMin,sAMax)))
		return ind


	def initBoids(self,ind_cls, wMin, wMax, rMin, rMax, aMin, aMax):
		"""
		An initializer to create individuals for the Boid 
		Behavior Evolution. Receives the type of individual as 
		well as the bounds for various parameters. Authored by 
		Ryan McArdle.

		:return: a DEAP individual for boid evolution
		"""

		nWeights = 3
		nRadii = 2

		ind = ind_cls((*[np.random.uniform(wMin,wMax) for _ in range(nWeights)],*[np.random.uniform(rMin,rMax) for _ in range(nRadii)],*[np.random.uniform(aMin,aMax)]))
		return ind


	def checkBounds(self,bounds):
		"""
		A wrapper to correct for parameters that go outside of the 
		parameter's desired bounds. Authored by Ryan McArdle; 
		modified from the DEAP documentation. 
		https://deap.readthedocs.io/en/master/
		"""
		def decorator(func):
			def wrapper(*args, **kargs):
				offspring = func(*args, **kargs)
				for child in offspring:
					for i,p in enumerate(child):
						## Identify the appropriate bounds for the parameter
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

						## Ensure separation radius is less than align/coh
						if i == 4 and p > child[3]:
							child[i] = child[3]

				return offspring
			return wrapper
		return decorator


	def checkStrategy(self,bounds):
		"""
		A wrapper to correct for strategy variables that go below
		a minimum value. Authored by Ryan McArdle; modified 
		from the DEAP documentation. 
		https://deap.readthedocs.io/en/master/
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
	evolution.loadClassifiers()
	
	##### Run Generational Algorithm
	#print(f"Generational Algorithm\n")
	#bestGen = evolution.evolveGeneration()
	#bestFit, bestDetailFit, bestFitWeight = evolution.boidFitness(bestGen.tolist(),detail=True)
	#print(f"Best Individual Fitness: {bestFit}")
	#print(f"Best Detail Fitness: {bestDetailFit}")

	#### Run Steady State Algorithm
	print(f"Steady State Algorithm\n")
	bestSteady = evolution.evolveSteady()
	bestFit, bestDetailFit, bestFitWeight = evolution.boidFitness(bestSteady.tolist(),detail=True)
	print(f"Best Individual Fitness: {bestFit}")
	print(f"Best Detail Fitness: {bestDetailFit}")

	#### Run MuPlusLambda Algorithm
	print(f"MuPlusLambda Algorithm\n")
	bestPlus = evolution.evolveMuPlusLambda()
	bestFit, bestDetailFit, bestFitWeight = evolution.boidFitness(bestPlus.tolist(),detail=True)
	print(f"Best Individual Fitness: {bestFit}")
	print(f"Best Detail Fitness: {bestDetailFit}")

	#### Run MuCommaLambda Algorithm
	print(f"MuCommaLambda Algorithm\n")
	bestComma = evolution.evolveMuCommaLambda()
	bestFit, bestDetailFit, bestFitWeight = evolution.boidFitness(bestComma.tolist(),detail=True)
	print(f"Best Individual Fitness: {bestFit}")
	print(f"Best Detail Fitness: {bestDetailFit}")


def test():
	evolution = BoidEvolution()
	evolution.loadClassifiers()
	test_ind = [2.0, 1.5, 2.0, 200, 125.0, 0.5]
	testFit, testDetailFit, testFitWeight = evolution.boidFitness(test_ind,detail=True)
	print(f"Test Individual Fitness: {testFit}")
	print(f"Test Detail Fitness: {testDetailFit}")
	
	count=200
	screen_width = 3000
	screen_height = screen_width
	swarm = Flock(random.randint(1,1e10), count, screen_width, screen_height, *test_ind)
	swarm.animate()

def testStrat():
	evolution = BoidEvolution()
	best_ind = evolution.evolveGeneration()
	print(best_ind)
	bestFit, bestDetailFit, bestFitWeight = evolution.boidFitness(best_ind.tolist(),detail=True)
	print(f"Best Individual Fitness: {bestFit}")
	print(f"Best Detail Fitness: {bestDetailFit}")

	count=200
	screen_width = 3000
	screen_height = screen_width
	swarm = Flock(random.randint(1,1e10), count, screen_width, screen_height, *best_ind.tolist())
	swarm.animate()


if __name__ == '__main__':
	main()
	#testStrat()
	#test()