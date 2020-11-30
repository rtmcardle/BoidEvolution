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


	def boidFitness(self, species=[1.0, 1.5, 1.35, 200, 75, 2.5], seed=0):
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

def main():
	evolution = BoidEvolution()
	evolution.loadClassifiers()
	fit = evolution.listFitness()
	print(f"Sample Species Fitnesses: {fit}")


if __name__ == '__main__':
	main()