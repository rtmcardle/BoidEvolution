# BoidEvolution
This project was originally created as a group term project for CSCI4560/6560: Evolutionary Computation, under the direction of Dr. K. Rasheed at the University of Georgia. 

The goal is to explore various evolutionary algorithms and their performance when tasked with optimizing the parameters of a boid simulation in order to best exhibit flocking behavior. 

The fitness function for these evolutions is built from a group of random forest classifiers that have been trained on the [Swarm Behavior Data Set][1], uploaded by Abpeikar et al. to the UCI Machine Learning Repository. This data set is based on the results of a [survey][2] exploring human-perceptions of flocking behavior hosted by the same researchers.

## The Simulation
The 2-d boid simulation can be found in the file [BoidSim.py](BoidSim.py) and is developed using [the orignial boid framework][4] presented by Craig Reynolds. The [original file][3] is provided by Nicolas Rougier under a BSD license and has been edited by Ryan McArdle for the purposes of speeding up the simulation and accepting as arguments the evolved parameters for a given boid 'species'. 

These evolved parameters include:
* Alignment Weight - The weight of the alignment force in determining acceleration.
* Separation Weight - The same for separation.
* Cohesion Weight - The same for cohesion.
* Alignment/Cohesion Radius - The maximum difference between two boids to induce alignment or cohesion forces on either.
* Separation Radius - The same for separation.
* Maximum Acceleration - The maximum acceleration for an individual boid. 

When an instance of the Flock class is initialized, it randomly generates 200 boids with initial conditions and begins to simulate their flight based on the provided parameters. After 50 timesteps, the simulation begins to record data about the flight of each boid. After another 100 timesteps, the data for each timestep is statistically aggregated in preparation for classification by the [classifiers](TreeModels/). These classifiers have been developed following the work done by [Liu et al.][6] which sought to train classifiers to perform well on the Swarm Behavior Data Set.

## The Fitness Function
After a given 'species' of boid has been simulated, the fitness for that set of parameters must be determined. 

This is done by passing the statistical data about the flight of the boids to the classifiers for each of the 3 behaviors provided by the original data set: Aligned, Flocking, and Grouped. Each classifier returns a list of 100 binary values, indicating whether the given behavior was expressed in each of the 100 recorded timesteps. The percentage of timesteps which return a positive value is used as a score for that behavior. These scores can then be combined to determine an overall fitness for the parameters.

Initial exploration found that the Aligned behavior is very difficult to satisfy while the Flocking and Grouped behaviors are essentially trivial. As such, the fitness function is implemented in 3 equally spaced phases in hopes of promoting the discover of an individual which express all behaviors consistently. The phases are as follows:

1. Initially, only the Aligned behavior is selected for. The score on the Aligned behavior is the score of the individual.
2. Next, 2/3 of the individual's weight is determined by Aligned, with the other 1/3 split between Flocking and Grouped. There is also a weighting which punishes the fitness of parameters which score differently on each of the 3 groups (promoting optimization of all behaviors equally).
3. Finally, each behavior is weighted equally. The weighting which punishes different fitnesses for each parameter remains.

## The Evolutions
We explore 4 different evolutionary algorithms in [BoidEvolution.py](BoidEvolution.py). 
* Genetic Algorithms
  1. Generational GA
  2. Steady State GA
  
* Evolutionary Strategies
  1. (mu,lambda)
  2. (mu+lambda)

For the evolutionary strategies, we explore multiple lambda:mu ratios in the range [3,5,7,9]. We run each algorithm for 10,000 fitness evaluations using the same parameters for crossover probability and mutation probability, and the same methods where appropriate.

## Results
We ultimately find little success in optimizing boid parameters to express realistic flocking behavior. The results of each evolution can be found in [runs_record](runs_record/), where one will find both a log of the evolution and a plot of the maximum and average fitnesses against generation. 

A screecapture of flight behavior for a sample of our best individuals and hand-picked individuals can be seen in [screen_caps](screen_caps/). The subfolders describe either the method used to evolve the individual, or the parameters hand-picked to produce the behavior. We note that quickly handpicking behaviors is a much more cost-effective and reliable way to produce realistic flocking behavior.

We ultimately corroborate suspicions raised by Liu et al. that suggest that the classifiers would perform poorly on samples of boids outside of the original data set. A written paper and more through exploration of the simulation, classifiers, evolutions, and results can be found [here][5].



[1]: https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour
[2]: https://unsw-swarm-survey.netlify.app/
[3]: https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/boid_python.py
[4]: https://www.researchgate.net/profile/Craig_Reynolds2/publication/2797343_Flocks_Herds_and_Schools_A_Distributed_Behavioral_Model/links/0fcfd5095a869204df000000.pdf
[5]: https://ryantmcardle.com/wp-content/uploads/2020/12/EvolvingBoidFlockingBehavior.pdf
[6]: https://ryantmcardle.com/wp-content/uploads/2020/12/ClassifyingBehaviorsBoidSwarms.pdf
