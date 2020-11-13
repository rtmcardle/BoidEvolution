# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
# Edited for the purpose of evolutionary optimization.
# Copyright (2020) Ryan T. McArdle - BSD license
#
# -----------------------------------------------------------------------------

from joblib import Parallel, delayed
from grispy import GriSPy
from vec2 import vec2
import multiprocessing
import pandas as pd
import numpy as np
import datetime
import random
import math

class Boid:
    def __init__(self, x, y, width, height, alignWeight, sepWeight, cohWeight, alignCohRadius, sepRadius, maxAccel):
        
        ## Frame variables
        self.position = vec2(x, y)
        self.width = width
        self.height = height
        self.bounds = (self.width,self.height)

        ## Species variables
        self.alignWeight = alignWeight
        self.sepWeight = sepWeight
        self.cohWeight = cohWeight
        self.alignCohRadius = alignCohRadius
        self.sepRadius = sepRadius
        self.maxAccel = maxAccel

        ## Initialization variables
        self.acceleration = vec2(0, 0)
        self.max_velocity = 20
        self.max_acceleration = self.maxAccel
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(2,self.max_velocity)
        self.velocity = vec2(speed*math.cos(angle), speed*math.sin(angle))
       

    def seek(self, target):
        desired = (target.position - self.position).wrap(self.bounds)
        desired = desired.normalized()
        desired *= self.max_velocity
        steer = desired - self.velocity
        steer = steer.limited(self.max_acceleration)
        return steer


    # Wraparound
    def borders(self):
        x, y = self.position
        x = x % self.width
        y = y % self.height
        self.position = vec2(x,y)


    # Separation
    # Method checks for nearby boids and steers away
    def separate(self, boids):
        desired_separation = self.sepRadius
        steer = vec2(0, 0)
        count = 0

        # For every boid in the system, check if it's too close
        for other in boids:
            d = (self.position - other.position).wrap(self.bounds).length()
            # If the distance is greater than 0 and less than an arbitrary
            # amount (0 when you are yourself)
            if 0 < d < desired_separation:
            # Calculate vector pointing away from neighbor
            #if 0 < d:
                diff = (self.position - other.position).wrap(self.bounds)
                diff = diff.normalized()
                steer += diff/d  # Weight by distance
                count += 1       # Keep track of how many

        self.nS = count
        # Average - divide by how many
        if count > 0:
            steer /= count

        # As long as the vector is greater than 0
        if steer.length() > 0:
            # Implement Reynolds: Steering = Desired - Velocity
            steer = steer.normalized()
            steer *= self.max_velocity
            steer -= self.velocity
            steer = steer.limited(self.max_acceleration)

        return steer


    # Alignment
    # For every nearby boid in the system, calculate the average velocity
    def align(self, boids):
        neighbor_dist = self.alignCohRadius
        sum = vec2(0, 0)
        count = 0
        for other in boids:
            d = (self.position - other.position).wrap(self.bounds).length()
            if 0 < d < neighbor_dist:
                #align = other.velocity/count
                #sum += align
                sum += other.velocity
                count += 1
        self.nAC = count
        if count > 0:
            sum /= count
            # Implement Reynolds: Steering = Desired - Velocity
            sum = sum.normalized()
            sum *= self.max_velocity
            steer = sum - self.velocity
            steer = steer.limited(self.max_acceleration)
            return steer
        else:
            return vec2(0, 0)


    # Cohesion
    # For the average position (i.e. center) of all nearby boids, calculate
    # steering vector towards that position
    def cohesion(self, boids):
        neighbor_dist = self.alignCohRadius
        sum = vec2(0, 0)  # Start with empty vector to accumulate all positions
        count = 0
        for other in boids:
            d = (self.position - other.position).wrap(self.bounds).length()
            if 0 < d < neighbor_dist:
                target = self.seek(other)
                sum += target
                count += 1
        if count > 0:
            return sum/count
        else:
            return vec2(0, 0)


    def flock(self, boids):
        self.sep = self.separate(boids)  # Separation
        self.ali = self.align(boids)  # Alignment
        self.coh = self.cohesion(boids)  # Cohesion

        # Arbitrarily weight these forces
        self.sep *= self.sepWeight
        self.ali *= self.alignWeight
        self.coh *= self.cohWeight

        # Add the force vectors to acceleration
        self.acceleration += self.sep
        self.acceleration += self.ali
        self.acceleration += self.coh
        return self.sep+self.ali+self.coh


    def update(self,accel=None):
        # Update velocity
        if accel is not None:
            self.acceleration = accel
        self.velocity += self.acceleration
        # Limit speed
        self.velocity = self.velocity.limited(self.max_velocity)
        self.position += self.velocity
        if not (0 <= self.position.x <= self.width) or not (0 <= self.position.y <= self.height):
            self.borders()
        # Reset acceleration to 0 each cycle
        self.acceleration = vec2(0, 0)

        return self


    def run(self, boids):
        self.flock(boids)
        self.update()
        return (self.position.x,
                self.position.y,
                self.velocity.x,
                self.velocity.y,
                self.ali.x,
                self.ali.y,
                self.sep.x,
                self.sep.y,
                self.coh.x,
                self.coh.y,
                self.nAC,
                self.nS,
                )



class Flock:
    def __init__(self, num_processes, count, width, height, alignWeight, sepWeight, cohWeight, alignCohRadius, sepRadius, maxAccel):
        ## Uniform variables
        self.count = count
        self.width = width
        self.height = height

        ## Species variables
        self.alignWeight = alignWeight
        self.sepWeight = sepWeight
        self.cohWeight = cohWeight
        self.alignCohRadius = alignCohRadius
        self.sepRadius = sepRadius
        self.maxAccel = maxAccel

        ## Initializes simulation
        random.seed(random.randint(0,1e10))
        self.boids = []

        ## Generates swarm of boids
        self.boids = [Boid(random.uniform(-1,1)*width, 
                           random.uniform(-1,1)*height, 
                           width, 
                           height, 
                           alignWeight, 
                           sepWeight, 
                           cohWeight, 
                           alignCohRadius, 
                           sepRadius, 
                           maxAccel,) 
                      for _ in range(count)
                      ]

        self.P = np.ndarray((count,2), buffer=np.array([(boid.position.x,boid.position.y) for boid in self.boids]))
        self.V = np.ndarray((count,2), buffer=np.array([(boid.velocity.x,boid.velocity.y) for boid in self.boids]))
        #self.n = self.count//num_processes


    def animate(self,num_processes=1,parallel=None):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def update(*args):
            self.run(self.P,num_processes,parallel,)
            for i,boid in enumerate(self.boids):
                self.P[i] = boid.position
                self.V[i] = boid.velocity
            agg_lists = [lst for lst in self.lists]
            self.instances.append([func(lst) for lst in self.lists for func in self.agg_funcs])
            arrows.set_offsets(self.P)
            arrows.set_UVC(self.V[:,0], self.V[:,1])

        self.start = datetime.datetime.now()
        self.frames = 0
        self.stop = False
        ############################################
        ## Prepares for data collection
        #self.agg_funcs = ['min','max','mean','std',np.median]
        self.agg_funcs = [np.min,np.max,np.mean,np.std,np.median]
        self.agg_names = ['min','max','mean','std','median']
        self.agg_labels = ['Min','Max','Mean','Std','Median']
        self.base_names = ['xPos','yPos','xVel','yVel','xA','yA','xS','yS','xC','yC','nAC','nS']

        self.dict_names = [base+column for base in self.base_names for column in self.agg_labels]
        #dict_names.extend(['aligned','flocking','grouped'])

        self.lists = [[] for _ in self.base_names]

        self.instances = []
        #######################################

        fig = plt.figure(figsize=(12.0,10.0))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=True)
        arrows = ax.quiver(self.P[:,0], self.P[:,1],self.V[:,0], self.V[:,1], scale = 2000, headaxislength=4.5, pivot = 'middle')
        self.animation = FuncAnimation(fig, update, fargs=[parallel,], interval=1)
        ax.set_xlim(0,self.width)
        ax.set_ylim(0,self.height)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


    def simulate(self,num_processes=1,par=False):
        ## Initializes simulation
        self.start = datetime.datetime.now()
        self.frames = 0
        self.stop = False

        ## Prepares for data collection
        #self.agg_funcs = ['min','max','mean','std',np.median]
        self.agg_funcs = [np.min,np.max,np.mean,np.std,np.median]
        self.agg_names = ['min','max','mean','std','median']
        self.agg_labels = ['Min','Max','Mean','Std','Median']
        self.base_names = ['xPos','yPos','xVel','yVel','xA','yA','xS','yS','xC','yC','nAC','nS']

        self.dict_names = [base+column for base in self.base_names for column in self.agg_labels]
        #dict_names.extend(['aligned','flocking','grouped'])

        self.lists = [[] for _ in self.base_names]

        self.instances = []

        ## Main simulation loops
        if par:
            with Parallel(n_jobs = num_processes, prefer='threads', verbose=0) as parallel:
                while not self.stop:
                    self.run(self.P,num_processes,parallel,)
                    for i,boid in enumerate(self.boids):
                        self.P[i] = boid.position
        else:
            while not self.stop:
                self.run(self.P,num_processes,par,)
                for i,boid in enumerate(self.boids):
                    self.P[i] = boid.position

                    #self.Px[i] = boid.position.x
                    #self.Py[i] = boid.position.y
                    #self.Vx[i] = boid.velocity.x
                    #self.Vy[i] = boid.velocity.y
                    #self.A[i] = boid.ali
                    #self.S[i] = boid.sep
                    #self.C[i] = boid.coh
                    #self.nAC[i] = boid.nAC
                    #self.nS[i] = boid.nS

                ## Run the aggregate functions and save to a list
                agg_lists = [lst for lst in self.lists]
                self.instances.append([func(lst) for func in self.agg_funcs for lst in self.lists])
        
        
    def run(self,positions,num_processes,parallel=False,):

        ### Builds gridsearch and finds nearby neighbors
        gsp = GriSPy(positions, N_cells = 10, periodic={0:(0,self.width), 1:(0,self.height)})
        dub = max(self.alignCohRadius, self.sepRadius)
        _, self.neighbor_indices = gsp.bubble_neighbors(positions,distance_upper_bound=dub)

        ### Breaks off code for single vs. multiprocessing
        if not parallel:
            for i in range(len(self.boids)):
                this_boid = self.boids[i]
                neighbors = [self.boids[j] for j in self.neighbor_indices[i]]
                ret_atts = this_boid.run(neighbors)
                for i,att in enumerate(ret_atts):
                    self.lists[i].append(att)

        else:
            #print("PARALLEL")
            self.accels = parallel(delayed(self.parallel_flock)(i) for i in range(len(self.boids)))
            self.boids = parallel(delayed(self.parallel_update)(i) for i in range(len(self.boids)))

        #######################################################
        #######################################################
        ###
        ### A Pooling method is written, but often destabilizes
        ### system. Run as is at your own risk, and only if 
        ### attempting to patch multiprocessing.
        ###
        #######################################################
        #######################################################
        #elif type(parallel) == type(multiprocessing.Pool()):
        #    pass
            #print("POOL")
            #self.flocks = [self.boids[i:i+self.n] for i in range(0,self.count,self.n)]
            #self.n_i = [self.neighbor_indices[i:i+self.n] for i in range(0,self.count,self.n)]
            ## Parallelize this
            ##accel_results = parallel.map(self.pool_flock, range(len(self.flocks)))
            #self.accels = [accel for sublist in accel_results for accel in sublist]
            #boid_result = parallel.map(self.pool_update, range(len(self.flocks)))
            #self.boids = [boid for sublist in boid_result for boid in sublist]
            #print("FINISHED")
        #######################################################


        ### Manages the number of instances calculated and 
        ### halting criteria.
        self.frames+=1
        print(self.frames)
        if self.frames==100:
            ## Save the record
            simulation_record = pd.DataFrame(self.instances, columns=self.dict_names)
            print(simulation_record)
            simulation_record.to_csv('./simulation_record.csv')

            ## Report the frame/instance rate
            self.end = datetime.datetime.now()
            self.elapsed = self.end-self.start
            print("Time: "+str(self.elapsed))
            print("FPS: "+str(self.frames/self.elapsed.total_seconds()))

            ## Stop the animation
            try:
                self.animation.event_source.stop()
            except:
                pass
            self.stop = True


    def parallel_flock(self,i):
        this_boid = self.boids[i]
        neighbors = [self.boids[j] for j in self.neighbor_indices[i]]
        return this_boid.flock(neighbors)


    def parallel_update(self,i):
        this_boid = self.boids[i]
        return this_boid.update(self.accels[i])


    ###########################################################
    ###
    ### The following two methods were intended for the pooling
    ### multiprocessing method. Unnecessary for the single 
    ### processing implementation.
    ###
    ###########################################################
    #def pool_flock(self,i):
    #    ret_accels = []
    #    this_flock = self.flocks[i]
    #    ret_accels = [this_flock[j].flock([self.boids[k] for k in self.n_i[i][j]]) for j in range(len(this_flock))]
    #    return ret_accels

    #def pool_update(self,i):
    #    this_flock = self.flocks[i]
    #    ret_boids = [this_flock[j].update(self.accels[len(this_flock)*i+j]) for j in range(len(this_flock))]
    #    return ret_boids
    ###########################################################



def main():        

    count=150
    screen_width = 3000
    screen_height = screen_width
    sample_species = [1.0, 1.5, 1.35, 200, 75, 2.5]

    num_cores = multiprocessing.cpu_count()
    num_processes = num_cores//2

    flock = Flock(num_processes, count, screen_width, screen_height, *sample_species)
    
    ### Run a visualization of the boids
    flock.animate()

    ### Simulate and produce data
    #flock.simulate()

    ### Simulate and procude data w/ multiprocessing
    #flock.simulate(num_processes=num_processes, par=True)


    
if __name__ == '__main__':
    main()