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
import pandas as pd
import numpy as np
import datetime
import random
import math
import os

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

        # Weights determined by species of boid
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
    def __init__(self, seed, count, width, height, alignWeight, sepWeight, cohWeight, alignCohRadius, sepRadius, maxAccel):
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
        random.seed(seed)
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

        ## Sets position and velocity arrays
        self.P = np.ndarray((count,2), buffer=np.array([(boid.position.x,boid.position.y) for boid in self.boids]))
        self.V = np.ndarray((count,2), buffer=np.array([(boid.velocity.x,boid.velocity.y) for boid in self.boids]))


    def animate(self,end=True,record=True,screen_cap=False,name='capture'):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        ## Main animation loop
        def update(*args):
            if end==True and self.stop:
                self.animation.event_source.stop()
            self.run(self.P)

            ## Record position and velocity for animation
            for i,boid in enumerate(self.boids):
                self.P[i] = boid.position
                self.V[i] = boid.velocity

            ## Record data for simulation
            if record==True and self.record == True:
                self.instances.append([func(lst) for lst in self.lists for func in self.agg_funcs])

            if self.capture:
                cur_dir = os.getcwd()
                prob_dir = os.path.join(cur_dir,'screen_caps\\')
                if not os.path.exists(prob_dir):
                    os.mkdir(prob_dir)
                time_dir = os.path.join(prob_dir,f'{name}\\')
                os.mkdir(time_dir)
                plt.savefig(os.path.join(time_dir,'plot.png'))

            ## Update animation information
            arrows.set_offsets(self.P)
            arrows.set_UVC(self.V[:,0], self.V[:,1])

        #self.start = datetime.datetime.now()
        self.frames = 0
        self.stop = False
        self.record = False
        self.capture = False
        self.screen_cap = screen_cap

        ## Prepares for data collection
        self.agg_funcs = [np.min,np.max,np.mean,np.std,np.median]
        self.agg_names = ['min','max','mean','std','median']
        self.agg_labels = ['Min','Max','Mean','Std','Median']
        self.base_names = ['xPos','yPos','xVel','yVel','xA','yA','xS','yS','xC','yC','nAC','nS']
        self.dict_names = [base+column for base in self.base_names for column in self.agg_labels]
        self.lists = [[] for _ in self.base_names]
        self.instances = []

        ## Animates simulation
        fig = plt.figure(figsize=(12.0,10.0))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=True)
        arrows = ax.quiver(self.P[:,0], self.P[:,1],self.V[:,0], self.V[:,1], scale = 2000, headaxislength=4.5, pivot = 'middle')
        self.animation = FuncAnimation(fig, update, interval=1)
        ax.set_xlim(0,self.width)
        ax.set_ylim(0,self.height)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


    def simulate(self):
        ## Initializes simulation
        self.start = datetime.datetime.now()
        self.frames = 0
        self.stop = False
        self.record = False
        self.screen_cap = False

        ## Prepares for data collection
        self.agg_funcs = [np.min,np.max,np.mean,np.std,np.median]
        self.agg_names = ['min','max','mean','std','median']
        self.agg_labels = ['Min','Max','Mean','Std','Median']
        self.base_names = ['xPos','yPos','xVel','yVel','xA','yA','xS','yS','xC','yC','nAC','nS']

        self.dict_names = [base+column for base in self.base_names for column in self.agg_labels]

        self.instances = []

        ## Main simulation loop
        while not self.stop:
            self.lists = [[] for _ in self.base_names]
            self.run(self.P)

            ## Update record of boid positions
            for i,boid in enumerate(self.boids):
                self.P[i] = boid.position

            ## Run the aggregate functions and save to a list
            if self.record == True:
                self.instances.append([func(lst) for lst in self.lists for func in self.agg_funcs])
        
        ## Save the record
        self.simulation_record = pd.DataFrame(self.instances, columns=self.dict_names)

        ######
        ## Optionally save the record of the simulation to file 
        #timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
        #print(simulation_record)
        #self.simulation_record.to_csv('./simulation_record.csv')
        ######

        ######
        ## Optionallay report the frame/instance rate
        #self.end = datetime.datetime.now()
        #self.elapsed = self.end-self.start
        #print("Time: "+str(self.elapsed))
        #print("FPS: "+str(self.frames/self.elapsed.total_seconds()))
        ######

        ## Return the record for classification of instances
        return self.simulation_record
        
        
    def run(self,positions,delay=50,length=150):

        ## Builds gridsearch and finds nearby neighbors
        gsp = GriSPy(positions, N_cells = 10, periodic={0:(0,self.width), 1:(0,self.height)})
        dub = max(self.alignCohRadius, self.sepRadius)
        _, self.neighbor_indices = gsp.bubble_neighbors(positions,distance_upper_bound=dub)

        ## Calculates and records data for each individual boid
        for i in range(len(self.boids)):
            this_boid = self.boids[i]
            neighbors = [self.boids[j] for j in self.neighbor_indices[i]]
            ret_atts = this_boid.run(neighbors)
            for i,att in enumerate(ret_atts):
                self.lists[i].append(att)

        ## Manages the number of instances calculated and 
        #  halting criteria.
        self.frames+=1
        if self.frames == delay:
            self.record = True
        if self.frames == delay+length:
            self.stop = True
        if self.screen_cap and self.frames==100:
            self.capture = True
        if self.screen_cap and self.frames==101:
            self.capture = False



def main():        

    count=150
    screen_width = 3000
    screen_height = screen_width
    sample_species = [1.0, 1.5, 1.35, 200, 75, 2.5]
    sample_species = [1.4615518242421464, 1.2793818826191314, 0.014844554893147854, 188.9352702321177, 1.0, 1.0]
    ## alignWeight, sepWeight, cohWeight, alignCohRadius, sepRadius, maxAccel

    seed = random.randint(1,1e10)

    flock = Flock(seed, count, screen_width, screen_height, *sample_species)
    
    ### Run a visualization of the boids
    flock.animate(record=False, end=True)

    ### Simulate and produce data
    #flock.simulate()



    
if __name__ == '__main__':
    main()