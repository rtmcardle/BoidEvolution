# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
# Edited for the purpose of evolutionary optimization.
# Copyright (2020) Ryan T. McArdle - BSD license
#
# -----------------------------------------------------------------------------

import math
import random
import datetime
import threading
import multiprocessing
import numpy as np
from vec2 import vec2
from joblib import Parallel, delayed

from grispy import GriSPy


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
        #random.seed(2)
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(2,self.max_velocity)
        self.velocity = vec2(speed*math.cos(angle), speed*math.sin(angle))
        self.r = 2.0 ## Pretty sure this is useless?
       

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

        #self.position.x = self.position.x % self.width
        #self.position.y = self.position.y % self.height


    # Separation
    # Method checks for nearby boids and steers away
    def separate(self, boids):
        desired_separation = self.sepRadius
        steer = vec2(0, 0)
        count = 0

        #######################################################
        ## Original
        ## For every boid in the system, check if it's too close
        #for other in boids:
        #    d = (self.position - other.position).length()
        #    # If the distance is greater than 0 and less than an arbitrary
        #    # amount (0 when you are yourself)
        #    if 0 < d < desired_separation:
        #        # Calculate vector pointing away from neighbor
        #        diff = self.position - other.position
        #        diff = diff.normalized()
        #        steer += diff/d  # Weight by distance
        #        count += 1       # Keep track of how many
        #######################################################
        ## RTM Edit
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
        #count = len(boids)
        #if count > 0:
        for other in boids:
            d = (self.position - other.position).wrap(self.bounds).length()
            if 0 < d < neighbor_dist:
            #################################
            #sum += other.position  # Add position
            #count += 1
            #################################
                target = self.seek(other)
                sum += target
        if count > 0:
            return sum/count
        else:
            return vec2(0, 0)

    def flock(self, boids):
        sep = self.separate(boids)  # Separation
        ali = self.align(boids)  # Alignment
        coh = self.cohesion(boids)  # Cohesion

        # Arbitrarily weight these forces
        sep *= self.sepWeight
        ali *= self.alignWeight
        coh *= self.cohWeight

        # Add the force vectors to acceleration
        self.acceleration += sep
        self.acceleration += ali
        self.acceleration += coh
        return sep+ali+coh

    def update(self,accel=None):
        #print("Boids Update")
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
        self.boids = []
        self.start = datetime.datetime.now()
        self.frames = 0
        self.stop = False
        #random.seed(3)
        #self.num_cores = multiprocessing.cpu_count()
        #self.processes = 8
        #print(f'Cores: {num_cores}')
        #self.lock = threading.Lock()

        ## Generates swarm of boids
        for i in range(count):
            boid = Boid(random.uniform(-1,1)*width, random.uniform(-1,1)*height, width, height, alignWeight, sepWeight, cohWeight, alignCohRadius, sepRadius, maxAccel)
            self.boids.append(boid)

        self.n = self.count//num_processes
        
        

    def run(self,positions,num_processes,parallel=None,):
        gsp = GriSPy(positions, N_cells = 5, periodic={0:(0,self.width), 1:(0,self.height)})
        dub = max(self.alignCohRadius, self.sepRadius)
        _, self.neighbor_indices = gsp.bubble_neighbors(positions,distance_upper_bound=dub)

        
        ### Splits flock for multiprocessing
        self.flocks = [self.boids[i:i+self.n] for i in range(0,self.count,self.n)]
        self.n_i = [self.neighbor_indices[i:i+self.n] for i in range(0,self.count,self.n)]
        ## Parallelize this

        #print("PARALLEL ELIF")
        if parallel==None:
            for i in range(len(self.boids)):
                this_boid = self.boids[i]
                neighbors = [self.boids[j] for j in self.neighbor_indices[i]]
                this_boid.run(neighbors)
        elif type(parallel) == type(Parallel()):
            #print("PARALLEL")
            self.accels = parallel(delayed(self.parallel_flock)(i) for i in range(len(self.boids)))
            self.boids = parallel(delayed(self.parallel_update)(i) for i in range(len(self.boids)))
        #elif type(parallel) == type(multiprocessing.Pool()):
        #    pass
        #    print("POOL")
        #    accel_results = parallel.map_async(self.pool_flock, range(len(self.flocks)))
        #    self.accels = accel_results.get()
        #    boid_result = parallel.map_async(self.pool_update, range(len(self.flocks)))
        #    self.boids = [boid for sublist in boid_result.get() for boid in sublist]
        #    print("FINISHED")

        

        #### Splits flock for multiprocessing
        #n = self.count//self.processes
        #self.flocks = [self.boids[i:i+n] for i in range(0,self.count,n)]
        #self.n_i = [self.neighbor_indices[i:i+n] for i in range(0,self.count,n)]
        #self.accels=[]
        #self.ret_boids=[]

        #inputs = self.boids
        #self.accels = Parallel(n_jobs=self.processes, prefer='threads')(delayed(self.pool_flock)(i) for i in range(len(self.flocks)))
        #self.ret_boids = Parallel(n_jobs=self.processes, prefer='threads')(delayed(self.pool_update)(i) for i in range(len(self.flocks)))
        
        ##############################################
        
        #self.accels = parallel(delayed(self.parallel_flock)(i) for i in range(len(self.boids)))
        #self.boids = parallel(delayed(self.parallel_update)(i) for i in range(len(self.boids)))
        
        ###############################################

        #self.accels = parallel(delayed(self.pool_flock)(i) for i in range(len(self.flocks)))
        #self.ret_boids = parallel(delayed(self.pool_update)(i) for i in range(len(self.flocks)))
        #self.boids = [boid for sublist in self.ret_boids for boid in sublist]

        ################################################

        #accel_results = parallel.map_async(self.pool_flock, range(len(self.flocks)))
        #self.accels = accel_results.get()
        #boid_result = parallel.map_async(self.pool_update, range(len(self.flocks)))
        #self.boids = [boid for sublist in boid_result.get() for boid in sublist]

        ##################################################

        #if __name__ == '__main__':
        #    with multiprocessing.Pool(4) as pool:
        #        _ = pool.map(self.parallel_flock, range(len(self.boids)))
        #    with multiprocessing.Pool(4) as pool:
        #        _ = pool.map(self.parallel_update, range(len(self.boids)))

        

        #with multiprocessing.Pool(self.processes) as pool:
        #    self.accels = pool.map(self.pool_flock, range(len(self.flocks)))
        #    self.boids = [boid for sublist in pool.map(self.pool_update, range(len(self.flocks))) for boid in sublist]

        #print(f"ParallelBoids: {self.parallel_boids}")
        #self.boids = [*self.parallel_boids]

        

        #processes = []
        #tasks_to_accomplish = multiprocessing.Queue()
        #tasks_completed = multiprocessing.Queue()
        #for i in range(n):
        #    tasks_to_accomplish.put("task" + str(i))

        #for w in range(self.processes):
        #    p = multiprocessing.Process(target=self.pool_flock, args=(i,))
        #    processes.append(p)
        #    p.start()
        #for job in processes:
        #    job.join()

        #processes = []
        #for i in range(self.count):
        #    p = multiprocessing.Process(target=self.pool_update, args=(i,))
        #    processes.append(p)
        #    p.start()
        #for job in processes:
        #    job.join()




        self.frames+=1

        if self.frames==100:
            self.end = datetime.datetime.now()
            self.elapsed = self.end-self.start
            print("Time: "+str(self.elapsed))
            print("FPS: "+str(self.frames/self.elapsed.total_seconds()))
            self.stop = True

    def parallel_flock(self,i):
        this_boid = self.boids[i]
        neighbors = [self.boids[j] for j in self.neighbor_indices[i]]
        return this_boid.flock(neighbors)

    def parallel_update(self,i):
        this_boid = self.boids[i]
        
        #if i == 0:
        #    print(f"P[0]: {this_boid.position}")
        return this_boid.update(self.accels[i])

    def pool_flock(self,i):
        ret_accels = []
        this_flock = self.flocks[i]
        #for j in range(len(this_flock)):
        #    this_boid = this_flock[j]
        #    neighbors = [self.boids[k] for k in self.n_i[i][j]]
        #    ret_accels.append(this_boid.flock(neighbors))
        ret_accels = [this_flock[j].flock([self.boids[k] for k in self.n_i[i][j]]) for j in range(len(this_flock))]
        #this_boid = self.boids[i]
        #neighbors = [self.boids[j] for j in self.neighbor_indices[i]]
        #self.accels.append(ret_accels)
        return ret_accels

    def pool_update(self,i):
        #ret_boids = []
        this_flock = self.flocks[i]
        #for j in range(len(this_flock)):
        #    this_boid = this_flock[j]
        #    this_boid.update(self.accels[i][j])
        #    ret_boids.append(this_boid)
        ret_boids = [this_flock[j].update(self.accels[i][j]) for j in range(len(this_flock))]
        #if i == 0:
        #    print(f"P[0]: {this_boid.position}")
        #self.ret_boids.append(ret_boids)
        return ret_boids




def main():        
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ##random.seed(1)

    count=150
    screen_width = 3000
    screen_height = screen_width
    sample_species = [1.0, 1.5, 1.35, 200, 75, 2.5]

    num_cores = multiprocessing.cpu_count()
    num_processes = num_cores//2

    flock = Flock(num_processes, count, screen_width, screen_height, *sample_species)
    P = np.ndarray((count,2), buffer=np.array([(boid.position.x,boid.position.y) for boid in flock.boids]))
    V = np.ndarray((count,2), buffer=np.array([(boid.velocity.x,boid.velocity.y) for boid in flock.boids]))

    def update(*args):
        #print(f'Update')
        flock.run(P,num_processes,parallel,)
        for i,boid in enumerate(flock.boids):
            #print(boid)
            P[i] = boid.position
            V[i] = boid.velocity
        #scatter.set_offsets(P)
        #print(f"P[0]: {P[0]}")
        arrows.set_offsets(P)
        arrows.set_UVC(V[:,0], V[:,1])


    ## Standard Run
    while not flock.stop:
        flock.run(P,num_processes,)
        for i,boid in enumerate(flock.boids):
            #print(boid)
                P[i] = boid.position

    ## Parallel Run
    flock = Flock(num_processes, count, screen_width, screen_height, *sample_species)
    P = np.ndarray((count,2), buffer=np.array([(boid.position.x,boid.position.y) for boid in flock.boids]))
    V = np.ndarray((count,2), buffer=np.array([(boid.velocity.x,boid.velocity.y) for boid in flock.boids]))
    
    with Parallel(n_jobs = num_processes, prefer='threads', verbose=0) as parallel:
        #print("STARTING")
        while not flock.stop:
            flock.run(P,num_processes,parallel,)
            for i,boid in enumerate(flock.boids):
            #print(boid)
                P[i] = boid.position

    ### Pool Run
    #flock = Flock(num_processes, count, screen_width, screen_height, *sample_species)
    #P = np.ndarray((count,2), buffer=np.array([(boid.position.x,boid.position.y) for boid in flock.boids]))
    #V = np.ndarray((count,2), buffer=np.array([(boid.velocity.x,boid.velocity.y) for boid in flock.boids]))
    #with multiprocessing.Pool(num_processes) as parallel:
    #    while not flock.stop:
    #        flock.run(P,num_processes,parallel)
    #        for i,boid in enumerate(flock.boids):
    #        #print(boid)
    #            P[i] = boid.position


    ### Visualization
    #fig = plt.figure(figsize=(12.0,10.0))
    #ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=True)
    ##scatter = ax.scatter(P[:,0], P[:,1],
    #                     #s=30, facecolor="red", edgecolor="None", alpha=0.5)
    #arrows = ax.quiver(P[:,0], P[:,1],V[:,0], V[:,1], scale = 2000, headaxislength=4.5, pivot = 'middle')
    #animation = FuncAnimation(fig, update, fargs=[parallel,], interval=1)
    #ax.set_xlim(0,screen_width)
    #ax.set_ylim(0,screen_height)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #plt.show()


if __name__ == '__main__':
    main()