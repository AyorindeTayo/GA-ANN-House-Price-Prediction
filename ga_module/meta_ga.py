import numpy as np
from sklearn.metrics import mean_squared_error
from .ga import GA
from .meta_runGA import run

# defining the GA class
class meta_GA:

    # initialization function for class, definition of attributes below:
    # features:     a list of strings, initial features (usualy all features minus the target)
    # init_ratio:   float between 0 and 1, the inital probability of activating a feature in 
    #               the initial population. 1 would activate all features, 0.5 half, etc...
    # cross_rate:   float between 0 and 1, the probability that a parent will cross genes
    #               with another individual in the population
    # mutate_rate:  float between 0 and 1, the probability that a gene will be mutated in
    #               a child. 1 would mutate all genes, 0.5 would mutate half, etc...
    # pop_size:     integer, decides the population size in each generation. number of
    #               individuals per generation.
    #
    def __init__(self, ranges, init_ratio, cross_rate, mutate_rate, pop_size, elitism):
        self.ranges = ranges
        self.chromosome_length = len(ranges)
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.elite = int(pop_size * elitism)

        # initialization of first generation
        initpop = []
        for _ in range(self.pop_size):
            temp = []
            for i in range(self.chromosome_length):
                temp.append(np.random.choice(a=range(len(ranges[i])), size=1)[0])
            initpop.append(np.array(temp))

        self.pop = np.array(initpop)

    # function for converting the chromosome binary data into a list of features
    # population:   np vstack consisting of np arrays of individuals
    #
    def readChromosomes(self):
        result = []
        for chromosome in self.pop:
            result.append([self.ranges[i][c] for i, c in enumerate(chromosome)])  
        return result

    # general fitness function that calculates a fitness score based on what type of fitness is selected
    # population:   np vstack of np arrays representing the population to score
    # regressor:    regressor to be used for predictions
    # X_train:      pandas dataframe containing the training data, normalized
    # X_test:       pandas dataframe containing the test data, normalized
    # y_train:      np array containing the training data target values
    # y_test:       np array containing the test data target values
    # f_type:       string defining what algorithm to use for fitness score, but 0 always bad, 1 always good!
    #
    def get_fitness(self, features, train_X_train, train_X_test, train_y_train, train_y_test, X_train, X_test, y_train, y_test):

        result = []
        predictions = []
        feats = []
        for individual in self.pop:
            evolution, bestFeatures, bestPredictions, initial_mape, final_mape, mape_y_test,final_mape_prediction,initial_mape_prediction = run(features, train_X_train, train_X_test, train_y_train, train_y_test, X_train, X_test, y_train, y_test, self.ranges[0][individual[0]], self.ranges[1][individual[1]], self.ranges[2][individual[2]], self.ranges[3][individual[3]], self.ranges[4][individual[4]], self.ranges[5][individual[5]], 
            ga_ann_iterations=self.ranges[6][individual[6]], ga_ann_layers=self.ranges[7][individual[7]], mape_ann_iterations=1000, mape_ann_layers=4,
            ga_score='score', ga_evolve='elitism',
            final_mape_idx='best')

            bestIdx = np.argmin(evolution)
            result.append(final_mape)
            predictions.append(final_mape_prediction)
            feats.append(bestFeatures[bestIdx])

        fitness = np.array(result)
        fitness = (np.max(result) - result)**0.5
        
        return result, fitness, predictions, feats

    # selection of population for the mating pool, initial next population, before cross and mutation
    # based on type of selection
    # fitness:  np array of fitness scores, greater number better, used as probability to be picked in
    #           the selection
    # stype:    string defining what type of selection algorithm to use, default: default
    #
    def select(self, fitness, stype='default'):
        # append new selection algorithms to if else statements below and define a stype parameter for it
        if (stype == 'default'):
            idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        elif (stype == 'elitism'):
            idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size-self.elite, replace=True, p=fitness/fitness.sum())
        elif (stype == 'new'):
            pass
        return self.pop[idx]

    # crossover function doing a crossover of individual in mating pool conditioned on the crossover rate and
    # depending on what type of crossover type selected
    # parent:   np array of binary chromosome representing the parent to be crossed
    # pop:      np vstack of np arrays of chromosomes representing the population to pick a mate from
    # ctype:    string defining what crossover algorithm to use (default : Multi point crossover, uniform: uniform crossover)
    #
    def crossover(self, parent, pop, ctype='default'):
        # append new crossover algorithms to if else statements below and define a ctype parameter for it
        if (ctype == 'default'):
            if np.random.rand() < self.cross_rate:
                mate = np.random.randint(0, len(pop), size=1)
                cross_points = np.sort(np.random.randint(0, self.chromosome_length, size=2))
                parent[:] = np.concatenate((parent[:cross_points[0]], pop[mate][0][cross_points[0]:cross_points[1]], parent[cross_points[1]:]))
        elif (ctype == 'uniform'):
            if np.random.rand() < self.cross_rate:
                mate = np.random.randint(0, self.pop_size, size=1)
                coinFlip = np.random.choice(a=[True, False], size=(self.chromosome_length))
                invertFlip = np.invert(coinFlip)
                parent[coinFlip] = parent[coinFlip]
                parent[invertFlip] = pop[mate][0][invertFlip]
        elif (ctype == 'new'):
            pass
        return parent

    # crossover 2!
    #
    def crossover2(self, parent1, parent2, ctype='default'):
        # append new crossover algorithms to if else statements below and define a ctype parameter for it
        child1 = parent1 
        child2 = parent2
        if (ctype == 'default'):
            if np.random.rand() < self.cross_rate:
                #mate = np.random.randint(0, self.pop_size, size=1)
                cross_points = np.sort(np.random.randint(0, self.chromosome_length, size=2))
                child1 = np.concatenate((parent1[:cross_points[0]], parent2[cross_points[0]:cross_points[1]], parent1[cross_points[1]:]))
                child2 = np.concatenate((parent2[:cross_points[0]], parent1[cross_points[0]:cross_points[1]], parent2[cross_points[1]:]))
        elif (ctype == 'uniform'):
            if np.random.rand() < self.cross_rate:
                coinFlip = np.random.choice(a=[True, False], size=(self.chromosome_length))
                invertFlip = np.invert(coinFlip)
                child1[coinFlip] = parent1[coinFlip]
                child1[invertFlip] = parent2[invertFlip]
                child2[coinFlip] = parent2[coinFlip]
                child2[invertFlip] = parent1[invertFlip] 
        elif (ctype == 'new'):
            pass
        
        return child1,child2
    
    # mutation function doing mutations conditioned on the mutation rate and mutation type selected
    # child:    np array representing chromosome of the child candidate for mutation
    # mtype:    string defining what mutation algorithm to use
    #
    def mutate(self, child, mtype='default'):
        # append new mutation algorithms to if else statements below and define a mtype parameter for it
        if(mtype == 'default'):
            mutations = np.random.choice(a=[1, 0], size=(self.chromosome_length), p=[self.mutate_rate, 1-self.mutate_rate])
            for i in range(mutations.shape[-1]):
                if (mutations[i]):
                    child[i] = np.random.choice(a=range(len(self.ranges[i])), size=1)[0]
        elif (mtype == 'new'):
            pass
        return child

    # evolution function creating the next generation/population based on fitness
    # fitness:  np array of fitness scores for the current population
    #
    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

    # evolve 2!
    #
    def evolve2(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        
        if (self.pop_size % 2 == 0):
            n = range(0, self.pop_size, 2)
        else:
            n = range(0,self.pop_size-1,2)
            pop_copy[-1,:] = pop[-1,:]
        for i in n:
            pop_copy[i], pop_copy[i+1]= self.crossover2(pop[i],pop[i+1])
            pop_copy[i] = self.mutate(pop_copy[i])
            pop_copy[i+1] = self.mutate(pop_copy[i])         
        self.pop = pop_copy

    # evolution function creating the next generation/population based on fitness and with elitism
    # fitness:  np array of fitness scores for the current population
    #
    def evolve_elitism(self, fitness):
        elite_idx = np.argsort(fitness)[::-1]
        elite = self.pop[elite_idx[:self.elite]]
        pop = self.select(fitness, stype='elitism')
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy, ctype='default')
            child = self.mutate(child)
            parent[:] = child
        self.pop = np.vstack(list(pop) + list(elite))


    
