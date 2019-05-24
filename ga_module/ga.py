import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

# defining the GA class
class GA:

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
    def __init__(self, features, init_ratio, cross_rate, mutate_rate, pop_size, elitism):
        self.features = features
        self.chromosome_length = len(features)
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.elite = int(pop_size * elitism)

        # initialization of first generation
        self.pop = np.vstack([np.random.choice(a=[1, 0], size=(self.chromosome_length), p=[init_ratio, 1-init_ratio]) for _ in range(pop_size)])

    # function for converting the chromosome binary data into a list of features
    # population:   np vstack consisting of np arrays of individuals
    #
    def readChromosomes(self, population):
        result = []
        for chromosome in population:
            result.append([f for i, f in enumerate(self.features) if chromosome[i]])
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
    def get_fitness(self, population, regressor, X_train, X_test, y_train, y_test, ftype='default'):
        # iterate over the individuals in the population (list of feature selections)
        result = []
        predictions = []
        score = []
        for individual in population:

            # fit the regressor to the data with the individuals selection of columns/features
            regressor.fit(X_train[individual], y_train)

            # get predictions of test data
            prediction = regressor.predict(X_test[individual])

            # store the predictions for graphing
            predictions.append(prediction)

            # get the score
            score.append(regressor.score(X_test[individual], y_test))

            # individual fitness score, mean squared error
            mse = mean_squared_error(y_test, prediction)

            # append new individual fitness algorithms to if else statements below and define a stype parameter for it
            if (ftype == 'default'):
                pass
                
            elif (ftype == 'score'):
                pass
            elif (ftype == 'new'):
                pass

            # append the score to the list of results
            result.append(mse)
        
        # append new population fitness algorithms to if else statements below and define a stype parameter for it
        if (ftype == 'default'):
            # convert fitness to greater number better (because fitness is used as probability of being picked for
            # mating pool)
            fitness = np.sqrt(np.array(result))
            fitness = (np.max(result) - result)**0.5
        elif (ftype == 'score'):
            fitness = minmax_scale(score)
        elif (ftype == 'new'):
            pass
            
        return result, predictions, fitness

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
        elif (stype == 'exp_rank'):
            desc_idx = np.argsort(fitness)[::-1]
            c = 0.9
            N = len(desc_idx)
            exp_rank = lambda i, c, N: ((1-c)/(1-c**N))*c**(i)
            probs = [exp_rank(i, c, N) for i in range(N)]
            idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=probs)
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

    # crossover 2 function is an alternative function of crossover
    # this will do crossover of the pair of parents with crossover method depending on type of crossover selected and return 2 offsprings
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
            child = np.logical_xor(mutations, child)
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

    # evolve 2 : alternative evolution function
    # This will pick in sequence, pair of parents in the mating pool to be crossovered
    # this will call function crossover 2 
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
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = np.vstack(list(pop) + list(elite))

    # evolution function creating the next generation/population based on fitness with exponential rank selection
    # fitness:  np array of fitness scores for the current population
    #
    def evolve_exp_rank(self, fitness):
        pop = self.select(fitness, stype='exp_rank')
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


    
