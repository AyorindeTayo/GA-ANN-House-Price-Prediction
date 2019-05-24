from ga_module.meta_ga import meta_GA
from data_module.data import data_cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# returns scaled data, does not scale target
def scale_data(data, target):

    # create a result vector minus the target column
    result = data.drop(target, axis=1).copy()

    # do normalization of the data
    scaler = StandardScaler()
    scaler.fit(result)
    result = scaler.transform(result)

    # convert the data from normalization (np.arrays) into pandas dataframes again
    # and set the column names
    result = pd.DataFrame(result)
    result.columns = data.drop(target, axis=1).columns

    # reintroduce the target column
    result[target] = data[target]
    
    #print('result target:',result[target])

    return result

# returns test and train data, does not remove target
def initial_split(data, target):
    # divide data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # recombine the y and x
    train = X_train.copy()
    train[target] = y_train
    test = X_test.copy()
    test[target] = y_test

    return train, test, X_train, X_test, y_train, y_test

# splits the data into x/y train and test
def split_data(data, target):
    # divide data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    return X_train, X_test, y_train, y_test

# MAPE formula
def mean_absolute_percentage_error(y_test, pred): 
    return np.mean(np.abs((y_test - pred) / y_test)) * 100

# meta GA run function
def meta_run(data, target, ranges, init_ratio, cross_rate, mutate_rate, pop_size, n_generations, elitism):

    # get the features from the data
    features = data.drop(target, axis=1).columns

    # scale the data
    scaled_data = scale_data(data, target)

    # do initial split of data
    train, test, X_train, X_test, y_train, y_test = initial_split(scaled_data, target)

    # further split into x/y train/test
    train_X_train, train_X_test, train_y_train, train_y_test = split_data(train, target)

    # initialize the meta GA
    meta_ga = meta_GA(ranges, init_ratio, cross_rate, mutate_rate, pop_size, elitism)

    # start the meta GA
    evolution = []
    bestFeatures = []
    bestPredictions = []
    bestParams = []
    for generation in range(n_generations):
        params = meta_ga.readChromosomes()
        result, fitness, predictions, feats = meta_ga.get_fitness(features, train_X_train, train_X_test, train_y_train, train_y_test, X_train, X_test, y_train, y_test)
        bestIdx = np.argmax(fitness)
        evolution.append(result[bestIdx])
        bestFeatures.append(feats[bestIdx])
        bestPredictions.append(predictions[bestIdx])
        bestParams.append(params[bestIdx])
        meta_ga.evolve_elitism(fitness)
        print(generation, result[bestIdx], len(feats[bestIdx]), params[bestIdx])


    return evolution, bestFeatures, bestPredictions, y_test, bestParams

# get the cleaned data
data = data_cleaning('octiba/data_module/AmesHousing2.csv', normalize=False)
target = 'SalePrice'

# set the parameter value ranges
init_ratios = np.arange(0.2, 1, 0.1)
cross_rates = np.arange(0.01, 1, 0.05)
mutate_rates = np.arange(0.001, 0.045, 0.005)
pop_sizes = np.arange(10, 141, 10)
gensizes = np.arange(5, 41, 5)
elitisms = np.arange(0.01,0.1,0.01)
ga_ann_iterations = np.array([20,50,100,200])
ga_ann_layers = np.array([1,2,3,4])
ranges = [init_ratios, cross_rates, mutate_rates, pop_sizes, gensizes, elitisms, ga_ann_iterations, ga_ann_layers]

# run the meta-GA
evolution, bestFeatures, bestPredictions, bestYtests, bestParams = meta_run(data, target, ranges, 0.2, 0.05, 0.001, 3, 2, 0.05)

# get unique value to identify current cycle (to compare values)
unique = str(int(round(time.time())))

# save the data
np.save('octiba/ga_module/evolutions/'+unique, evolution)
np.save('octiba/ga_module/features/'+unique, bestFeatures)
np.save('octiba/ga_module/predictions/'+unique, bestPredictions)
np.save('octiba/ga_module/y_tests/'+unique, bestYtests)
np.save('octiba/ga_module/params/'+unique, bestParams)