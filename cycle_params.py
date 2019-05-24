from ga_module.ga import GA
from data_module.data import data_cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# set baseline values
INITRATE_BASELINE = 0.5
CROSSRATE_BASELINE = 0.5
MUTATERATE_BASELINE = 0.02
POPSIZE_BASELINE = 100
GENSIZE_BASELINE = 20
ELITISM_BASELINE = 0.05

# define target value
target = 'SalePrice'

# initialize the regressor
def scikitANNregressor(data, target):
    # divide data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # do normalization of the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # convert the data from normalization (np.arrays) into pandas dataframes again
    # and set the column names
    X_train = pd.DataFrame(X_train)
    X_train.columns = data.drop(target, axis=1).columns
    X_test = pd.DataFrame(X_test)
    X_test.columns = data.drop(target, axis=1).columns

    # initialize the ANN regressor
    regressor = MLPRegressor(
        hidden_layer_sizes=(4,),
        activation='relu',
        solver='lbfgs',
        learning_rate='adaptive',
        max_iter=1000,
        learning_rate_init=0.01,
        alpha=0.01
    )

    return regressor, X_train, X_test, y_train, y_test

# run the GA algorithm function
def runGA(data, target, init_ratio, cross_rate, mutate_rate, pop_size, n_generations, elitism, regressor, X_train, X_test, y_train, y_test):

    # get the features from the data
    features = data.drop(target, axis=1).columns

    # initialize the GA
    ga = GA(features, init_ratio, cross_rate, mutate_rate, pop_size, elitism)

    # start the GA
    evolution = []
    bestFeatures = []
    bestPredictions = []
    for generation in range(n_generations):
        feature_selections = ga.readChromosomes(ga.pop)
        fitness, predictions, probs = ga.get_fitness(feature_selections, regressor, X_train, X_test, y_train, y_test)
        bestIdx = np.argmax(probs)
        evolution.append(fitness[bestIdx])
        bestFeatures.append(feature_selections[bestIdx])
        bestPredictions.append(predictions[bestIdx])
        ga.evolve_elitism(probs)
        print(generation, fitness[bestIdx] ** 0.5, len(feature_selections[bestIdx]))

    return evolution, bestFeatures, bestPredictions

# save function to put np arrays in files
def save_to_file(evolution, feats, preds, unique, value):
    np.save('octiba/ga_module/evolutions/'+unique+'_'+value, evolution)
    np.save('octiba/ga_module/features/'+unique+'_'+value, feats)
    np.save('octiba/ga_module/predictions/'+unique+'_'+value, preds)

# cycle functions
def initrate_cycler(data, unique, regressor, X_train, X_test, y_train, y_test, target=target,
    initrates=np.arange(0.1, 1, 0.1),
    crossrate=CROSSRATE_BASELINE,
    mutaterate=MUTATERATE_BASELINE,
    popsize=POPSIZE_BASELINE,
    gensize=GENSIZE_BASELINE,
    elitism=ELITISM_BASELINE):

    for value in initrates:
        evolution, feats, preds = runGA(data, target, value, crossrate, mutaterate, popsize, gensize, elitism, regressor, X_train, X_test, y_train, y_test)
        save_to_file(evolution, feats, preds, unique, 'initrate_'+str(value))

def crossrate_cycler(data, unique, regressor, X_train, X_test, y_train, y_test, target=target,
    initrate=INITRATE_BASELINE,
    crossrates=np.arange(0.1, 1, 0.1),
    mutaterate=MUTATERATE_BASELINE,
    popsize=POPSIZE_BASELINE,
    gensize=GENSIZE_BASELINE,
    elitism=ELITISM_BASELINE):

    for value in crossrates:
        evolution, feats, preds = runGA(data, target, initrate, value, mutaterate, popsize, gensize, elitism, regressor, X_train, X_test, y_train, y_test)
        save_to_file(evolution, feats, preds, unique, 'crossrate_'+str(value))

def mutaterate_cycler(data, unique, regressor, X_train, X_test, y_train, y_test, target=target,
    initrate=INITRATE_BASELINE,
    crossrate=CROSSRATE_BASELINE,
    mutaterates=np.arange(0.001, 0.045, 0.005),
    popsize=POPSIZE_BASELINE,
    gensize=GENSIZE_BASELINE,
    elitism=ELITISM_BASELINE):

    for value in mutaterates:
        evolution, feats, preds = runGA(data, target, initrate, crossrate, value, popsize, gensize, elitism, regressor, X_train, X_test, y_train, y_test)
        save_to_file(evolution, feats, preds, unique, 'mutaterate_'+str(value))

def popsize_cycler(data, unique, regressor, X_train, X_test, y_train, y_test, target=target,
    initrate=INITRATE_BASELINE,
    crossrate=CROSSRATE_BASELINE,
    mutaterate=MUTATERATE_BASELINE,
    popsizes=np.arange(50, 140, 10),
    gensize=GENSIZE_BASELINE,
    elitism=ELITISM_BASELINE):

    for value in popsizes:
        evolution, feats, preds = runGA(data, target, initrate, crossrate, mutaterate, value, gensize, elitism, regressor, X_train, X_test, y_train, y_test)
        save_to_file(evolution, feats, preds, unique, 'popsize_'+str(value))

def gensize_cycler(data, unique, regressor, X_train, X_test, y_train, y_test, target=target,
    initrate=INITRATE_BASELINE,
    crossrate=CROSSRATE_BASELINE,
    mutaterate=MUTATERATE_BASELINE,
    popsize=POPSIZE_BASELINE,
    gensizes=np.arange(5, 50, 5),
    elitism=ELITISM_BASELINE):

    for value in gensizes:
        evolution, feats, preds = runGA(data, target, initrate, crossrate, mutaterate, popsize, value, elitism, regressor, X_train, X_test, y_train, y_test)
        save_to_file(evolution, feats, preds, unique, 'gensize_'+str(value))

def elitism_cycler(data, unique, regressor, X_train, X_test, y_train, y_test, target=target,
    initrate=INITRATE_BASELINE,
    crossrate=CROSSRATE_BASELINE,
    mutaterate=MUTATERATE_BASELINE,
    popsize=POPSIZE_BASELINE,
    gensize=GENSIZE_BASELINE,
    elitism=np.arange(0.01,0.1,0.01)):

    for value in elitism:
        evolution, feats, preds = runGA(data, target, initrate, crossrate, mutaterate, popsize, gensize, value, regressor, X_train, X_test, y_train, y_test)
        save_to_file(evolution, feats, preds, unique, 'elitism_'+str(value))

# get the cleaned data
data = data_cleaning('octiba/data_module/AmesHousing2.csv', normalize=False)

# initalize the regressor and split and normalize data
regressor, X_train, X_test, y_train, y_test = scikitANNregressor(data, target)

# get unique value to identify current cycle (to compare values)
unique = str(int(round(time.time())))

# save the y_test for this cycle
np.save('octiba/ga_module/y_tests/'+unique+'_y_test', y_test)

# save the features for this cycle
np.save('octiba/ga_module/saleprice_features/'+unique+'_features', data.drop(target, axis=1).columns)

# run the cycles on the parameters
initrate_cycler(data, unique, regressor, X_train, X_test, y_train, y_test)
crossrate_cycler(data, unique, regressor, X_train, X_test, y_train, y_test)
mutaterate_cycler(data, unique, regressor, X_train, X_test, y_train, y_test)
popsize_cycler(data, unique, regressor, X_train, X_test, y_train, y_test)
gensize_cycler(data, unique, regressor, X_train, X_test, y_train, y_test)
elitism_cycler(data, unique, regressor, X_train, X_test, y_train, y_test)

