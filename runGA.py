from ga_module.ga import GA
from data_module.data import data_cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

# run feature select GA
def run(data, target, init_ratio, cross_rate, mutate_rate, pop_size, n_generations, elitism, 
    ga_ann_iterations=20, ga_ann_layers=2, mape_ann_iterations=1000, mape_ann_layers=4,
    ga_score='score', ga_evolve='elitism',
    final_mape_idx=-1):

    # get the features from the data
    features = data.drop(target, axis=1).columns

    # scale the data
    scaled_data = scale_data(data, target)

    # do initial split of data
    train, test, X_train, X_test, y_train, y_test = initial_split(scaled_data, target)

    # further split into x/y train/test
    train_X_train, train_X_test, train_y_train, train_y_test = split_data(train, target)

    # initialize the ANN regressor for insise GA
    ga_regressor = MLPRegressor(
        hidden_layer_sizes=(ga_ann_layers,),
        activation='relu',
        solver='lbfgs',
        learning_rate='adaptive',
        max_iter=ga_ann_iterations,
        learning_rate_init=0.01,
        alpha=0.01
    )

    # initialize the ANN regressor for MAPE
    mape_regressor1 = MLPRegressor(
        hidden_layer_sizes=(mape_ann_layers,),
        activation='relu',
        solver='lbfgs',
        learning_rate='adaptive',
        max_iter=mape_ann_iterations,
        learning_rate_init=0.01,
        alpha=0.01
    )

    # initialize the ANN regressor for MAPE (renny added)
    mape_regressor2 = MLPRegressor(
        hidden_layer_sizes=(mape_ann_layers,),
        activation='relu',
        solver='lbfgs',
        learning_rate='adaptive',
        max_iter=mape_ann_iterations,
        learning_rate_init=0.01,
        alpha=0.01
    )

    # initialize the GA
    ga = GA(features, init_ratio, cross_rate, mutate_rate, pop_size, elitism)

    # do initial MAPE score - without feature selection
    mape_regressor1.fit(X_train, y_train)
    initial_mape_prediction = mape_regressor1.predict(X_test)
    mape_y_test = np.array(list(y_test))
    initial_mape_prediction = np.array(list(initial_mape_prediction))
    initial_mape = mean_absolute_percentage_error(mape_y_test, initial_mape_prediction)

    # start the GA
    evolution = []
    bestFeatures = []
    bestPredictions = []
    print('RMSE and number of features for best individuals in each generation :')
    for generation in range(n_generations):
        feature_selections = ga.readChromosomes(ga.pop)
        mse, predictions, fitness = ga.get_fitness(feature_selections, ga_regressor, train_X_train, train_X_test, train_y_train, train_y_test, ftype=ga_score)
        bestIdx = np.argmax(fitness)
        evolution.append(mse[bestIdx])
        bestFeatures.append(feature_selections[bestIdx])
        bestPredictions.append(predictions[bestIdx])
        if (ga_evolve == 'elitism'):
            ga.evolve_elitism(fitness)
        elif (ga_evolve == 'evolve2'):
            ga.evolve2(fitness)
        elif (ga_evolve == 'exp_rank'):
            ga.evolve_exp_rank(fitness)
        else:
            ga.evolve(fitness)
        print(generation, mse[bestIdx] ** 0.5, len(feature_selections[bestIdx]))

    
    #try to get best features based on the lowest MSE
    if(final_mape_idx =='best'):
        final_mape_idx = np.argmin(evolution)
        print('best index : ',final_mape_idx)

    #do final MAPE score based on initial train and test data - with feature selection
    mape_regressor2.fit(X_train[bestFeatures[final_mape_idx]], y_train)
    final_mape_prediction = mape_regressor2.predict(X_test[bestFeatures[final_mape_idx]])
    final_mape_prediction = np.array(list(final_mape_prediction))
    final_mape = mean_absolute_percentage_error(mape_y_test, final_mape_prediction)

    return evolution, bestFeatures, bestPredictions, initial_mape, final_mape, mape_y_test, final_mape_prediction,initial_mape_prediction,train_y_test
