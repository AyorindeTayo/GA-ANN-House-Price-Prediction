from .ga import GA
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

# MAPE formula
def mean_absolute_percentage_error(y_test, pred): 
    return np.mean(np.abs((y_test - pred) / y_test)) * 100

# run the feature select GA inside meta GA
def run(features, train_X_train, train_X_test, train_y_train, train_y_test, X_train, X_test, y_train, y_test, init_ratio, cross_rate, mutate_rate, pop_size, n_generations, elitism, 
    ga_ann_iterations=20, ga_ann_layers=2, mape_ann_iterations=1000, mape_ann_layers=4,
    ga_score='score', ga_evolve='elitism',
    final_mape_idx=-1):

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

    # do initial MAPE score (Renny Added)
    mape_regressor1.fit(X_train, y_train)
    initial_mape_prediction = mape_regressor1.predict(X_test)
    mape_y_test = np.array(list(y_test))
    initial_mape_prediction = np.array(list(initial_mape_prediction))
    initial_mape = mean_absolute_percentage_error(mape_y_test, initial_mape_prediction)

    # start the GA
    evolution = []
    bestFeatures = []
    bestPredictions = []
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

    #do final MAPE score based on initial train and test data (Renny Added)
    mape_regressor2.fit(X_train[bestFeatures[final_mape_idx]], y_train)
    final_mape_prediction = mape_regressor2.predict(X_test[bestFeatures[final_mape_idx]])
    final_mape_prediction = np.array(list(final_mape_prediction))
    final_mape = mean_absolute_percentage_error(mape_y_test, final_mape_prediction)

    return evolution, bestFeatures, bestPredictions, initial_mape, final_mape, mape_y_test, final_mape_prediction,initial_mape_prediction