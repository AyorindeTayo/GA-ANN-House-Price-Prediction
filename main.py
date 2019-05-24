from ga_module.ga import GA
from data_module.data import data_cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
        hidden_layer_sizes=(2,),
        activation='relu',
        solver='lbfgs',
        learning_rate='adaptive',
        max_iter=2,
        learning_rate_init=0.01,
        alpha=0.01
    )

    return regressor, X_train, X_test, y_train, y_test

def runGA(data, target, init_ratio, cross_rate, mutate_rate, pop_size, n_generations, elitism=0):

    # get the features from the data
    features = data.drop(target, axis=1).columns

    # initalize the regressor and split and normalize data
    regressor, X_train, X_test, y_train, y_test = scikitANNregressor(data, target)

    # initialize the GA
    ga = GA(features, init_ratio, cross_rate, mutate_rate, pop_size, elitism)

    # start the GA
    evolution = []
    bestFeatures = []
    bestPredictions = []
    for generation in range(n_generations):
        feature_selections = ga.readChromosomes(ga.pop)
        mse, predictions, fitness = ga.get_fitness(feature_selections, regressor, X_train, X_test, y_train, y_test, ftype='score')
        bestIdx = np.argmax(fitness)
        evolution.append(mse[bestIdx])
        bestFeatures.append(feature_selections[bestIdx])
        bestPredictions.append(predictions[bestIdx])
        ga.evolve_elitism(fitness)
        print(generation, mse[bestIdx] ** 0.5, len(feature_selections[bestIdx]))

    return evolution, bestFeatures, bestPredictions, y_test

def runRegressorAndPlot(X_train, y_train, X_test, y_test):

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

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    regressor.fit(X_train, y_train)
    prediction = regressor.predict(X_test)

    x = np.arange(0., len(y_test), 1)

    y = np.array(list(y_test))
    p = np.array(list(prediction))
    idx = y.argsort()

    print(mean_absolute_percentage_error(y_test, prediction))

    plt.figure(figsize=(15,7))
    plt.subplot(111)
    plt.plot(x, y[idx], 'ro', x, p[idx], 'bo')
    plt.show()

def mean_absolute_percentage_error(y_test, pred): 
    return np.mean(np.abs((y_test - pred) / y_test)) * 100

# get the cleaned data
data = data_cleaning('octiba/data_module/AmesHousing2.csv', normalize=False)
target = 'SalePrice'

# initial split of data
# divide data into features and target
X = data.drop(target, axis=1)
y = data[target]

# split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y)

data_ga = X_train.copy()
data_ga[target] = y_train

runRegressorAndPlot(X_train, y_train, X_test, y_test)

# run the GA
evolution, feats, preds, _ = runGA(data_ga, target, 0.1, 0.05, 0.001, 130, 150, 0.08)

# plot the data
x = np.arange(0., len(evolution), 1)

plt.figure(figsize=(15,7))
plt.subplot(111)
plt.plot(x, evolution, 'ro', x, [len(x) for x in feats], 'bo')
plt.show()

runRegressorAndPlot(X_train[feats[-1]], y_train, X_test[feats[-1]], y_test)

'''
bestFeatures = ['MSZoning', 'LotArea', 'Street', 'Alley', 'LotShape',
       'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageQual', 'PavedDrive', 'PoolQC', 'MiscVal', 'YrSold',
       'SaleCondition','SalePrice']

#regressor, X_train, X_test, y_train, y_test = scikitANNregressor(data[['OverallQual', 'OverallCond', 'BsmtQual', 'BsmtExposure', 'GrLivArea', 'PoolArea', 'SalePrice']], target)
regressor, X_train, X_test, y_train, y_test = scikitANNregressor(data[bestFeatures], target)

regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)

x = np.arange(0., len(y_test), 1)

y = np.array(list(y_test))
p = np.array(list(prediction))
idx = y.argsort()

print(np.sqrt(mean_squared_error(y_test, prediction)))

plt.figure(figsize=(15,7))
plt.subplot(111)
plt.plot(x, y[idx], 'ro', x, p[idx], 'bo')
plt.show()
'''
