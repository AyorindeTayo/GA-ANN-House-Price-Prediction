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

# scikitlearn MLPRegressor with hidden layers=4 as fitness maker, using evolve_elitism and same x_train, x_test for all

# baseline values
INITRATE_BASELINE = 0.5
CROSSRATE_BASELINE = 0.5
MUTATERATE_BASELINE = 0.02
POPSIZE_BASELINE = 100
GENSIZE_BASELINE = 20
ELITISM_BASELINE = 0.05

# params
params = ['initrate', 'crossrate', 'mutaterate', 'popsize', 'gensize', 'elitism']

# cycles
initrates=np.arange(0.1, 1, 0.1)
crossrates=np.arange(0.1, 1, 0.1)
mutaterates=np.arange(0.001, 0.045, 0.005)
popsizes=np.arange(50, 140, 10)
gensizes=np.arange(5, 50, 5)
elitisms=np.arange(0.01,0.1,0.01)

cycles = []
cycles.append([str(round(value,1)) for value in initrates])
cycles.append([str(round(value,1)) for value in crossrates])
cycles.append([str(round(value,3)) for value in mutaterates])
cycles.append([str(value) for value in popsizes])
cycles.append([str(value) for value in gensizes])
cycles.append([str(round(value,2)) for value in elitisms])

# unique
unique = str(1542601227)

# features for baseline3
features = np.load('octiba/ga_module/saleprice_features/'+unique+'_features.npy')

# create multiindex
index_tuples = []
index_tuples += [('initrate', str(round(value,1)), int(gen+1)) for value in initrates for gen in range(20)]
index_tuples += [('crossrate', str(round(value,1)), int(gen+1)) for value in crossrates for gen in range(20)]
index_tuples += [('mutaterate', str(round(value,3)), int(gen+1)) for value in mutaterates for gen in range(20)]
index_tuples += [('popsize', str(value), int(gen+1)) for value in popsizes for gen in range(20)]
for g in gensizes:
    for gen in range(g):
        index_tuples += [('gensize', str(g), int(gen+1))]
index_tuples += [('elitism', str(value), int(gen+1)) for value in elitisms for gen in range(20)]
index = pd.MultiIndex.from_tuples(index_tuples, names=['param', 'value', 'gen'])

# create the dataframe
baseline3_df = pd.DataFrame(index=index, columns = ['MSE'] + list(features), dtype='bool')
baseline3_df['MSE'] = baseline3_df['MSE'].astype('float64')
baseline3_df[features] = False
baseline3_df.sort_index(inplace=True)

# open the data and put into dataframe
for index, p in enumerate(params):
    for value in cycles[index]:
        evolution = np.load('octiba/ga_module/{}/{}_{}_{}.npy'.format('evolutions', unique, p, value))
        feats = np.load('octiba/ga_module/{}/{}_{}_{}.npy'.format('features', unique, p, value))
        for gen, mse in enumerate(evolution):
            baseline3_df.loc[(p, value, int(gen+1)), 'MSE'] = mse
            baseline3_df.loc[(p, value, int(gen+1)), feats[gen]] = True

# save the dataframe
baseline3_df.to_pickle('octiba/dataframes/baseline3.pkl')
print(baseline3_df)
