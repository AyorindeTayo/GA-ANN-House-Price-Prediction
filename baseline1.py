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

# scikitlearn MLPRegressor with hidden layers=4 as fitness maker

# baseline values
INITRATE_BASELINE = 0.5
CROSSRATE_BASELINE = 0.5
MUTATERATE_BASELINE = 0.02
POPSIZE_BASELINE = 100
GENSIZE_BASELINE = 20

# params
params = ['initrate', 'crossrate', 'mutaterate', 'popsize', 'gensize']

# cycles
initrates=np.arange(0.1, 1, 0.1)
crossrates=np.arange(0.1, 1, 0.1)
mutaterates=np.arange(0.001, 0.045, 0.005)
popsizes=np.arange(50, 140, 10)
gensizes=np.arange(5, 50, 5)

cycles = []
cycles.append([str(round(value,1)) for value in initrates])
cycles.append([str(round(value,1)) for value in crossrates])
cycles.append([str(round(value,3)) for value in mutaterates])
cycles.append([str(value) for value in popsizes])
cycles.append([str(value) for value in gensizes])

# unique
unique = 20

# features for baseline1
features = np.load('octiba/ga_module/saleprice_features/20_features.npy')
features = np.delete(features, np.where(features == 'SalePrice'), axis=0)

# create multiindex
index_tuples = []
index_tuples += [('initrate', str(round(value,1)), int(gen+1)) for value in initrates for gen in range(20)]
index_tuples += [('crossrate', str(round(value,1)), int(gen+1)) for value in crossrates for gen in range(20)]
index_tuples += [('mutaterate', str(round(value,3)), int(gen+1)) for value in mutaterates for gen in range(20)]
index_tuples += [('popsize', str(value), int(gen+1)) for value in popsizes for gen in range(20)]
for g in gensizes:
    for gen in range(g):
        index_tuples += [('gensize', str(g), int(gen+1))]
index = pd.MultiIndex.from_tuples(index_tuples, names=['param', 'value', 'gen'])

# create the dataframe
baseline1_df = pd.DataFrame(index=index, columns = ['MSE'] + list(features), dtype='bool')
baseline1_df['MSE'] = baseline1_df['MSE'].astype('float64')
baseline1_df[features] = False
baseline1_df.sort_index(inplace=True)

# open the data and put into dataframe
for index, p in enumerate(params):
    for value in cycles[index]:
        evolution = np.load('octiba/ga_module/{}/{}_{}_{}.npy'.format('evolutions', unique, p, value))
        feats = np.load('octiba/ga_module/{}/{}_{}_{}.npy'.format('features', unique, p, value))
        for gen, mse in enumerate(evolution):
            baseline1_df.loc[(p, value, int(gen+1)), 'MSE'] = mse
            baseline1_df.loc[(p, value, int(gen+1)), feats[gen]] = True

# save the dataframe
baseline1_df.to_pickle('octiba/dataframes/baseline1.pkl')
print(baseline1_df)
