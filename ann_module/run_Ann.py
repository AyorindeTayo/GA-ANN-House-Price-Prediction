# Version 0.0
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

import numpy as np
import ann_module as ann
import octiba.data_module.DataCleaning as dc

# Load rennys data.
df = dc.data_cleaning('octiba/AmesHousing.csv')
## Could use only most predictive columns.
#df = df[['OverallQual','OverallCond','YearBuilt','BsmtQual','BsmtExposure','GrLivArea','PoolArea','SalePrice']]
index = 2300
df1 = df.iloc[0:index,:]
df2 = df.iloc[index:,:]
solution1 = df1.iloc[:,-1].values
solution2 = df2.iloc[:,-1].values
#for i in range(78):
data1 = df1.iloc[:,0:-1].values
data2 = df2.iloc[:,0:-1].values

# Create prediction.
for i in range(1):
	data = ann.ANNModule.fit(data1,solution1,precision=1e4,minError=1,mutationRate=2e-2)
	(model,weights,score) = data['model'],data['weights'],data['score']

	# To store data for the animator.
	#pickle.dump((data['history'],solution1), open( "octiba/ann_module/score_data/history3.p", "wb" ) )
	#pickle.dump(data['allBestScores'], open( 'octiba/ann_module/score_data/scores_'+str(np.round(score,8-len(str(score))))+'_'+str(i)+'.p', "wb" ) )
	#pickle.dump(data['allBestScores'], open( 'octiba/ann_module/score_data/scores_all_feat_3_hid_3.p', "wb" ) )

# To store/load the weights for re-use
#pickle.dump((weights,score), open( "weights.p", "wb" ) )
#(weights,score) = pickle.load( open( "weights_linear_25045.p", "rb" ) )

prediction = model.predict(data2)

# Test the prediction.
## Print solution, prediction and difference in each column.
#print(np.append(np.append(solution.reshape([-1,1]),prediction.reshape([-1,1]),axis=1),(solution-prediction).reshape([-1,1]),axis=1))
print('Fitted squared error score: '+str(score))

indices = np.argsort(solution2)

print('Prc: ',np.mean(np.abs(prediction[indices]-solution2[indices])/solution2[indices]))

print('RMSE: ',np.mean(np.abs(prediction[indices]-solution2[indices])))

# Plot prediction.
fig=plt.figure(figsize=(15,10))
p=fig.add_subplot(1,1,1)
p.plot(prediction[indices],'bo',color='#FF0000')
p.plot(solution2[indices],'bo',color='#00FF00')
plt.show()




