# Version 0.0

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import octiba.ann_module.neural_network as nn

class ANNModule():
	def __init__(self):
		pass
	@staticmethod
	def initNetwork(data):
		# Initialize neural network.
		net = nn.NeuralNetwork()
		nInn = data.shape[1]
		net.setLayers(np.array([
			nInn,
			1,
		]))
		f = .25
		# net.setLayers(np.array([
		# 	nInn,
		# 	max(int(nInn* 1/2/f),1),
		# 	max(int(nInn* 1/4/f),1),
		# 	max(int(nInn* 1/8/f),1),
		# 	1,
		# ]))
		# print(max(int(nInn* 1/2/f),1),
		# 	max(int(nInn* 1/4/f),1),
		# 	max(int(nInn* 1/8/f),1))
		net.setInput(data)
		
		return net
	@staticmethod
	def fit(data,solution,precision=1e4,minError=0.1,mutationRate=1e-4):
		# Error check.
		if type(data) != np.ndarray:
			raise ValueError('Argument \'data\' is of type '+str(type(data))+', expected np.ndarray.')
		# Initialize neural network.
		net = ANNModule.initNetwork(data)
		net.mutationRate = mutationRate
		# Train weights.
		bestScore = np.inf
		bestWeights = None
		bestOutput = None
		allOut = []
		allWeights = []
		allBestScores = []
		i=0
		while i<precision:
			output = net.getOutput().reshape([-1])
			allOut.append(output)
			score = np.power(np.abs(output-solution),2).sum()
			#score = ((output/solution-1)**2).sum()
			if score < bestScore:
				bestScore = score
				#allBestScores.append(bestScore)
				if i%10==0:
					print(bestScore)
				if bestScore < minError:
					break
				if bestWeights != None:
					net.accelerate(bestWeights)
				bestWeights = net.getWeights()
				bestOutput = output

				allWeights.append(bestWeights)

				net.mutationRate *= 1.1
			else:
				net.setWeights(bestWeights)
				net.deAccelerate()
				net.mutationRate /= 1.01
			i+=1
			net.mutate()
			allBestScores.append(bestScore)
		#print(np.append(np.append(solution.reshape([-1,1]),output.reshape([-1,1]),axis=1),(solution-output).reshape([-1,1]),axis=1))
		#print(bestScore)
		model = Model(bestWeights)
		return {'model':model,'weights':bestWeights, 'score':bestScore,'history':allOut,'weightLog':allWeights,'allBestScores':allBestScores}
class Model:
	def __init__(self,weights):
		self.weights = weights
	def predict(self,data):
		# Initialize neural network.
		net = ANNModule.initNetwork(data)
		net.setWeights(self.weights)

		return net.getOutput().reshape([-1])

# TEST.
## Create data.
#data = np.random.rand(int(1e2),3)*10-5
#solution = data[:,0]*data[:,1]/(data[:,2]+20)*20

##
#ann = ANNModule()
#(weights,score) = ann.getWeights(data,solution)
#prediction = ann.getSolution(data,weights)

#print(np.append(np.append(solution.reshape([-1,1]),prediction.reshape([-1,1]),axis=1),(solution-prediction).reshape([-1,1]),axis=1))
#print(score)
