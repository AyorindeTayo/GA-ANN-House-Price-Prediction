# Version 0.0

import numpy as np
import octiba.ann_module.neural_network as nn

class ga:
	def __init__(self):
		pass
	def initNetwork(self,data):
		# Initialize neural network.
		net = nn.NeuralNetwork()
		nInn = data.shape[1]
		f = .25
		net.setLayers(np.array([
			nInn,
			max(int(nInn* 1/2/f),1),
			max(int(nInn* 1/4/f),1),
			max(int(nInn* 1/8/f),1),
			1,
		]))
		print(max(int(nInn* 1/2/f),1),
			max(int(nInn* 1/4/f),1),
			max(int(nInn* 1/8/f),1))
		net.setInput(data)
		
		return net
	def getWeights(self,data,solution,precision=1e4,minError=0.1):
		# Error check.
		if type(data) != np.ndarray:
			raise ValueError('Argument \'data\' is of type '+str(type(data))+', expected np.ndarray.')
		# Train weights.
		## Rember best.
		bestScore = np.inf
		bestWeights = None
		## Simulate evolution.
		### Create population.
		popSize = 10
		population = []
		for i in range(popSize):
			net = nn.NeuralNetwork()
			net.setInput(data)
			population.append({'score':None,'net':net})

		i=0
		while i<precision:
			# Calculate scores.
			for p in population:
				output = p.net.getOutput().reshape([-1])
				score = np.power(np.abs(output-solution),2).sum()
				p['score'] = score
			p.sort(key=lambda x:x['score'])
			# Get best score.
			if bestScore < p[0]['score']:
				bestScore = p[0]['score']
				bestWeights = p[0]['net'].getWeights()
			# Kill.
			population = population[int(len(population)*.5):]
			# Mate.
			j = 0
			while len(population) < popSize:
				child1, child2 = self.mate(population.net[j],population.net[j+1])
				population += [{'score':None,'net':child1}, {'score':None,'net':child2}]
				j+=1
			# Mutate.
			for p in population:
				p.net.mutate()
			i+=1
		
		return bestWeights, bestScore
	def getSolution(self,data,weights):
		# Initialize neural network.
		net = self.initNetwork(data)
		net.setWeights(weights)

		return net.getOutput().reshape([-1])
	def mate(self,net1,net2):
		newWeights1 = []
		newWeights2 = []

		weights1 = net1.weights
		weights2 = net2.weights
		for i in range(len(weights1)):
			shape = weights1.shape
			weight1 = weights1[i].reshape([-1])
			weight2 = weights2[i].reshape([-1])
			choice = np.random.choice(a=[False, True],size=weight1.shape[0])
			
			combinedWeight1 = choice*weight1+~choice*weight2
			combinedWeight1 = combinedWeight1.reshape(shape)
			combinedWeight2 = ~choice*weight1+choice*weight2
			combinedWeight2 = combinedWeight2.reshape(shape)

			newWeights1.append(combinedWeight1)
			newWeights2.append(combinedWeight2)
		inputData = net1.input
		
		newNet1 = nn.NeuralNetwork()
		newNet1.setInput(inputData)
		
		newNet2 = nn.NeuralNetwork()
		newNet2.setInput(inputData)
		
		return newNet1, newNet2



		

		





















