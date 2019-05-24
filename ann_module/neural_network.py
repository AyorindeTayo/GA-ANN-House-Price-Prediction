# Version 0.0

import numpy as np
import numbers

class NeuralNetwork:
	def __init__(self):
		self.input = None
		self.layers = None
		self.weights = []
		self.velocities = []
		self.mutationRate = 1e-4
		self.isInDebugMode = True
		
		smooth = 1
		self.relu1 = np.vectorize(lambda x:np.log(1+np.exp(x*smooth))/smooth)
		self.relu2 = np.vectorize(lambda x:(x+np.sqrt(x**2+1.5/smooth**2))/2) # TODO: Make in place.
	def setLayers(self,layers):
		# Test input.
		if self.isInDebugMode:
			if type(layers) is not np.ndarray:
				raise ValueError('The \'layers\' argument is not of type \'np.ndarray\', but is of type '+str(type(layers)))
			for numNodes in layers:
				if type(numNodes) is not int and type(numNodes) is not np.int32:
					raise ValueError('The \'layers\' argument contain element(s) that are not integers.')
		# Construt layers.
		self.layers = layers.copy()
		# Construct matrices.
		prev = self.layers[0]
		i = 1
		while i < len(layers):
			current = self.layers[i]
			# Weights.
			w = (1-np.random.random((prev+1)*current)*2).reshape([prev+1,current]) # +1 due to allow constants being added in the neural network.
			self.weights.append(w)
			# Velocities
			v = np.zeros(w.shape)
			self.velocities.append(v)
			prev = current
			i+=1
	def setInput(self,input):
		# Check input.
		if self.isInDebugMode:
			if self.layers is None:
				raise ValueError('The network has no layers. Call the \'NeuralNetwork.setLayers(layers)\' method to set the layers. See the \'how_to_use_ann.txt\' file for documentation.')
			if type(input) is not np.ndarray:
				raise ValueError('The \'input\' argument is not of type \'np.ndarray\', but is of type '+str(type(input)))
			if len(input.shape) != 2:
				raise ValueError('The \'input\' argument has the wrong format. Expected 2d np.ndarray. See \'how_to_use_ann.txt\' file for documentation.')
			if input.shape[1] != self.layers[0]:
				raise ValueError('The \'input\' argument has a different number of columns than the number of input nodes. \'input\' has '+str(input.shape[1])+' number of columns, expected '+str(self.layers[0])+'.')
			for row in input:
				for num in row:
					if type(num) is not int and type(num) is not np.int32 and type(num) is not np.float64:
						raise ValueError('The \'input\' argument contain element(s) that are not integers.')
			if input.shape[1] != self.layers[0]:
				raise ValueError('The \'input\' argument has a different number of elements than the number of input nodes. \'input\' has '+str(len(input))+' but the first layer has '+str(self.layers[0])+' nodes.')
		# Set input.
		self.input = input.reshape([input.shape[0],1,input.shape[1]]) # Input is a stack of n vectors (first dimension). n is the size of the data. The vectors are 1 by m dimensions (second and third dimension). m is the numer of features.
	def getOutput(self):
		# Error check.
		if self.isInDebugMode:
			if self.layers is None:
				raise ValueError('The network has no layers. Call the \'NeuralNetwork.setLayers(layers)\' method to set the layers. See the \'how_to_use_ann.txt\' file for documentation.')
		# Run network.
		output = self.input
		i = 0
		while i < len(self.weights):
			# Append constant.
			output = np.append(output,np.ones(output.shape[0]).reshape([output.shape[0],1,1]),axis=2)
			#
			output = np.matmul(output,self.weights[i])
			if i+1 < len(self.weights):
				#output = self.relu2(output) # Apply activation function.
				
				output[output<0] = 0
				
				#output = np.sqrt(output**2+1)
				
				#smooth = 1
				#output = (output+np.sqrt(output**2+1.5/smooth**2))/2
			i+=1
		#
		return output.reshape([-1,self.layers[-1]]) # 2d matrix. Each row is a solution. Each column is one parameter of the solutions.
	#def calculateScore(self,)
	def mutate(self):
		i = 0
		while i < len(self.weights):
			w = self.weights[i]
			self.weights[i] = w+(np.random.rand(w.shape[0],w.shape[1])*2-1)*self.mutationRate+self.velocities[i]
			i+=1
	def getWeights(self):
		listW = self.weights.copy()
		i = 0
		while i<len(listW):
			listW[i] = listW[i].copy()
			i+=1
		return listW
	def setWeights(self,weights):
		# Update weights.
		self.weights = weights.copy()
		i = 0
		while i<len(self.weights):
			self.weights[i] = self.weights[i].copy()
			i+=1
		# Update layers.
		self.layers = []
		for w in self.weights:
			self.layers.append(w.shape[0]-1) # -1 due to constants.
		self.layers.append(self.weights[-1].shape[1])
	def accelerate(self,fromW):
		return
		for i in range(len(self.velocities)):
			self.velocities[i] += self.weights[i] - fromW[i]
	def deAccelerate(self):
		for i in range(len(self.velocities)):
			self.velocities[i] /= 1.5


# net = NeuralNetwork()
# net.setLayers(np.array([3,4,3,2]))
# net.setInput(np.array([
# 	[1,2,3],
# 	[4,3,2],
# 	[2,2,1],
# 	[0,-2,0]
# ]))
# out = net.getOutput()
# print(out)


#a = np.arange(1,13,1).reshape([2,2,3])
#print(np.log(2))









