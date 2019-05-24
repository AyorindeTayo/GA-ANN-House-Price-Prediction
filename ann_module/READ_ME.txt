# Version 0.0

File structure:
	Run the 'run_Ann.py' file to calculate a prediction.
		The 'ann_module.py' file is used to optimize find a model.
		The 'neural_network.py' is used by the 'ann_module.py' as a neural network data structure.
	Run the 'run_graphics.py' file to animate data.
		The 'graphics.py' file is used by the 'run_graphics.py' file for animations.

How to use the ann module:
	Example:
		import numpy as np
		import ann_module as ann

		# Generate some random data and solution.
		data = np.random.rand(int(1e2),3)*10-5
		solution = data[:,0]*data[:,1]/(data[:,2]+20)*20

		# Create prediction.
		model = ann.ANNModule.fit(data,solution,precision=1e4)['model']
		prediction = model.predict(data)

		# Test the prediction.
		print(np.append(np.append(solution.reshape([-1,1]),prediction.reshape([-1,1]),axis=1),(solution-prediction).reshape([-1,1]),axis=1)) # Print solution, prediction and difference in each column.








