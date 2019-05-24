
import graphics as g

import numpy as np

exampleSolution = np.arange(1,1000)
examplePrediction = [exampleSolution + (np.random.random(exampleSolution.shape[0])*2-1)**7*10000/(i*.1+1) for i in range(1000)]

ani = g.Animation()
ani.prediction(examplePrediction,exampleSolution)