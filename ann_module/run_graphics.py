
import graphics as g

import numpy as np
import pickle
import time

# Load data.
(prediction1,solution1) = pickle.load( open( "score_data/history3.p", "rb" ) )
(prediction2,solution2) = pickle.load( open( "score_data/history2.p", "rb" ) )

weightTime = pickle.load( open( "score_data/weightTime.p", "rb" ) )

predictionsNils = list(np.load('octiba/ga_module/predictions/1542601227_elitism_0.01.npy'))
#predictionsNils = [np.sort(n) for n in predictionsNils]
solutionNils = np.load('octiba/ga_module/y_tests/1542601227_y_test.npy')
#solutionNils = np.sort(solutionNils)

score1em3 = pickle.load( open( "score_data/scores_mutation_1e-3.p", "rb" ) )
score1em4 = pickle.load( open( "score_data/scores_mutation_1e-4.p", "rb" ) )
score1em5 = pickle.load( open( "score_data/scores_mutation_1e-5.p", "rb" ) )
score1em6 = pickle.load( open( "score_data/scores_mutation_1e-6.p", "rb" ) )

scoreVelOff = pickle.load( open( "score_data/scores_vel_off.p", "rb" ) )
scoreVelOn = pickle.load( open( "score_data/scores_vel_on.p", "rb" ) )

scoreHigh = pickle.load( open( "score_data/scores_mut_rate_1e0.p", "rb" ) )
scoreLow = pickle.load( open( "score_data/scores_mut_rate_2e-2.p", "rb" ) )

scoreAdjOn = pickle.load( open( "score_data/scores_adjustable_on.p", "rb" ) )
scoreAdjOff = pickle.load( open( "score_data/scores_adjustable_off.p", "rb" ) )

scoreComb = pickle.load( open( "score_data/scores_vel_adjust.p", "rb" ) )

scoreAll = pickle.load( open( "score_data/scores_all_feat_no_hid.p", "rb" ) )
scoreAllHid = pickle.load( open( "score_data/scores_all_feat_3_hid_2.p", "rb" ) )
scoreSome = pickle.load( open( "score_data/scores_some_feat_3_hid.p", "rb" ) )

# Animation.
ani = g.Animation()

state = 0
while True:
	#print(ani.getRight())
	if ani.getRight():
		state = min(state+1,7)
	if ani.getLeft():
		state = max(state-1,0)

	# Render mutation.
	if state == 0:
		scores = [score1em3,score1em4,score1em5,score1em6]
		ani.renderGraphs(scores,f=3,pace=2,names=['mutation 1e-3','mutation 1e-4','mutation 1e-5','mutation 1e-6'])

	# Render mutation2.
	if state == 1:
		scores = [scoreLow,scoreHigh]
		ani.renderGraphs(scores,f=10,pace=10,names=['low mutation','high mutation'])
	
	# Adjustable mutation
	if state == 2:
		scores = [scoreAdjOn,scoreAdjOff]
		ani.renderGraphs(scores,f=3,pace=15,names=['adjustable mutation','static mutation'])
	
	# Render velocities.
	if state == 3:
		scores = [scoreVelOff,scoreVelOn]
		ani.renderGraphs(scores,minV=0,f=3,pace=5,names=['no acceleration','acceleration'])

	# Mutation vs mix.
	if state == 4:
		scores = [scoreAdjOn,scoreComb]
		ani.renderGraphs(scores,minV=0,f=3,pace=10,names=['adjustable mutation','combined'])

	# All vs some features.
	if state == 5:
		scores = [scoreAll,scoreSome,scoreAllHid]
		ani.renderGraphs(scores,minV=0,f=3,pace=50,names=['all features and no hidden layers','some features and hidden layers','all features and hidden layers'])

	# Nils test.
	# for i in range(1):
	# 	ani.prediction(predictionsNils,solutionNils)

	# Prediction ann.
	if state == 6:
		pace = np.array([5,5,10,10,10,10,10,10,10,10,10,10,1])
		pace = 20
		ani.prediction(prediction1[0:-4000],solution1,pace=pace,title='The fitting process')
	#ani.prediction(prediction2[0:-4000],solution2,pace=pace)
	
	# Weights adjustement.
	if state == 7:
		pace = 1
		ani.animateData(weightTime[0:-500,:],weightTime[-1],(255,0,0),(0,255,0),pace=pace,title='Adjustment of weights')









