import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

import pickle

import pygame, sys
from pygame.locals import *

import time

# Load data.

class Animation():
	def __init__(self):
		self.width = 1000
		self.height = 500
		# Init graphics.
		pygame.init()
		self.windowSurface = pygame.display.set_mode((int(self.width), int(self.height)), 0, 32)
		self.windowSurface.fill((0,0,0))
		self.debugger = 0
	def getLeft2(self):
		return pygame.key.get_pressed()
	def getLeft(self):
		return pygame.key.get_pressed()[276] == 1
	def getRight(self):
		return pygame.key.get_pressed()[275] == 1
	def renderGraph(self,d,maxV=None,minV=None,color=(0,255,0)):
		if maxV is None:
			maxV = np.max(d)
		if minV is None:
			minV = np.min(d)
		for i in range(1,len(d)):
			pygame.draw.line(self.windowSurface, color,
				(self.width/len(d)*i, self.height-(d[i]-minV)/(maxV-minV)*self.height),
				(self.width/len(d)*(i-1), self.height-(d[i-1]-minV)/(maxV-minV)*self.height),
			4)
	def prediction(self, prediction, solution,pace=1,title=None):
		# Check input.
		if prediction == None:
			raise ValueError('Missing parameter \'prediction\'.')
		if solution == None:
			raise ValueError('Missing parameter \'solution\'.')
		if type(prediction) is not list:
			raise ValueError('The parameter \'prediction\' is of type '+str(type(prediction))+' but the expected type is list.')
		if type(solution) is not np.ndarray:
			raise ValueError('The parameter \'solution\' is of type '+str(type(solution))+' but the expected type is numpy.ndarray.')
		for p in prediction:
			if type(p) is not np.ndarray:
				raise ValueError('The parameter \'prediction\' has element(s) of type '+str(type(p))+' but \'prediction\' should only have elements of type numpy.ndarray')
			if solution.shape[0] != p.shape[0]:
				raise ValueError('The length of the solution should be equal to all the lengths of elements of the prediction list.')
		
		#
		indices = np.argsort(solution)
		prediction = prediction.copy()
		for i in range(len(prediction)):
			prediction[i] = prediction[i][indices]
		solution = solution[indices]

		# Run graphics.
		maxV = np.max(solution)
		minV = np.min(solution)-100000
		fps = 60
		t = 0
		while True:
			# Clear.
			self.windowSurface.fill((0,0,0))
			# Render
			## Render text.
			if title is not None:
				font = pygame.font.SysFont('Consolas', self.height//10)
				text = font.render(title, True, (0,0,255))
				self.windowSurface.blit(text,(
						self.width // 2 - text.get_width()//2,
						self.height // (3)- text.get_height()//2
					))
			## Render solution.
			self.renderGraph(solution,minV=minV,maxV=maxV)
			## Render prediction.
			#for i in range(len(prediction)):
			h = prediction[t]
			for i in range(1,len(h)):
				pygame.draw.circle(self.windowSurface, (255,0,0),
					(int(self.width/len(h)*i), self.height-int((h[i]-minV)/(maxV-minV)*self.height)),
				int(1))
			# Update.
			pygame.display.update()
			# Sleep.
			time.sleep(1/fps)
			# Quit?
			if t+pace >= len(prediction):
				return
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()
			# Counter.
			t+=pace
	def animateData(self,data,solution=None,color1=(255,0,0),color2=(0,255,0),pace=1,title=None):
		# Check input.
		if data == None:
			raise ValueError('Missing parameter \'data\'.')
		for d in data:
			if type(d) is not np.ndarray:
				raise ValueError('The parameter \'data\' has element(s) of type '+str(type(d))+' but \'d\' should only have elements of type numpy.ndarray.')
		# Run graphics.
		maxV = np.max(data)
		minV = np.min(data)
		fps = 60
		t = 0
		while True:

			# Clear.
			self.windowSurface.fill((0,0,0))
			# Render
			## Render text.
			if title is not None:
				font = pygame.font.SysFont('Consolas', self.height//10)
				text = font.render(title, True, (0,0,255))
				self.windowSurface.blit(text,(
						self.width // 2 - text.get_width()//2,
						self.height // (3)- text.get_height()//2
					))
			## Render solution.
			if solution is not None:
				self.renderGraph(solution,maxV,minV,color2)
			self.renderGraph(data[t],maxV,minV,color1)
			# Update.
			pygame.display.update()
			# Sleep.
			time.sleep(1/fps)
			# Quit?
			if t+pace >= data.shape[0]:
				return
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()
			# Counter.
			t+=pace
	def renderGraphs(self,graphs, colors=None,maxV=None,minV=None,f=1,pace=1,names=None):
		# Multiple runs.
		if f > 1:
			m = max([len(g) for g in graphs])
			for i in range(f):
				self.renderGraphs([g[int(i*len(g)/f):int((i+1)*len(g)/f)] for g in graphs],minV=0,colors=colors,pace=pace,names=names)
			return
		# Check input.
		if graphs == None:
			raise ValueError('Missing parameter \'graphs\'.')
		if type(graphs) is not list:
			raise ValueError('The parameter \'graphs\' is of type '+str(type(graphs))+' but the expected type is list.')
		for g in graphs:
			if type(g) is not list:
				raise ValueError('The parameter \'data\' has element(s) of type '+str(type(g))+' but \'d\' should only have elements of type numpy.ndarray.')
		# Set colors.
		if colors is None:
			colors = []
			for i in range(len(graphs)):
				colors.append((
					int(255/2 + 255/2*np.sin(i/len(graphs)*np.pi*2+0)),
					int(255/2 + 255/2*np.sin(i/len(graphs)*np.pi*2+np.pi*2/3)),
					int(255/2 + 255/2*np.sin(i/len(graphs)*np.pi*2+np.pi*4/3))
				))
		# Run graphics.
		m = max([(max(g) if len(g) != 0 else -np.Infinity) for g in graphs])
		if maxV is None:
			maxV = m
		else:
			maxV = max(m,maxV)
		m = min([(min(g) if len(g) != 0 else np.Infinity) for g in graphs])
		if minV is None:
			minV = m
		else:
			minV = min(m,minV)
		fps = 60
		t = 1
		maxT = max([len(g) for g in graphs])-1
		# Clear.
		self.windowSurface.fill((0,0,0))
		# Create text.
		if names is not None:
			font = pygame.font.SysFont('Consolas', self.height//10)
			for i in range(len(graphs)):
				text = font.render(names[i], True, colors[i])
				self.windowSurface.blit(text,(
					self.width // 2 - text.get_width()//2,
					self.height // (len(graphs)+1)*(i+1) - text.get_height()//2
				))
		while True:
			# Render
			
			# try:
			# 	print(list(pygame.key.get_pressed()).index(1))
			# except:
			# 	pass

			for i in range(pace):
				## Render solution.
				for j in range(len(graphs)):
					g = graphs[j]
					# if t >= len(g):
					# 	continue
					ind = round(t/maxT*(len(g)-1))
					if ind == 0:
						ind = 1
					self.debugger += 1
					if self.debugger == 469:
						a=999
					pygame.draw.line(self.windowSurface, colors[j],
						(self.width/len(g)*(ind-1), self.height-(g[ind-1]-minV)/(maxV-minV)*self.height),
						(self.width/len(g)*ind, self.height-(g[ind]-minV)/(maxV-minV)*self.height),
					4)
				# Quit?
				if t >= maxT:
					return
				# Counter.
				t+=1
			# Update.
			pygame.display.update()
			# Sleep.
			time.sleep(1/fps)
			#
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()

