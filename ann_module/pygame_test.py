
import pygame, sys
import time

from pygame.locals import *

pygame.init()
windowSurface = pygame.display.set_mode((500, 400), 0, 32)
windowSurface.fill((0,0,0))


fps = 30
t = -1
while True:
	t+=1

	windowSurface.fill((0,0,0))
	pygame.draw.circle(windowSurface, (255,1,1), (int(t), int(t)), 20, 0)
	pygame.display.update()
	
	time.sleep(1/fps)

	# Quit?
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			sys.exit()













