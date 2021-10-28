import cv2 as cv
import numpy as np
#import tensorflow as tf

img_orig  = cv.imread('1.png', -1)

screen = img_orig.copy()
mask = np.zeros((screen.shape[0],screen.shape[1], screen.shape[2]), dtype=np.uint8)




def render():
	global img_orig, screen, mask

	screen = cv.addWeighted(img_orig, 1, mask,1, 1) 
	#screen = img_orig.copy()
	#screen[np.where(mask==255)]=(255,255,255)
	cv.imshow('img', screen)


def mouseCb(event, x, y,flags, params):
	global img_orig,screen, mask
	

	if event == cv.EVENT_MOUSEMOVE and flags==1:
		#print(flags)
		#print(params)
		mask = cv.circle(mask, (x,y), 10, (255,255,255), -1)
		render()
	elif event == cv.EVENT_MOUSEMOVE and flags==2:
		#print(flags)
		#print(params)
		mask = cv.circle(mask, (x,y), 10, (0,0,0), -1)
		render()




cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)


key = 0
while key != 27:
	render()
	key = cv.waitKey(0)