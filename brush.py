import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import cv2 as cv
import numpy as np
import threading
import tensorflow as tf
from keras_unet_collection import models
from keras_unet_collection import losses
BRUSH_SIZE = 10
BRUSH_COLOR = (250,250,250)

img_orig  = cv.imread('1.png', -1)
N_CH = img_orig.shape[2]
screen = img_orig.copy()
mask = np.zeros((screen.shape[0],screen.shape[1], screen.shape[2]), dtype=np.uint8)
brush = np.zeros((screen.shape[0],screen.shape[1], screen.shape[2]), dtype=np.uint8)


pixelPool = []
tilePool = []
labelPool = []
runPixelPoolThread = True
needRecount = True


TILE_SIDE = 64
NN_SIDE = 32


def render(pointer_coords):
	global img_orig, screen, mask
	screen = cv.bitwise_or(img_orig,mask)
	if pointer_coords[0]!=-1 and pointer_coords[1]!=-1:
		cv.circle(screen, pointer_coords, BRUSH_SIZE, BRUSH_COLOR, -1)
	cv.imshow('img', screen)


def refreshPool():
	global mask, pixelPool, img_orig, TILE_SIDE, tilePool, runPixelPoolThread, labelPool, needRecount
	while runPixelPoolThread:
		if needRecount:
			m = mask[:,:,0].copy()
			old_N = len(pixelPool)
			pixelPool = np.moveaxis(np.array(np.nonzero(m)), 0, -1)
			new_N = len(pixelPool)
			if old_N != new_N:
				tilePool = []
				labelPool = []		
				for coords in pixelPool:
					y,x = coords
					t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
					l = mask[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2]

					if t.shape == (TILE_SIDE,TILE_SIDE, N_CH):
						t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))
						l = cv.resize(l, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))/255.0
						tilePool.append(t)
						labelPool.append(l)
				print('Tile pool size: ', len(tilePool))
			else:
				print('no change in pixel count')


			needRecount = False
	print('pixelPool stopped by global flag!')


def mouseCb(event, x, y,flags, params):
	global img_orig,screen, mask, tilePool, needRecount
	

	if event == cv.EVENT_MOUSEMOVE:
		if flags==1:
			#print(flags)
			#print(params)
			mask = cv.circle(mask, (x,y), BRUSH_SIZE, (255,255,255), -1)
		elif flags==2:
			#print(flags)
			#print(params)
			mask = cv.circle(mask, (x,y), BRUSH_SIZE, (0,0,0), -1)
			
		render((x,y))
	if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
		needRecount = True



def parseKey(key):
	global BRUSH_SIZE
	if key==43:
		BRUSH_SIZE+=1
	elif key==45:
		BRUSH_SIZE-=1



print('set up a keras')
model = tf.keras.models.Sequential()





print('init window')
cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)

print('start a tile thread')
refreshThread = threading.Thread(target = refreshPool)
refreshThread.start()

key = 0
while key != 27:
	render((-1,-1))
	key = cv.waitKey(0)
	parseKey(key)

cv.destroyAllWindows()
runPixelPoolThread = False
refreshThread.join()

