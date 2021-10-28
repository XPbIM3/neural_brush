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
runPixelPoolThread = True
needRecount = True
model = None

TILE_SIDE = 64
NN_SIDE = 32


def getBatchGenerator():
	global tilePool
	while True:
		out_tiles = []
		out_labels = []
		indexes = np.random.randint(0,len(tilePool), 50)
		for i in indexes:
			out_tiles.append(np.array(tilePool[i][0]))
			out_labels.append(np.array(tilePool[i][1]))
		yield (np.array(out_tiles), np.array(out_labels))


def prepareTiles():
	global mask, pixelPool, img_orig, TILE_SIDE, tilePool, runPixelPoolThread, labelPool, needRecount
	y_shape = img_orig.shape[0]
	x_shape = img_orig.shape[1]
	for i in range(1000):
		x = np.random.randint(0,x_shape)
		y = np.random.randint(0,y_shape)
		t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2]
		l = mask[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, 0]
		if t.shape == (TILE_SIDE,TILE_SIDE, N_CH):
			t = (cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))).astype(np.uint8)
			l = (cv.resize(l, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE,1))/255.0).astype(np.float32)
			tilePool.append([t,l])





print('set up a keras')
u_model = models.att_unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=1, output_activation='Sigmoid', batch_norm=False)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE, NN_SIDE, N_CH)))
model.add(tf.keras.layers.Lambda(lambda x: x/255))
model.add(u_model)
model.compile(loss=losses.focal_tversky, optimizer='adam', metrics = 'accuracy')
prepareTiles()
gen = getBatchGenerator()
model.fit(x=gen)




