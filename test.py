import os
import sys
import line_profiler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import cv2 as cv
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l1, l2, L1L2

from keras_unet_collection import models
from keras_unet_collection import losses
import imgaug.augmenters as iaa
import imgaug as ia





USE_EVERY_NTH = 2

BRUSH_SIZE = 12
BRUSH_TYPES = {49:'POSITIVE', 50:'NEGATIVE', 51: 'NEURAL'}
BRUSH_TYPE_CURRENT = 'POSITIVE'
BATCH_SIZE = 16
N_CH = 3
BRUSH_COLOR = (250,250,250)

F_NAME = "1_small.png"


print(F_NAME)
img_orig  = cv.imread(F_NAME, -1)
screen = img_orig.copy()


mask_positive = np.zeros((screen.shape[0],screen.shape[1]), dtype=np.uint8)
mask_negative = np.zeros((screen.shape[0],screen.shape[1]), dtype=np.uint8)

tiles_batches = []
labels_batches = []

pixel_list = [[],[]]
needRedraw = True
needRecount = False
model_global = None

TILE_SIDE = 128
NN_SIDE = 64
HALF_NN_SIDE = NN_SIDE//2
HALF_TILE_SIDE = TILE_SIDE//2
SCALE_FACTOR = NN_SIDE/TILE_SIDE


img_orig_resize = cv.resize(img_orig, (0,0) , fx=SCALE_FACTOR, fy=SCALE_FACTOR)



print('set up keras model')
model_unet = models.unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=1, output_activation='Sigmoid', batch_norm=False)
#model_unet = U_Net((NN_SIDE,NN_SIDE, N_CH))

model_global = tf.keras.Sequential()
model_global.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
model_global.add(tf.keras.layers.Lambda(lambda x: x/255))
model_global.add(model_unet)

model_global.compile(loss='bce', optimizer='adam', metrics = 'accuracy')

model_inference = tf.keras.Sequential()
model_inference.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
model_inference.add(model_global)
model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(TILE_SIDE, TILE_SIDE, interpolation = 'bilinear'))
model_inference.add(tf.keras.layers.Lambda(lambda x: x*255))


def conditionCoords(y,x, shape, h_side=HALF_TILE_SIDE):
	if x < h_side:
		x = h_side
	if y<h_side:
		y = h_side
	if x > shape[1]-h_side:
		x=shape[1]-h_side
	if y > shape[0]-h_side:
		y=shape[0]-h_side
	return (y,x)
	



@profile
def render_CPU(pointer_coords = (1,1)):
	global img_orig, screen, mask, BRUSH_TYPE_CURRENT, model_global,needRedraw,HALF_SIDE, img_orig_resize, SCALE_FACTOR
	screen = img_orig.copy()
	x=pointer_coords[0]
	y=pointer_coords[1]
	y,x = conditionCoords(y,x, img_orig.shape)	
	y_resized = round(y*SCALE_FACTOR)
	x_resized = round(x*SCALE_FACTOR)
	t = img_orig_resize[y_resized-HALF_NN_SIDE:y_resized+HALF_NN_SIDE, x_resized-HALF_NN_SIDE:x_resized+HALF_NN_SIDE, :]
	t = t.reshape(1,*t.shape)
	with tf.device('/cpu:0'):
		r = model_inference(t, training=False)
	r = tf.cast(r, tf.uint8)
	th = r.numpy()
	target = np.broadcast_to(th[0], (TILE_SIDE, TILE_SIDE, 3))
	screen[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :] = target


@profile
def render_GPU(pointer_coords = (1,1)):
	global img_orig, screen, mask, BRUSH_TYPE_CURRENT, model_global,needRedraw,HALF_SIDE, img_orig_resize, SCALE_FACTOR
	screen = img_orig.copy()
	x=pointer_coords[0]
	y=pointer_coords[1]
	y,x = conditionCoords(y,x, img_orig.shape)	
	y_resized = round(y*SCALE_FACTOR)
	x_resized = round(x*SCALE_FACTOR)
	t = img_orig_resize[y_resized-HALF_NN_SIDE:y_resized+HALF_NN_SIDE, x_resized-HALF_NN_SIDE:x_resized+HALF_NN_SIDE, :]
	t = t.reshape(1,*t.shape)
	#with tf.device('/cpu:0'):
	r = model_inference(t, training=False)
	r = tf.cast(r, tf.uint8)
	th = r.numpy()
	target = np.broadcast_to(th[0], (TILE_SIDE, TILE_SIDE, 3))
	screen[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :] = target





for i in range(10):
	render_CPU()
	render_GPU()