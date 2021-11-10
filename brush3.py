import os
import sys

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


if len(sys.argv)<2:
	print('Pass an image file as an argument!')
	sys.exit()
F_NAME = sys.argv[1]
if not os.path.exists(F_NAME):
	print('File not exists')
	sys.exit()

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
HALF_TILE_SIDE = TILE_SIDE//2
NN_SIDE = int(round(128//16)*16)
HALF_NN_SIDE = NN_SIDE//2
SCALE_FACTOR = NN_SIDE/TILE_SIDE


#breakpoint()
img_orig_resize = cv.resize(img_orig, (0,0) , fx=SCALE_FACTOR, fy=SCALE_FACTOR)

global_seq = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.Flipud(0.5),
		iaa.Rot90((0, 3)),
	    iaa.LinearContrast((0.9, 1.1)),
	    iaa.GaussianBlur(sigma=(0, 1)),
	    iaa.OneOf(  [iaa.Multiply((0.9, 1.1), per_channel=True), iaa.Add((-10, 10), per_channel=True)]),
		], random_order=True)


def batchAug(samples, labels):
	samples2, labels2 = global_seq(images=samples, segmentation_maps=labels)
	return np.array(samples2).astype(np.uint8), (np.array(labels2)/255.0).astype(np.float32)




#masked loss

def masked_bce_2ch(y_true, y_pred):
	#sample , y , x , channel
	mask = y_true[:,:,:,1:2]
	y_tr = y_true[:,:,:,0:1]
	y_tr_mult = y_tr*mask
	y_pr_mult = y_pred*mask
	loss_func = tf.keras.losses.BinaryCrossentropy(from_logits = False)
	val  = loss_func(y_tr_mult, y_pr_mult)
	return val

##################





def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)



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
	
	

def render(pointer_coords):
	global img_orig, screen, mask, BRUSH_TYPE_CURRENT, model_global,needRedraw,HALF_NN_SIDE,img_orig_resize, SCALE_FACTOR
	
	screen = img_orig.copy()
	screen[mask_positive == 255] = (0,255,0)
	screen[mask_negative == 255] = (0,0,255)

	if pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and (BRUSH_TYPE_CURRENT == 'POSITIVE' or BRUSH_TYPE_CURRENT == 'NEGATIVE'):
		cv.circle(screen, pointer_coords, BRUSH_SIZE, BRUSH_COLOR, -1)
	elif pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and BRUSH_TYPE_CURRENT == 'NEURAL':
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

	cv.imshow('img', screen)



def mouseCb(event, x, y,flags, params):
	global mask_positive, mask_negative, BRUSH_TYPE_CURRENT,needRedraw, needRecount, img_orig


	if (BRUSH_TYPE_CURRENT == 'NEURAL' and event==cv.EVENT_LBUTTONUP):
		t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2,:]
		t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))
		t = t.reshape(1,NN_SIDE,NN_SIDE,N_CH)
		r = model_global.predict(t)
		r=(r*255).astype(np.uint8)[0]
		r = cv.resize(r, (TILE_SIDE, TILE_SIDE))
		a,target = cv.threshold(r,128, 255,cv.THRESH_BINARY)
		mask_positive[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2] = target
		
		print('draw!')
		


	if (event == cv.EVENT_MOUSEMOVE or event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONUP) and (BRUSH_TYPE_CURRENT == 'POSITIVE' or BRUSH_TYPE_CURRENT=='NEGATIVE'):
		
		if BRUSH_TYPE_CURRENT == 'POSITIVE':
			target_mask = mask_positive
		elif BRUSH_TYPE_CURRENT == 'NEGATIVE':
			target_mask = mask_negative
		if flags==1:
			target_mask = cv.circle(target_mask, (x,y), BRUSH_SIZE, 255, -1)
		elif flags==2:
			target_mask = cv.circle(target_mask, (x,y), BRUSH_SIZE, 0, -1)
	
	if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
		needRecount = True

		
	render((x,y))


def parseKey(key):
	global BRUSH_SIZE, runTrainThread, BRUSH_TYPE_CURRENT
	if key==ord('+'):
		BRUSH_SIZE+=1
	elif key==ord('-'):
		BRUSH_SIZE-=1
		if BRUSH_SIZE <=0:
			BRUSH_SIZE=1
	elif key in BRUSH_TYPES:
		print('have brush')
		print(BRUSH_TYPES[key])
		BRUSH_TYPE_CURRENT = BRUSH_TYPES[key]
	elif key == ord('t'):
		print('train single shot')
		pixelRecount()
		trainSingleShot()
		print('train done')
	elif key == ord('e'):
		print('Export set!')
		pixelRecount()
		print('exports done!')





	else:
		print('key pressed:', key)



def pixelRecount():
	global mask_negative, mask_positive, pixel_list
	mask_total = cv.bitwise_or(mask_positive, mask_negative)
	pixel_list = np.where(mask_total == 255)



def checkCoords(y,x,img_shape):
	global HALF_TILE_SIDE
	res = True
	if x < HALF_TILE_SIDE or y < HALF_TILE_SIDE:
		res = False
	if x>img_shape[1]-HALF_TILE_SIDE or y>img_shape[0]-HALF_TILE_SIDE:
		res = False
	return res

def getBatchFromIndex(y_arr,x_arr):
	global img_orig, mask_positive, TILE_SIDE, NN_SIDE, mask_negative, HALF_TILE_SIDE, HALF_NN_SIDE
	assert len(x_arr)==len(y_arr)
	tiles  = []
	labels = []
	for i in range(len(x_arr)):
		y,x = y_arr[i], x_arr[i]

		if checkCoords(y,x,img_orig.shape):
			t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
			l = mask_positive[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2]
			m_n = mask_negative[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2]

			t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))
			l = cv.resize(l, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, 1))
			m_n = cv.resize(m_n, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, 1))

			m_sum = np.bitwise_or(l, m_n)
			
			assert m_sum.shape == (NN_SIDE, NN_SIDE, 1)
			l_concat = np.zeros((NN_SIDE,NN_SIDE,2), dtype=np.uint8)
			l_concat[:,:,0:1] = l
			l_concat[:,:,1:2] = m_sum
			tiles.append(t)
			labels.append(l_concat)

	#tiles_aug, labels_aug = batchAug(tiles, labels)
	#tiles = np.array(tiles).astype(np.uint8)
	#labels = (np.array(labels)/255).astype(np.float32)
	tiles_aug, labels_aug = batchAug(tiles, labels)
	return (tiles_aug, labels_aug)


def getBatchGenerator():
	global pixel_list,BATCH_SIZE
	y,x = pixel_list
	y = y[::USE_EVERY_NTH]
	x = x[::USE_EVERY_NTH]

	shuffle_in_unison_scary(y,x)
	assert len(y)==len(x)
	b_count = len(y)//BATCH_SIZE
	y = y[0:b_count*BATCH_SIZE]
	x = x[0:b_count*BATCH_SIZE]
	assert len(y)==len(x)
	y = y.reshape(b_count, BATCH_SIZE)
	x = x.reshape(b_count, BATCH_SIZE)
	for b_index in range(b_count):
		b = getBatchFromIndex(y[b_index], x[b_index])
		yield b



def trainSingleShot():
	global model_global, BATCH_SIZE, pixel_list
	g = getBatchGenerator()
	b_count = len(pixel_list[0][::USE_EVERY_NTH])//BATCH_SIZE
	print(b_count)
	model_global.fit(x=g, steps_per_epoch=b_count, workers = 1)











print('set up keras model')
model_unet = models.unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=1, output_activation='Sigmoid', batch_norm=False)

#model_unet = U_Net((NN_SIDE,NN_SIDE, N_CH))

model_global = tf.keras.Sequential()
model_global.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
model_global.add(tf.keras.layers.Lambda(lambda x: x/255))
model_global.add(model_unet)
model_global.compile(loss=masked_bce_2ch, optimizer='adam', metrics = 'accuracy')



model_inference = tf.keras.Sequential()
model_inference.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
model_inference.add(model_global)
model_inference.add(tf.keras.layers.Lambda(lambda x: x*255))
model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(TILE_SIDE, TILE_SIDE, interpolation = 'bilinear'))

print('get generator')



print('init window')
cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)
key = 0
print('start main loop')
while key != 27:
	render((-1,-1))
	key = cv.waitKey(0)
	parseKey(key)
cv.destroyAllWindows()


