import os
import sys
from typing import Sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


from keras_unet_collection import models
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

mouse_coords = (0,0)

TILE_SIDE = 128
HALF_TILE_SIDE = TILE_SIDE//2
NN_SIDE = 128
NN_SIDE = int(round(NN_SIDE//16)*16)
HALF_NN_SIDE = NN_SIDE//2
SCALE_FACTOR = NN_SIDE/TILE_SIDE


#breakpoint()
global_seq = iaa.Sequential([
		iaa.Affine(translate_percent={'x':(-0.25, 0.25), 'y':(-0.25,0.25)}, mode='wrap'),
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

def masked_bce_2ch(class_weights):
	print('loss function with:', class_weights)
	def masked_bce_inner(y_true, y_pred):
		mask = y_true[:,:,:,0:1]
		y_tr = y_true[:,:,:,1:3]
		bce  = K.binary_crossentropy(y_tr, y_pred)*mask*class_weights
		val = tf.reduce_mean(bce)
		return val
	return(masked_bce_inner)

##################



def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)





class DSet_weightened(tf.keras.utils.Sequence):
	def __init__(self, img, mask_pos, mask_neg, skip=4, batch_size=16, half_side = HALF_TILE_SIDE, nn_side=NN_SIDE) -> None:
		self.mask_neg = mask_neg
		self.mask_pos = mask_pos
		self.batch_size = batch_size
		self.img = img
		if len(img.shape)==3:
			self.n_ch = img.shape[-1]
		else:
			self.n_ch = 1
		self.mask_total = cv.bitwise_or(mask_pos, mask_neg)
		self.pixel_list_all = np.where(self.mask_total == 255)
		y,x = self.pixel_list_all
		y = y[::skip]
		x = x[::skip]
		shuffle_in_unison_scary(y,x)
		assert len(x)==len(y)
		self.y_list=y
		self.x_list=x
		self.class_weights = self.getClassWeights()
		self.half_side = half_side
		self.nn_side = nn_side


	def getClassWeights(self):
		p = np.count_nonzero(self.mask_pos)
		n = np.count_nonzero(self.mask_neg)
		tot = p+n
		return (n/tot, p/tot)

	def checkCoords(self,y,x):
		res = True
		if x < HALF_TILE_SIDE or y < HALF_TILE_SIDE:
			res = False
		if x>self.img.shape[1]-HALF_TILE_SIDE or y>self.img.shape[0]-HALF_TILE_SIDE:
			res = False
		return res

	def getTileByCoords(self,y,x):
		t = self.img[y-self.half_side:y+self.half_side, x-self.half_side:x+self.half_side]
		assert t.shape[0]==self.half_side*2
		assert t.shape[1]==self.half_side*2
		t = cv.resize(t, (NN_SIDE,NN_SIDE)).reshape((NN_SIDE,NN_SIDE,self.n_ch))
		return t
	
	def getPosByCoords(self,y,x):
		t = self.mask_pos[y-self.half_side:y+self.half_side, x-self.half_side:x+self.half_side]
		assert t.shape[0]==self.half_side*2
		assert t.shape[1]==self.half_side*2
		t = cv.resize(t, (NN_SIDE,NN_SIDE)).reshape((NN_SIDE,NN_SIDE,1))
		return t
	
	def getNegByCoords(self,y,x):
		t = self.mask_neg[y-self.half_side:y+self.half_side, x-self.half_side:x+self.half_side]
		assert t.shape[0]==self.half_side*2
		assert t.shape[1]==self.half_side*2
		t = cv.resize(t, (NN_SIDE,NN_SIDE)).reshape((NN_SIDE,NN_SIDE,1))
		return t
	
	def getMaskByCoords(self,y,x):
		t = self.mask_total[y-self.half_side:y+self.half_side, x-self.half_side:x+self.half_side]
		assert t.shape[0]==self.half_side*2
		assert t.shape[1]==self.half_side*2
		t = cv.resize(t, (NN_SIDE,NN_SIDE)).reshape((NN_SIDE,NN_SIDE,1))
		return t


	def __len__(self) -> int:
		count = len(self.x_list)//self.batch_size
		return count
	def getSample(self, sample_idx):
		y = self.y_list[sample_idx]
		x = self.x_list[sample_idx]
		if self.checkCoords(y,x):
			t=self.getTileByCoords(y,x)
			p=self.getPosByCoords(y,x)
			n=self.getNegByCoords(y,x)
			m=self.getMaskByCoords(y,x)
			assert t.shape == (NN_SIDE, NN_SIDE, self.n_ch)
			l = np.concatenate([m, p, n], axis=-1)
			assert l.shape == (NN_SIDE,NN_SIDE,3)
			return t, l
		else:
			rnd_idx = np.random.randint(len(self.x_list))
			t, l = self.getSample(rnd_idx)
			return t,l

	def __getitem__(self, idx):
		samples, labels = [], []
		for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
			s, l = self.getSample(i)
			samples.append(s)
			labels.append(l)
		samples, labels = batchAug(samples,labels)
		return samples,labels





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
	


def modelSave():
	global model_global, model_inference
	model_global.save("model.hdf5", include_optimizer=False)

def modelLoad():
	global model_global, model_inference
	model_global = tf.keras.models.load_model('model.hdf5', compile=False)
	model_inference = tf.keras.Sequential()
	model_inference.add(tf.keras.layers.InputLayer(input_shape = (TILE_SIDE,TILE_SIDE,N_CH)))
	model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(NN_SIDE, NN_SIDE, interpolation = 'bilinear'))
	model_inference.add(model_global)
	model_inference.add(tf.keras.layers.Lambda(lambda x: x*255))
	model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(TILE_SIDE, TILE_SIDE, interpolation = 'bilinear'))


def modelReset():
	global model_global, model_inference
	model_global = tf.keras.Sequential()
	model_global.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
	model_global.add(tf.keras.layers.Lambda(lambda x: x/255))
	model_global.add(model_unet)
	model_inference = tf.keras.Sequential()
	model_inference.add(tf.keras.layers.InputLayer(input_shape = (TILE_SIDE,TILE_SIDE,N_CH)))
	model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(NN_SIDE, NN_SIDE, interpolation = 'bilinear'))
	model_inference.add(model_global)
	model_inference.add(tf.keras.layers.Lambda(lambda x: x*255))
	model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(TILE_SIDE, TILE_SIDE, interpolation = 'bilinear'))

def render(pointer_coords=None):
	global img_orig, screen, mask, BRUSH_TYPE_CURRENT, model_global,needRedraw,HALF_NN_SIDE,img_orig_resize, SCALE_FACTOR

	if pointer_coords == None:
		pointer_coords = mouse_coords
	
	screen = img_orig.copy()
	screen[mask_positive >= 128] = (0,255,0)
	screen[mask_negative >= 128] = (0,0,255)

	if pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and (BRUSH_TYPE_CURRENT == 'POSITIVE' or BRUSH_TYPE_CURRENT == 'NEGATIVE'):
		cv.circle(screen, pointer_coords, BRUSH_SIZE, BRUSH_COLOR, -1)
	elif pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and BRUSH_TYPE_CURRENT == 'NEURAL':
		x=pointer_coords[0]
		y=pointer_coords[1]
		y,x = conditionCoords(y,x, img_orig.shape)
		
		t = img_orig[y-HALF_TILE_SIDE:y+HALF_TILE_SIDE, x-HALF_TILE_SIDE:x+HALF_TILE_SIDE,:]
		target = predict(t)
		target_0phase = target[:,:,0].copy()
		
		'''
		target_1[target_1>=128]=255
		target_1[target_1<128]=0
		target_not = 255-target_1
		target_flood = cv.floodFill(target_not.copy(), None, (HALF_TILE_SIDE,HALF_TILE_SIDE), 255)[1]
		target = cv.bitwise_and(target_1, target_flood)
		circle_tile = np.zeros((TILE_SIDE, TILE_SIDE), dtype=np.uint8)
		cv.circle(circle_tile,(HALF_TILE_SIDE,HALF_TILE_SIDE),HALF_TILE_SIDE, 255, -1)
		target = cv.bitwise_and(target, circle_tile).reshape(TILE_SIDE, TILE_SIDE, 1)

		'''
		brush = brushFromPredict(target_0phase)

		target = np.broadcast_to(brush, (TILE_SIDE,TILE_SIDE, 3))
		screen_tile = screen[y-HALF_TILE_SIDE:y+HALF_TILE_SIDE, x-HALF_TILE_SIDE:x+HALF_TILE_SIDE, :]
		screen_tile[target==255] = target[target==255]

		screen[y-HALF_TILE_SIDE:y+HALF_TILE_SIDE, x-HALF_TILE_SIDE:x+HALF_TILE_SIDE, :] = screen_tile
		#screen[y-HALF_TILE_SIDE:y+HALF_TILE_SIDE, x-HALF_TILE_SIDE:x+HALF_TILE_SIDE, 2:3] = target[:,:,1:2]

	cv.imshow('img', screen)


def brushFromPredict(predictedTile):
	assert predictedTile.shape == (TILE_SIDE, TILE_SIDE)
	t = predictedTile.copy()
	t[t>=128]=255
	t[t<128]=0
	target_not = 255-t
	target_flood = cv.floodFill(target_not.copy(), None, (HALF_TILE_SIDE,HALF_TILE_SIDE), 255)[1]
	target = cv.bitwise_and(t, target_flood)
	circle_tile = np.zeros((TILE_SIDE, TILE_SIDE), dtype=np.uint8)
	cv.circle(circle_tile,(HALF_TILE_SIDE,HALF_TILE_SIDE),HALF_TILE_SIDE, 255, -1)
	target = cv.bitwise_and(target, circle_tile).reshape(TILE_SIDE, TILE_SIDE, 1)
	return target




	


def predict(t, channels=1):
	global model_inference, model_global
	t = t.reshape(1,*t.shape)
	r = model_inference(t, training=False)
	r = tf.cast(r, tf.uint8)
	target = r.numpy()[0]

	if channels>1:
		target = np.broadcast_to(target, (TILE_SIDE, TILE_SIDE, channels))
	return target



def mouseCb(event, x, y,flags, params):
	global mask_positive, mask_negative, BRUSH_TYPE_CURRENT,needRedraw, needRecount, img_orig, mouse_coords

	mouse_coords = (x,y)

	if (BRUSH_TYPE_CURRENT == 'NEURAL' and event==cv.EVENT_LBUTTONUP):
		t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2,:]
		target = predict(t)
		target_0phase = target[:,:,0].copy()
		brush = brushFromPredict(target_0phase).reshape(TILE_SIDE,TILE_SIDE)
		mask_tile = mask_positive[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2]
		mask_tile[brush==255]=brush[brush==255]
		mask_positive[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2] = mask_tile

	


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
		render()
	elif key==ord('-'):
		BRUSH_SIZE-=1
		render()
		if BRUSH_SIZE <=0:
			BRUSH_SIZE=1
	elif key in BRUSH_TYPES:
		print('have brush')
		print(BRUSH_TYPES[key])
		BRUSH_TYPE_CURRENT = BRUSH_TYPES[key]
	elif key == ord('t'):
		print('train single shot')
		trainSingleShot()
		print('train done')
	elif key == ord('y'):
		print('train single shot w/o weights')
		trainSingleShotNoWeights()
		print('train done')
	elif key==ord('s'):
		modelSave()
		print('saved!')
	elif key==ord('l'):
		modelLoad()
		print('loaded!')
	elif key==ord('r'):
		modelReset()
		print('Reset!!')





	else:
		print('key pressed:', key)






def trainSingleShot():
	global model_global, BATCH_SIZE, mask_negative, mask_positive, model_inference
	g = DSet_weightened(img = img_orig, mask_pos = mask_positive, mask_neg = mask_negative, skip=2, batch_size=BATCH_SIZE)
	class_weights = g.class_weights
	print(class_weights)
	loss_func = masked_bce_2ch(class_weights)
	model_global.compile(loss=loss_func, optimizer='adam', metrics = 'accuracy')
	model_global.fit(x=g, workers = 8)





def trainSingleShotNoWeights():
	global model_global, BATCH_SIZE, mask_negative, mask_positive, model_inference
	g = DSet_weightened(img = img_orig, mask_pos = mask_positive, mask_neg = mask_negative, skip=16, batch_size=BATCH_SIZE)
	class_weights = (1,1)
	print(class_weights)
	loss_func = masked_bce_2ch(class_weights)
	model_global.compile(loss=loss_func, optimizer='adam', metrics = 'accuracy')
	model_global.fit(x=g, workers = 8)





print('set up keras model')
model_unet = models.unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=2, output_activation='Softmax', batch_norm=False)

#model_unet = U_Net((NN_SIDE,NN_SIDE, N_CH))

model_global = tf.keras.Sequential()
model_global.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
model_global.add(tf.keras.layers.Lambda(lambda x: x/255))
model_global.add(model_unet)


model_inference = tf.keras.Sequential()
model_inference.add(tf.keras.layers.InputLayer(input_shape = (TILE_SIDE,TILE_SIDE,N_CH)))
model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(NN_SIDE, NN_SIDE, interpolation = 'bilinear'))
model_inference.add(model_global)
model_inference.add(tf.keras.layers.Lambda(lambda x: x*255))
model_inference.add(tf.keras.layers.experimental.preprocessing.Resizing(TILE_SIDE, TILE_SIDE, interpolation = 'bilinear'))



print('init window')
cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)
key = 0
print('start main loop')
while key != 27:
	render()
	key = cv.waitKey(0)
	parseKey(key)
cv.destroyAllWindows()


