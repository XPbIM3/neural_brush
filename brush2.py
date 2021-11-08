import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import cv2 as cv
import numpy as np
import tensorflow as tf
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


img_orig  = cv.imread('blood.png', -1)
screen = img_orig.copy()

mask_positive = np.zeros((screen.shape[0],screen.shape[1]), dtype=np.uint8)
mask_negative = np.zeros((screen.shape[0],screen.shape[1]), dtype=np.uint8)

tiles_batches = []
labels_batches = []

pixel_list = [[],[]]
needRedraw = True
needRecount = False
model_global = None

TILE_SIDE = 64
NN_SIDE = 32



global_seq = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.Flipud(0.5),
		iaa.Rot90((0, 3)),
	    iaa.LinearContrast((0.9, 1.1)),
	    iaa.GaussianBlur(sigma=(0, 2)),
	    iaa.OneOf(  [iaa.Multiply((0.9, 1.1), per_channel=True), iaa.Add((-10, 10), per_channel=True)]),
	    iaa.Sometimes(0.9, iaa.PerspectiveTransform(scale=(0.01, 0.15)))
		], random_order=True)


def batchAug(samples, labels):
	samples2, labels2 = global_seq(images=samples, segmentation_maps=labels)
	return np.array(samples2), np.array(labels2)



def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)



def render(pointer_coords):
	global img_orig, screen, mask, BRUSH_TYPE_CURRENT, model_global,needRedraw
	
	screen = img_orig.copy()
	screen[mask_positive == 255] = (0,255,0)
	screen[mask_negative == 255] = (0,0,255)

	if pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and (BRUSH_TYPE_CURRENT == 'POSITIVE' or BRUSH_TYPE_CURRENT == 'NEGATIVE'):
		cv.circle(screen, pointer_coords, BRUSH_SIZE, BRUSH_COLOR, -1)
	elif pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and BRUSH_TYPE_CURRENT == 'NEURAL':
		x=pointer_coords[0]
		y=pointer_coords[1]
		t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
		t_orig = t.copy()
		t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))
		t = t.reshape(1,NN_SIDE,NN_SIDE,N_CH)
		r = model_global.predict(t)
		r=(r*255).astype(np.uint8)[0]
		a,th = cv.threshold(r,128, 255,cv.THRESH_BINARY)
		target = cv.resize(th, (TILE_SIDE, TILE_SIDE))
		#target_sc = screen[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
		px_lst = np.where(target>0)
		t_orig[px_lst]=[255,255,255]
		screen[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :] = t_orig
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
		exportSet()
		print('exports done!')





	else:
		print('key pressed:', key)



def pixelRecount():
	global mask_negative, mask_positive, pixel_list
	mask_total = cv.bitwise_or(mask_positive, mask_negative)
	pixel_list = np.where(mask_total == 255)



def getBatchFromIndex(y_arr,x_arr):
	global img_orig, mask_positive, TILE_SIDE, NN_SIDE
	assert len(x_arr)==len(y_arr)
	tiles  = []
	labels = []
	for i in range(len(x_arr)):
		y,x = y_arr[i], x_arr[i]
		t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
		l = mask_positive[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2]
		t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))
		l = cv.resize(l, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, 1))
		tiles.append(t)
		labels.append(l)

	tiles_aug, labels_aug = batchAug(tiles, labels)
	return (tiles_aug, (labels_aug/255.0).astype(np.float32))


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


def exportSet():
	global USE_EVERY_NTH, BATCH_SIZE, pixel_list
	g = getBatchGenerator()
	b_count = len(pixel_list[0][::USE_EVERY_NTH])//BATCH_SIZE
	for i in range(b_count):
		t_batch, l_batch = g.__next__()
		tile_res = (np.vstack(t_batch*255).astype(np.uint8))
		label_res = (np.vstack(l_batch*255).astype(np.uint8))
		cv.imwrite('res/res_tiles_'+str(i)+'.jpg', tile_res)
		cv.imwrite('res/res_tiles_'+str(i)+'_l.jpg', label_res)




def trainSingleShot():
	global model_global, BATCH_SIZE, pixel_list
	g = getBatchGenerator()
	b_count = len(pixel_list[0][::USE_EVERY_NTH])//BATCH_SIZE
	print(b_count)
	model_global.fit(x=g, steps_per_epoch=b_count, workers = 1)










print('set up keras model')
model_unet = models.unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=1, output_activation='Sigmoid', batch_norm=False)

model_global = tf.keras.Sequential()
model_global.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE,NN_SIDE,N_CH)))
model_global.add(tf.keras.layers.Lambda(lambda x: x/255))
model_global.add(model_unet)

model_global.compile(loss='bce', optimizer='adam', metrics = 'accuracy')


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


