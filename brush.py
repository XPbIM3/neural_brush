import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import cv2 as cv
import numpy as np
import threading
import tensorflow as tf
from keras_unet_collection import models
from keras_unet_collection import losses

TILE_STEP = 4

BRUSH_SIZE = 10
BRUSH_COLOR = (250,250,250)

img_orig  = cv.imread('1.png', -1)
N_CH = img_orig.shape[2]

screen = img_orig.copy()
mask = np.zeros((screen.shape[0],screen.shape[1], screen.shape[2]), dtype=np.uint8)
brush = np.zeros((screen.shape[0],screen.shape[1], screen.shape[2]), dtype=np.uint8)

tiles_batches = []
labels_batches = []

pixelPool = []
tilePool = []
labelPool = []
runPixelPoolThread = True
needRecount = True
model = None

TILE_SIDE = 64
NN_SIDE = 32


def getBatchGenerator():
	global tiles_batches, labels_batches
	while True:
		i = np.random.randint(0,len(tiles_batches))
		yield (tiles_batches[i], labels_batches[i])

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def trainHandler():
	global model, gen,runPixelPoolThread, tiles_batches
	while runPixelPoolThread:
		if len(tiles_batches)>1:
			model.fit(x=gen, steps_per_epoch=len(tiles_batches))


def render(pointer_coords):
	global img_orig, screen, mask
	screen = cv.bitwise_or(img_orig,mask)
	if pointer_coords[0]!=-1 and pointer_coords[1]!=-1:
		cv.circle(screen, pointer_coords, BRUSH_SIZE, BRUSH_COLOR, -1)
	cv.imshow('img', screen)

def refreshBatchPool():
	global tilePool, labelPool, tiles_batches, labels_batches
	if len(tilePool)>=32 and len(labelPool)>=32:
		BATCH_SIZE = 32
		tl = len(tilePool)
		ll = len(labelPool)
		assert tl == ll
		count = tl // BATCH_SIZE
		tiles = np.array(tilePool[0:count*BATCH_SIZE])
		labels = np.array(labelPool[0:count*BATCH_SIZE])
		
		shuffle_in_unison_scary(tiles, labels)

		t_shape = tiles[0].shape
		l_shape = labels[0].shape
		tiles_batches = tiles.reshape(count, BATCH_SIZE,*t_shape)
		labels_batches = labels.reshape(count, BATCH_SIZE,*l_shape)




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
				np.random.shuffle(pixelPool)
				for coords in pixelPool[::TILE_STEP]:
					y,x = coords
					t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
					l = mask[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2,0]
					if t.shape == (TILE_SIDE,TILE_SIDE, N_CH):
						t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))
						l = cv.resize(l, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, 1))/255.0
						tilePool.append(t)
						labelPool.append(l)
				refreshBatchPool()
			else:
				pass


			needRecount = False
	print('pixelPool stopped by global flag!')


def mouseCb(event, x, y,flags, params):
	global img_orig,screen, mask, tilePool, needRecount
	

	if event == cv.EVENT_MOUSEMOVE or event == cv.EVENT_LBUTTONDOWN:
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
		if BRUSH_SIZE <=0:
			BRUSH_SIZE=1



print('set up a keras')
u_model = models.att_unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=1, output_activation='Sigmoid', batch_norm=False)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE, NN_SIDE, N_CH)))
model.add(tf.keras.layers.Lambda(lambda x: x/255))
model.add(u_model)
model.compile(loss=losses.iou_seg, optimizer='adam', metrics = 'accuracy')

print('init window')
cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)

gen = getBatchGenerator()

print('start a tile thread')

refreshThread = threading.Thread(target = refreshPool)
refreshThread.start()

trainThread = threading.Thread(target = trainHandler)
trainThread.start()

key = 0
while key != 27:
	render((-1,-1))
	key = cv.waitKey(0)
	parseKey(key)

cv.destroyAllWindows()
runPixelPoolThread = False
refreshThread.join()



