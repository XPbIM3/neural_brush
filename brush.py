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
BRUSH_TYPES = {49:'REGULAR', 50:'MAGIC', 51: 'NEURAL'}
BRUSH_TYPE_CURRENT = 'REGULAR'

BATCH_SIZE = 16
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
runTrainThread = False
needRecount = True
model = None

TILE_SIDE = 128
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
	global model, gen,runPixelPoolThread, tiles_batches, runTrainThread
	while runPixelPoolThread:
		if len(tiles_batches)>1 and runTrainThread:
			model.fit(x=gen, steps_per_epoch=len(tiles_batches))
			runTrainThread = False


def render(pointer_coords):
	global img_orig, screen, mask, BRUSH_TYPE_CURRENT, model
	screen = cv.bitwise_or(img_orig,mask)
	if pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and BRUSH_TYPE_CURRENT == 'REGULAR':
		cv.circle(screen, pointer_coords, BRUSH_SIZE, BRUSH_COLOR, -1)
	elif pointer_coords[0]!=-1 and pointer_coords[1]!=-1 and BRUSH_TYPE_CURRENT == 'NEURAL':
		x=pointer_coords[0]
		y=pointer_coords[1]
		t = img_orig[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]
		t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))/255.0
		t = t.reshape(1,NN_SIDE,NN_SIDE,N_CH)
		r = model.predict(t)
		
		r=(r*255).astype(np.uint8)[0]
		a,th = cv.threshold(r,128, 255,cv.THRESH_BINARY)
		target = cv.resize(th, (TILE_SIDE, TILE_SIDE))
		target3 = np.zeros((TILE_SIDE,TILE_SIDE,N_CH), dtype=np.uint8)
		target3[:,:,0]=target
		target3[:,:,1]=target
		target3[:,:,2]=target
		screen[y-TILE_SIDE//2:y+TILE_SIDE//2, x-TILE_SIDE//2:x+TILE_SIDE//2, :]=target3
	cv.imshow('img', screen)





def refreshPool():
	global mask, pixelPool, img_orig, TILE_SIDE, tilePool, runPixelPoolThread, labelPool, needRecount
	print('refresh thread started')
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


def refreshPoolSingle():
	global mask, pixelPool, img_orig, TILE_SIDE, tilePool, runPixelPoolThread, labelPool, needRecount
	print('refresh process started')
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
					t = cv.resize(t, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, N_CH))/255.0
					l = cv.resize(l, (NN_SIDE, NN_SIDE)).reshape((NN_SIDE,NN_SIDE, 1))/255.0
					tilePool.append(t)
					labelPool.append(l)
			needRecount = False
			refreshBatchPool()
		else:
			pass
		



def refreshBatchPool():
	global tilePool, labelPool, tiles_batches, labels_batches, BATCH_SIZE
	if len(tilePool)>=BATCH_SIZE and len(labelPool)>=BATCH_SIZE:
		
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
		exportSet()
		print('batchPool:', len(tiles_batches))




def trainSingle():
	global runTrainThread
	refreshPoolSingle()
	runTrainThread = True


def mouseCb(event, x, y,flags, params):
	global img_orig,screen, mask, tilePool, needRecount
	

	if event == cv.EVENT_MOUSEMOVE or event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONUP:
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

def exportSet():
	global tiles_batches, labels_batches, tilePool, labelPool
	print('batches len:' , len(tiles_batches))
	for i in range(len(tiles_batches)):
		tile_res = (np.vstack(tiles_batches[i])*255).astype(np.uint8)
		label_res = (np.vstack(labels_batches[i])*255).astype(np.uint8)
		cv.imwrite('res/res_tiles_'+str(i)+'.jpg', tile_res)
		cv.imwrite('res/res_tiles_'+str(i)+'_l.jpg', label_res)

	print('writes done')



def parseKey(key):
	global BRUSH_SIZE, runTrainThread, BRUSH_TYPE_CURRENT
	if key==43:
		BRUSH_SIZE+=1
	elif key==45:
		BRUSH_SIZE-=1
		if BRUSH_SIZE <=0:
			BRUSH_SIZE=1
	elif key in BRUSH_TYPES:
		print('have brush')
		print(BRUSH_TYPES[key])
		BRUSH_TYPE_CURRENT = BRUSH_TYPES[key]
	elif key == 116:
		print('train single shot')
		trainSingle()
	else:
		print('key pressed:', key)



print('set up a keras')
u_model = models.unet_2d(input_size = (NN_SIDE, NN_SIDE, N_CH), filter_num = [16,32,64,128], n_labels=1, output_activation='Sigmoid', batch_norm=False)
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.InputLayer(input_shape = (NN_SIDE, NN_SIDE, N_CH)))
#model.add(tf.keras.layers.Lambda(lambda x: x/255))
#model.add(u_model)
model = u_model
model.compile(loss='bce', optimizer='adam', metrics = 'accuracy')

print('init window')
cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)
gen = getBatchGenerator()

print('start a tile thread')

#refreshThread = threading.Thread(target = refreshPool)
#refreshThread.start()

trainThread = threading.Thread(target = trainHandler)
trainThread.start()

key = 0
while key != 27:
	render((-1,-1))
	key = cv.waitKey(0)
	parseKey(key)

cv.destroyAllWindows()
runPixelPoolThread = False
#refreshThread.join()
#trainThread.join()


