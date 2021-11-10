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

TILE_SIDE = 64
NN_SIDE = 32



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




##################



def convolution_operation(entered_input, filters=64, regularizer=None):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = (3,3), padding = "same", kernel_regularizer=regularizer )(entered_input)
    act1 = ReLU()(conv1)
    
    # Taking first input and implementing the second conv block
    conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same", kernel_regularizer= regularizer)(act1)
    act2 = ReLU()(conv2)
    
    return act2


def encoder(entered_input, filters=64):
    # Collect the start and end of each sub-block for normal pass and skip connections
    enc1 = convolution_operation(entered_input, filters, regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))
    MaxPool1 = MaxPooling2D(strides = (2,2))(enc1)
    return enc1, MaxPool1





def decoder(entered_input, skip, filters=64):
    # Upsampling and concatenating the essential features
    Upsample = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input)
    Connect_Skip = Concatenate()([Upsample, skip])
    out = convolution_operation(Connect_Skip, filters)
    return out


def U_Net(Image_Size):
    # Take the image size and shape
    input1 = Input(Image_Size)
    
    # Construct the encoder blocks
    skip1, encoder_1 = encoder(input1, 16)
    skip2, encoder_2 = encoder(encoder_1, 16*2)
    skip3, encoder_3 = encoder(encoder_2, 16*4)
    skip4, encoder_4 = encoder(encoder_3, 16*8)
    
    # Preparing the next block
    conv_block = convolution_operation(encoder_4, 64*16, regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))
    
    # Construct the decoder blocks
    decoder_1 = decoder(conv_block, skip4, 16*8)
    decoder_2 = decoder(decoder_1, skip3, 16*4)
    decoder_3 = decoder(decoder_2, skip2, 16*2)
    decoder_4 = decoder(decoder_3, skip1, 16)
    
    out = Conv2D(1, 1, padding="same", activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(decoder_4)

    model = Model(input1, out)
    return model

##################

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
	global img_orig, mask_positive, TILE_SIDE, NN_SIDE, mask_negative
	assert len(x_arr)==len(y_arr)
	tiles  = []
	labels = []
	for i in range(len(x_arr)):
		y,x = y_arr[i], x_arr[i]
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


