import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.keras import regularizers
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



def masked_bce_2ch(y_true, y_pred):
	#requires two channels in y_pred.  First - for value,  second - for mask
	#sample_in_bach, y, x, channel(0 - value, 1 - mask)
	y_tr = y_true[:,:,:,0:1]
	y_pr = y_pred.copy()
	y_tr_mask = y_true[:,:,:,1]
	y_tr[y_tr_mask==0] = 0
	y_pr[y_tr_mask==0] = 0
	loss_func = tf.keras.losses.BinaryCrossentropy(from_logits = False)
	#breakpoint()
	return loss_func(y_tr, y_pr)


def masked_bce_byvalue(y_true, y_pred):
	#If value in y_true assigned to anything larger tnah  1.0 , that means this values "masked" and should be excluded ffrom loss
	y_tr = y_true.copy()
	y_pr = y_pred.copy()
	y_pr[y_tr>1.0]=0.0
	y_tr[y_tr>1.0]=0.0
	loss_func = tf.keras.losses.BinaryCrossentropy(from_logits = False)
	return loss_func(y_tr, y_pr)




y_true = np.ones((5,10,10,1), dtype=np.float32)
y_pred = np.zeros((5,10,10,1), dtype=np.float32)

#y_pred [0:5,1:10,0:10,0] = 2.0


loss_vall = masked_bce_byvalue(y_true, y_pred)
print(loss_vall.numpy())