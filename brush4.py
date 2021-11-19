import os
import sys

import cv2 as cv
import numpy as np

import threading
import requests

if len(sys.argv)<2:
	print('Pass an image file as an argument!')
	sys.exit()
F_NAME = sys.argv[1]
if not os.path.exists(F_NAME):
	print('File not exists')
	sys.exit()

print(F_NAME)
img_orig  = cv.imread(F_NAME, cv.IMREAD_COLOR)
screen = img_orig.copy()

need_redraw = False
need_predict = False
run_loop = False


mouse_coords = (0,0)

TILE_SIDE = 128
HALF_TILE_SIDE = TILE_SIDE//2
N_CH = 3
assert len(img_orig.shape)==3
assert N_CH == img_orig.shape[2]
current_tile = np.zeros((TILE_SIDE, TILE_SIDE, N_CH), dtype=np.uint8)
predicted_tile = np.zeros((TILE_SIDE, TILE_SIDE, 1), dtype=np.uint8)





def mouseCb(event, x, y, flags, params):
	global mouse_coords, need_predict, need_redraw
	if (event == cv.EVENT_MOUSEMOVE):
		mouse_coords = (y,x)
		need_predict = True





def checkCoords(y,x,img_shape):
	global HALF_TILE_SIDE
	res = True
	if x < HALF_TILE_SIDE or y < HALF_TILE_SIDE:
		res = False
	if x>img_shape[1]-HALF_TILE_SIDE or y>img_shape[0]-HALF_TILE_SIDE:
		res = False
	return res




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
	



def parseKey(key):
	global run_loop
	print('key pressed:', key)
	if key == 27:
		run_loop = False


def getTile():
	y,x = mouse_coords
	y,x = conditionCoords(y,x, img_orig.shape)
	t = img_orig[y-HALF_TILE_SIDE:y+HALF_TILE_SIDE, x-HALF_TILE_SIDE:x+HALF_TILE_SIDE,:]
	assert t.shape == (TILE_SIDE, TILE_SIDE, N_CH)
	return t


def predictTile(tile):
	headers = {'content-type':'application/octet-stream'}
	resp = None
	try:
		resp = requests.post('http://127.0.0.1:1488/', data = tile.tobytes(), headers = headers)
	except:
		print('Request failed!')
	if resp!=None:
		p = np.frombuffer(resp.content, dtype=np.uint8)
		p = p.reshape(TILE_SIDE, TILE_SIDE,1)
	else:
		p=np.zeros((TILE_SIDE,TILE_SIDE,1), dtype=np.uint8)
	return p




def predictThreadFunc():
	global predicted_tile, need_redraw, need_predict, run_loop
	while run_loop:
		if need_predict:
			need_predict = False
			t = getTile()
			p = predictTile(t)
			predicted_tile = np.broadcast_to(p, (TILE_SIDE, TILE_SIDE, 3))
			need_redraw = True
		else:
			pass



def screenThreadFunc():
	global run_loop, need_redraw, mouse_coords, img_orig, screen
	while run_loop:
		if need_redraw:
			need_redraw = False
			screen = img_orig.copy()
			y,x = mouse_coords
			y,x =  conditionCoords(y,x, img_orig.shape)
			screen[y-HALF_TILE_SIDE:y+HALF_TILE_SIDE, x-HALF_TILE_SIDE:x+HALF_TILE_SIDE,:] = predicted_tile
			cv.imshow('img', screen)


print('init window')
cv.namedWindow('img')
cv.setMouseCallback("img", mouseCb)
cv.imshow('img', screen)
key = 0
print('start main loop')

predictThread = threading.Thread(target=predictThreadFunc)
screenThread = threading.Thread(target = screenThreadFunc)
run_loop = True
predictThread.start()
print('Thread create')
screenThread.start()
print('Thread run')
while key != 27:
	key = cv.waitKey(0)
	parseKey(key)
cv.destroyAllWindows()


