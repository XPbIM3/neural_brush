import os
import sys
import time


import cv2 as cv
import numpy as np
from time import perf_counter

if len(sys.argv)<2:
	print('Pass an image file as an argument!')
	sys.exit()
F_NAME = sys.argv[1]
if not os.path.exists(F_NAME):
	print('File not exists')
	sys.exit()

print(F_NAME)
img_orig  = cv.imread(F_NAME, -1)



window = cv.namedWindow('img')


def render():
    screen = img_orig.copy()
    screen = np.bitwise_or(img_orig, img_orig)
    screen = np.bitwise_or(img_orig, screen)
    cv.imshow('img', screen)


t1 = perf_counter()
for i in range(1000):
    render()

t2=perf_counter()


frames_per_sec = 1000/(t2-t1)

print(frames_per_sec)