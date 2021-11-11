from skimage import data
import napari
import cv2 as cv

img = cv.imread('1_full.png',-1)



viewer = napari.view_image(img[:,:,::-1])
@viewer.bind_key('r')
def res(viewer):
    viewer.reset_view()