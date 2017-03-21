# @Author: Suhas Kashyap
# Date:	2017-03-22

# UE14CS348 Digital Image Processing Mini Project
# Indian paper currency detection

# utils.py
# contains utility functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
#from scipy import ndimage


# read image as is
def read_img(file_name):
	img = cv2.imread(file_name)
	return img


# convert image to grayscale
def img_to_gray(image):
	img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return img_gray


# convert image to negative
def img_to_neg(image):
	img_neg = 255 - image
	return img_neg


# binarize (threshold)
# retval not used currently
def binarize(image, threshold):
	retval, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
	return img_thresh


# sobel edge operator
def sobel_edge(image, align):
	img_horiz = cv2.Sobel(image, cv2.CV_64F, 0, 1)
	img_vert = cv2.Sobel(image, cv2.CV_64F, 1, 0)
	if align == 'h':
		return img_horiz
	elif align == 'v':
		return img_vert
	else:
		print('use h or v')


# calculate histogram
def histogram(image):
	hist = cv2.calcHist([image], [0], None, [256], [0, 256])
	# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
	plt.plot(hist)
	plt.show()


# fast fourier transform
def fourier(image):
	f = np.fft.fft2(image)
	fshift = np.fft.fshift(f)
	magnitude_spectrum = 20 * np.log(np.abs(fshift))

	plt.subplot(121), plt.imshow(image, cmap='gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()


# calculate scale and fit into display
def display(window_name, image):
	screen_res = 1440, 900	# MacBook Air
	scale_width = screen_res[0] / image.shape[1]
	scale_height = screen_res[1] / image.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(image.shape[1] * scale)
	window_height = int(image.shape[0] * scale)

	# reescale the resolution of the window
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, window_width, window_height)

	# display image
	cv2.imshow(window_name, image)

	# wait for any key to quit the program
	cv2.waitKey(0)
	cv2.destroyAllWindows()