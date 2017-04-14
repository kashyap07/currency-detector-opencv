#!usr/bin/env python
# @Author:	Suhas Kashyap
# @Date:	2017-03-22

# test file

from utils import *

img = read_img('files/500.jpg')

img = img_to_gray(img)
img = canny_edge(img)
#img = find_contours(img)
#img = img_to_neg(img)
#img = binary_thresh(img, 128)
#img = adaptive_thresh(img)
#img = sobel_edge(img, 'h')
#histogram(img)
#fourier(img)

display('image', img)
