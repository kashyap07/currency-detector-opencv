#usr/bin/env python
# @Date:	2017-03-22

# test file

from utils import *

img = read_img('files/500.jpg')
img = resize_img(img, 0.6)

img = img_to_gray(img)
#img = canny_edge(img, 720, 350)
#img = canny_edge(img, 270, 390)

#img = laplacian_edge(img)

#img = find_contours(img)
#img = img_to_neg(img)
#img = binary_thresh(img, 85)
#img = close(img)
#img = adaptive_thresh(img)
#img = sobel_edge(img, 'v')
img = sobel_edge2(img)
#img = median_blur(img)
#img = binary_thresh(img, 106)
#img = dilate_img(img)
#img = binary_thresh(img, 120)

#img = foo_convolution(img)
#histogram(img)
#fourier(img)
#img = harris_edge(img)

display('image',img)
'''

kernel = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
display('image', closing)
'''