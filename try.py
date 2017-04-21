#!/usr/bin/python
# @Date: 	2017-03-22

# test file
# TODO:
# 	Figure out four point transform
#	Figure out testing data warping
# 	Use webcam as input
# 	Figure out how to use contours
# 		Currently detects inner rect -> detect outermost rectangle
# 	Try using video stream from android phone


from utils import *
from matplotlib import pyplot as plt

import subprocess
from gtts import gTTS

# image = read_img('files/500_1.jpg')
# orig = image
# orig = resize_img(orig, 0.5)
# img = resize_img(img, 0.6)

# img = img_to_gray(img)
# img = canny_edge(img, 720, 350)
# img = canny_edge(img, 270, 390)

# img = laplacian_edge(img)

# img = find_contours(img)
# img = img_to_neg(img)
# img = binary_thresh(img, 85)
# img = close(img)
# img = adaptive_thresh(img)
# img = sobel_edge(img, 'v')
# img = sobel_edge2(img)
# img = median_blur(img)
# img = binary_thresh(img, 106)
# img = dilate_img(img)
# img = binary_thresh(img, 120)

# img = foo_convolution(img)
# histogram(img)
# fourier(img)
# img = harris_edge(img)

# display('image',img)

# show the original image and the edge detected image
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour

# must define here

'''

kernel = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
display('image', closing)
'''

'''
r = 500.0/ image.shape[1]
dim = (500, int(image.shape[0] * r))
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#image = resize_img(image, 0.6)
ratio = image.shape[0] / 500.0

#display('image',image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200
'''

'''
show the original image and the edge detected image
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

find largest contours
'''

'''
(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#print "cnts: ", cnts
screenCnt = 0 

n = .02
flag = True
while(n<.9 and flag==True):		#remove while loop if wrong contour is being detected
		print(n)
		for c in cnts:
				# approximate the contour
				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c,n*peri, True)
				print("Approx: ", len(approx))
		
				# if our approximated contour has four points, then we
				# can assume that we have found our screen
				if len(approx) == 4:
						screenCnt = approx
						flag=False
						break
		n+=.01

warped = image
'''

'''
print('Screen count:', screenCnt)

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
warped = four_point_transform(orig, screenCnt.reshape(4, 2))
#warped = orig[ screenCnt[0][0][1]:screenCnt[1][0][1],screenCnt[0][0][0]:screenCnt[2][0][0]]
cv2.imshow('Warped',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
r = 500.0/warped.shape[1]
dim = (500, 240)
warped = cv2.resize(warped, dim, interpolation = cv2.INTER_AREA)

#cv2.imshow("Original", image)
cv2.imshow("Orignal", orig)
cv2.imshow("Scanned", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

max_val = 8
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()
# orb is an alternative to SIFT

#test_img = read_img('files/test_100_2.jpg')
#test_img = read_img('files/test_50_2.jpg')
test_img = read_img('files/test_20_2.jpg')
#test_img = read_img('files/test_100_3.jpg')
# test_img = read_img('files/test_20_4.jpg')

# resizing must be dynamic
original = resize_img(test_img, 0.4)
display('original', original)

# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected denomination: Rs. ', note)

	audio_file = 'audio/' + note + '.mp3'

	# audio_file = "value.mp3
	# tts = gTTS(text=speech_out, lang="en")
	# tts.save(audio_file)
	return_code = subprocess.call(["afplay", audio_file])

	(plt.imshow(img3), plt.show())
else:
	print('No Matches')