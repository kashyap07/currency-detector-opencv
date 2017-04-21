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
edged = cv2.Canny(gray, 75, 200)	
'''

# show the original image and the edge detected image
#cv2.imshow("Image", image)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
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


maxVal = 8
maxPt = -1
maxKp = 0
orb = cv2.ORB_create()


# hard coded here
# must preprcoess and warp the image

#img1 = read_img('files/test_100_2.jpg')
#img1 = read_img('files/test_50_2.jpg')
img1 = read_img('files/test_20_2.jpg')

# img = img_to_gray(img1)
# display('original', img1)

(kp1, des1) = orb.detectAndCompute(img1, None)
# orb is an alternative for SIFT

l = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

for i in range(0, len(l)):

	# print l[i]

	img2 = cv2.imread(l[i])  # trainImag

	# cv2.imshow("Input Image", img1)
	# cv2.imshow("Stored image", img2)
	# cv2.waitKey(0)
	# cv2.imwrite('10x.png',img1)
	# cv2.destroyAllWindows()
	# Initiate SIFT detector

	# find the keypoints and descriptors with SIFT

	(kp2, des2) = orb.detectAndCompute(img2, None)

	# create BFMatcher object
	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# Match descriptors.
	# matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	# matches = sorted(matches, key = lambda x:x.distance)
	# Draw first 10 matches.
	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None, flags=2)

	# plt.imshow(img3),plt.show()

	# ### brute force matcher

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)

	# Apply ratio test

	good = []
	for (m, n) in matches:
		if m.distance < 0.75 * n.distance:
			good.append([m])

	if len(good) > maxVal:
		maxVal = len(good)
		maxPt = i
		maxKp = kp2

		# print good

	#print(i, ' ', l[i], ' ', len(good))

	# cv2.drawMatchesKnn expects list of lists as matches.

if maxVal != 8:
	#print(l[maxPt])
	#print('good matches ', maxVal)
	img2 = cv2.imread(l[maxPt])
	img3 = cv2.drawMatchesKnn(img1, kp1, img2, maxKp, good, 2,)
	(plt.imshow(img3), plt.show())

	note = str(l[maxPt])[6:-4]
	print('\nDetected denomination: Rs. ', note)
else:
	print('No Matches')
