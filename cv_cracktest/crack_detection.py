'''
Detect and measure a single crack.

Note:
    Put the crackmeasure.py file in the same folder.
'''

import numpy as np
import cv2 as cv
import crackmeasure as cmeas
from matplotlib import pyplot as plt

PIXEL = 0.0061

fname = "AMG5_135000C__rec00001203.bmp"
img = cv.imread(fname, 0)
cv.imshow(fname, img)

# apply global hist equalization
equ = cv.equalizeHist(img)
# cv.imshow('After global hist equalization', equ)

# denoising using gaussian filter
gaus = cv.GaussianBlur(equ, (5, 5), 0)
# cv.imshow('After Gaussian blur, k=5', gaus)

# denoising using median filter
# median = cv.medianBlur(img, 5)
# cv.imshow('After Median blur, k=5', median)

# median7 = cv.medianBlur(img, 7)
# cv.imshow('After Median blur, k=7', median7)

# denoising using nonlocal means algorithm
dst1 = cv.fastNlMeansDenoising(gaus, None, 8, 7, 21)
# cv.imshow('After nonlocal denoising (gaus)', dst1)

# dst2 = cv.fastNlMeansDenoising(img, None, 20, 7, 21)
# cv.imshow('After nonlocal denoising (original)', dst2)

# detect edges using canny algorithm
# edges = cv.Canny(img, 100, 200)
# cv.imshow('Edges', edges)

# apply global threshhold
ret, th1 = cv.threshold(dst1, 100, 255, cv.THRESH_BINARY)
# cv.imshow('After global threshold', th1)

# apply Otsu's binarization
# ret, ostu = cv.threshold(dst1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.imshow('After Otsu binarization', ostu)

# create an intact cross-section
img_blur = cv.GaussianBlur(img, (15, 15), 0)
ret2, ostu2 = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cnt = cmeas.contour_sort_area(ostu2)

# blank = np.zeros((736, 736), np.uint8)
# cv.drawContours(blank, [cnt[0]], 0, 250, -1)
# cv.imshow("Crack", blank)


# find the contour of the cross-section
cross = cnt[0]
M = cv.moments(cross)
cx = int(M['m10']/M['m00'])  # centroid
cy = int(M['m01']/M['m00'])
_, r = cv.minEnclosingCircle(cross)

# create a circle mask
mask = np.zeros(img.shape[:2], np.uint8)
r = int(r-14)
cv.circle(mask, (cx, cy), r, 255, thickness=-1)
# cv.imshow('Mask circle', mask)

# apply bitwise operation to get the crack object
crack = cv.bitwise_not(th1, mask=mask)
# crack_opening = cv.morphologyEx(crack, cv.MORPH_OPEN, kernel)
cv.imshow('Crack', crack)

# extract all cracks
# crack_hull = cmeas.contour_sort_area(crack)
crack_hull = cmeas.cracks_extraction(crack, (cx, cy, r), num=2)

print(len(crack_hull))
# draw the crack contours
blank1 = np.zeros(img.shape[:2], np.uint8)
cv.drawContours(blank1, [crack_hull[0]], 0, 250, -1)
cv.imshow("Crack1", blank1)

blank2 = np.zeros(img.shape[:2], np.uint8)
cv.drawContours(blank2, [crack_hull[1]], 0, 250, -1)
cv.imshow("Crack2", blank2)

# (area, depth, side_length) = cmeas.crack_geo_calc(
#     (np.mean(cx), np.mean(cy), np.mean(r)), crack_hull)
# print("Crack: area =", area*PIXEL*PIXEL, ", depth =",
#       depth*PIXEL, ", side length =", side_length*PIXEL)


cv.waitKey()
cv.destroyAllWindows()
