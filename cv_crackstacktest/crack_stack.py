import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

fname = "AMG6a_130000C__rec00001235.bmp"
img = cv.imread(fname, 0)
# cv.imshow(fname, img)

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
# ret, th1 = cv.threshold(dst1, 110, 255, cv.THRESH_BINARY)
# cv.imshow('After global threshold', th1)

# apply Otsu's binarization
ret, ostu = cv.threshold(dst1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('After Otsu binarization', ostu)

# create an intact cross-section
img_blur = cv.GaussianBlur(img, (15, 15), 0)
ret2, ostu2 = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
ostu2_opening = cv.morphologyEx(ostu2, cv.MORPH_OPEN, kernel)
# cv.imshow('After Otsu binarization (cross-section)', ostu2_opening)

# find the contour of the cross-section
contours, hierarchy = cv.findContours(
    ostu2_opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnt = contours[0]
M = cv.moments(cnt)
cx = int(M['m10']/M['m00'])  # centroid
cy = int(M['m01']/M['m00'])
_, radius = cv.minEnclosingCircle(cnt)

# create a circle mask
mask = np.zeros(img.shape[:2], np.uint8)
cv.circle(mask, (cx, cy), int(radius-10), 255, thickness=-1)
# cv.imshow('Mask circle', mask)

# apply bitwise operation
crack = cv.bitwise_not(ostu, mask=mask)
# crack_opening = cv.morphologyEx(crack, cv.MORPH_OPEN, kernel)
cv.imshow('Crack', crack)

# apply adaptive thresholding
# th = cv.adaptiveThreshold(dst1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                           cv.THRESH_BINARY, 19, 8)
# cv.imshow('After adaptive thresholding', th)

# create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img)
# cv.imshow('After CLAHE hist equalization', cl1)

# plot the histogram
# plt.hist(equ.ravel(), 256, [0, 256])
# plt.show()


cv.waitKey()
cv.destroyAllWindows()
