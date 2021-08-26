import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

fname = "AMG6a_crack1_130000.bmp"
img = cv.imread(fname, 0)
# cv.imshow(fname, img)

# denoising using gaussian filter
gaus = cv.GaussianBlur(img, (5, 5), 0)
# cv.imshow('After Gaussian blur, k=5', gaus)

# denoising using median filter
# median = cv.medianBlur(img, 5)
# cv.imshow('After Median blur, k=5', median)

# median7 = cv.medianBlur(img, 7)
# cv.imshow('After Median blur, k=7', median7)

# denoising using nonlocal means algorithm
dst1 = cv.fastNlMeansDenoising(gaus, None, 20, 7, 21)
cv.imshow('After nonlocal denoising (gaus)', dst1)

dst2 = cv.fastNlMeansDenoising(img, None, 20, 7, 21)
cv.imshow('After nonlocal denoising (original)', dst2)

# apply Otsu's binarization
# ret, th = cv.threshold(median, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.imshow('After Otsu binarization', th)

# apply adaptive thresholding
# th = cv.adaptiveThreshold(gaus, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                           cv.THRESH_BINARY, 9, 2)
# cv.imshow('After adaptive thresholding', th)

# # apply global hist equalization
# equ = cv.equalizeHist(median7)
# cv.imshow('After global hist equalization', equ)

# create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img)
# cv.imshow('After CLAHE hist equalization', cl1)

# plot the histogram
# plt.hist(dst.ravel(), 256, [0, 256])
# plt.show()


cv.waitKey()
cv.destroyAllWindows()
