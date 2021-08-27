import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

fname = "AMG6a_130000C__rec00001230.bmp"
img = cv.imread(fname, 0)
cv.imshow(fname, img)

# apply global hist equalization
equ = cv.equalizeHist(img)
cv.imshow('After global hist equalization', equ)

# denoising using gaussian filter
gaus = cv.GaussianBlur(equ, (9, 9), 0)
# cv.imshow('After Gaussian blur, k=5', gaus)

# denoising using median filter
# median = cv.medianBlur(img, 5)
# cv.imshow('After Median blur, k=5', median)

# median7 = cv.medianBlur(img, 7)
# cv.imshow('After Median blur, k=7', median7)

# denoising using nonlocal means algorithm
dst1 = cv.fastNlMeansDenoising(gaus, None, 16, 7, 21)
# cv.imshow('After nonlocal denoising (gaus)', dst1)

# dst2 = cv.fastNlMeansDenoising(img, None, 20, 7, 21)
# cv.imshow('After nonlocal denoising (original)', dst2)

# detect edges using canny algorithm
# edges = cv.Canny(img, 100, 200)
# cv.imshow('Edges', edges)

# apply global threshhold
ret, th1 = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
cv.imshow('After global threshold', th1)

# create a filled circle


# apply bitwise operation


# apply Otsu's binarization
# ret, th = cv.threshold(dst1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.imshow('After Otsu binarization', th)

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
