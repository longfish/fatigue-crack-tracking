import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

fname = "AMG6a_130000C__rec00001230.bmp"
img = cv.imread(fname, 0)
cv.namedWindow("image")


def nothing(x):
    pass


def crack_detection(img, h, th_g):

    # apply global hist equalization
    equ = cv.equalizeHist(img)
    # cv.imshow('After global hist equalization', equ)

    # denoising using gaussian filter
    gaus = cv.GaussianBlur(equ, (5, 5), 0)
    # cv.imshow('After Gaussian blur, k=5', gaus)

    # denoising using nonlocal means algorithm
    dst1 = cv.fastNlMeansDenoising(gaus, None, h, 7, 21)
    # cv.imshow('After nonlocal denoising (gaus)', dst1)

    # apply global threshhold
    _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY)
    # cv.imshow('After global threshold', th1)

    # apply Otsu's binarization
    # _, th1 = cv.threshold(dst1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imshow('After Otsu binarization', ostu)

    return th1


# create trackbars for filter strength h and global threshold
cv.createTrackbar('h', 'image', 0, 20, nothing)
cv.createTrackbar('th_g', 'image', 0, 150, nothing)

img1 = crack_detection(img, 0, 0)

while(True):
    cv.imshow('image', img1)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of two trackbars
    h = cv.getTrackbarPos('h', 'image')
    th_g = cv.getTrackbarPos('th_g', 'image')

    img1 = crack_detection(img, h, th_g)


cv.destroyAllWindows()
