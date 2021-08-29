import numpy as np
import cv2 as cv
import os


def crack_detection(img, h, th_g):
    '''Detect the crack, return the cross-section as an object'''

    equ = cv.equalizeHist(img)  # apply global hist equalization
    gaus = cv.GaussianBlur(equ, (5, 5), 0)  # denoising using gaussian filter
    # denoising using nonlocal means algorithm
    dst1 = cv.fastNlMeansDenoising(gaus, None, h, 7, 21)

    # apply Otsu's binarization
    # _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # apply global threshhold
    _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY)

    return th1


def cross_section_contour(img, offset):
    '''Get the cross-section contour of the image
       Output: the centroid coordinates and radius of the contour circle (reduced by an offset)
    '''

    # create an intact cross-section
    img_blur = cv.GaussianBlur(img, (15, 15), 0)
    _, ostu2 = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    ostu2_opening = cv.morphologyEx(ostu2, cv.MORPH_OPEN, kernel)

    # find the contour of the cross-section
    contours, _ = cv.findContours(
        ostu2_opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = contours[0]
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])  # centroid
    cy = int(M['m01']/M['m00'])
    _, radius = cv.minEnclosingCircle(cnt)

    return cx, cy, int(radius-offset)


def find_max_contour(img):
    '''find the maximum contour by area of bounding rectangular'''
    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    max_area = 0
    max_cont = contours[0]
    for cont in contours:
        _, _, w, h = cv.boundingRect(cont)
        area = w*h
        if (area > max_area):
            max_area = area
            max_cont = cont
    return max_cont


def crack_extraction(img):
    '''
    Extract crack from the noised image
    Output: the convex hull image of the crack
    '''
    kernel = np.ones((2, 2), np.uint8)
    crack_final = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    max_cont = find_max_contour(crack_final)
    crack_hull = cv.convexHull(max_cont)
    blank = np.zeros((736, 736), np.uint8)
    cv.drawContours(blank, [crack_hull], 0, 250, -1)

    return blank


# find the directories of the crack, and list the crack images
cwd = os.getcwd()
folders = [dir for dir in os.listdir(cwd) if os.path.isdir(dir)]

for f in folders:
    # loop the folders
    dir = os.path.join(cwd, f)
    imgs = os.listdir(dir)
    crack_final = np.zeros((736, 736), np.uint8)
    for img in imgs:
        # loop the images in the crack folder
        img_dir = os.path.join(dir, img)
        img_array = cv.imread(img_dir, 0)
        crack = crack_detection(img_array, 12, 100)
        # cv.imshow(img, crack)

        cx, cy, r = cross_section_contour(
            img_array, 14)  # get the contour circle
        # create a circle mask
        cir_mask = np.zeros(img_array.shape[:2], np.uint8)
        cv.circle(cir_mask, (cx, cy), r, 255, thickness=-1)

        # apply bitwise operation to obtain the crack object
        crack = cv.bitwise_not(crack, mask=cir_mask)
        crack_final = cv.add(crack_final, crack)  # add all the images together

    crack_final = crack_extraction(crack_final)
    cv.imshow('Final crack object', crack_final)


cv.waitKey()
cv.destroyAllWindows()
