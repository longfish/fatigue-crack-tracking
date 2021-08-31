'''
Compute the crack geometry from a stack of crack images

Usage:
  Put the script into the parent path of all crack images (for one sample)

Version:
  2021-8-30, initial finished version of crack tracking program
'''

import numpy as np
import cv2 as cv
import os

PIXEL = 0.0061  # 1 pixel == 0.0061 mm


def crack_detection(img, h, th_g):
    '''Detect the crack, return the cross-section as an object'''

    equ = cv.equalizeHist(img)  # apply global hist equalization
    gaus = cv.GaussianBlur(equ, (5, 5), 0)  # denoising using gaussian filter
    # denoising using nonlocal means algorithm
    dst1 = cv.fastNlMeansDenoising(gaus, None, h, 7, 21)

    # apply Otsu's binarization (may be more stable)
    _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # apply global threshhold (may be more accurate)
    # _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY)

    return th1


def cross_section_contour(img, offset):
    '''Get the cross-section contour of the image
       Output: the centroid coordinates and radius of the contour circle (reduced by an offset)
    '''

    # create an intact cross-section
    img_blur = cv.GaussianBlur(img, (15, 15), 0)
    _, ostu2 = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # find the contour of the cross-section (largest contour)
    cnt = find_max_contour(ostu2)
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


def crack_hull_extraction(img):
    '''
    Extract crack from the noised image
    Output: the approximated polygon of the crack
    '''
    kernel = np.ones((2, 2), np.uint8)
    crack_final = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    max_cont = find_max_contour(crack_final)
    crack_hull = cv.convexHull(max_cont)
    # use approxPolyDP()

    return crack_hull


def crack_geo_calc(c, cnt):
    '''
    Calculate the geometric information of the contour
    Input: c = (cx, cy, r); cnt is the contour of crack
    Output: area, depth, side_length of the contour (pixel length)
    '''
    area = cv.contourArea(cnt)

    min_ang = np.arctan2(cnt[0][0][0]-c[0], cnt[0][0][1]-c[1])
    max_ang = min_ang
    min_dis = 1000
    for p in cnt:
        ang = np.arctan2(p[0][0]-c[0], p[0][1]-c[1])
        if (ang < 0):
            ang += 2*np.pi

        # modify the [min,max] range of the angle
        if(ang < min_ang):
            min_ang = ang
        elif(ang > max_ang):
            max_ang = ang

        # find the minimum distance between contour points and center
        dis = np.sqrt((p[0][0]-c[0])*(p[0][0]-c[0]) +
                      (p[0][1]-c[1])*(p[0][1]-c[1]))
        if(dis < min_dis):
            min_dis = dis

    return area, (c[2]-min_dis), c[2]*(max_ang-min_ang)


def main():
    cwd = os.getcwd()  # find the directories of the crack, and list the crack images
    folders = [dir for dir in os.listdir(cwd) if (
        os.path.isdir(dir) and str.isalnum(dir))]

    for f in folders:
        # loop the folders
        dir = os.path.join(cwd, f)
        imgs = os.listdir(dir)
        img_test = cv.imread(os.path.join(dir, imgs[0]), 0)  # image size
        crack_final = np.zeros(img_test.shape, np.uint8)
        cx_stack = []
        cy_stack = []
        r_stack = []
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
            cx_stack.append(cx)
            cy_stack.append(cy)
            r_stack.append(r)

            # apply bitwise operation to obtain the crack object
            crack = cv.bitwise_not(crack, mask=cir_mask)
            # add all the images together
            crack_final = cv.add(crack_final, crack)

        # cv.imshow("Added crack ", crack_final)
        crack_hull = crack_hull_extraction(crack_final)
        (area, depth, side_length) = crack_geo_calc(
            (np.mean(cx_stack), np.mean(cy_stack), np.mean(r_stack)), crack_hull)
        print("Crack-"+f+": area =", area*PIXEL*PIXEL, ", depth =",
              depth*PIXEL, ", side length =", side_length*PIXEL)

        # draw the crack on a blank plate
        blank = np.zeros((736, 736), np.uint8)
        cv.drawContours(blank, [crack_hull], 0, 250, -1)
        cv.imshow("Crack-"+f, blank)
    cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
