'''
Compute the crack geometry from a stack of crack images

Usage:
  Put the script into the parent path of all crack images (for one sample)
'''

import numpy as np
import cv2 as cv
import os

PIXEL = 0.0061  # 1 pixel == 0.0061 mm


def image_filter(img, h, th_g):
    '''
    Filter the image, apply a global threshold to get it binarized.
    '''

    equ = cv.equalizeHist(img)  # apply global hist equalization
    gaus = cv.GaussianBlur(equ, (5, 5), 0)  # denoising using gaussian filter
    # denoising using nonlocal means algorithm
    dst1 = cv.fastNlMeansDenoising(gaus, None, h, 7, 21)

    # apply Otsu's binarization (may be more stable)
    # _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # apply global threshhold (may be more accurate)
    _, th1 = cv.threshold(dst1, th_g, 255, cv.THRESH_BINARY)

    return th1


def cross_section_contour(img, little):
    '''
    Get the cross-section contour of the (original) image.
    Output: 
        The centroid coordinates and radius of the contour circle (reduced by a little).
    '''

    # create an intact cross-section
    img_blur = cv.GaussianBlur(img, (15, 15), 0)
    _, ostu2 = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # find the contour of the cross-section (largest one)
    cnt = contour_sort_area(ostu2)
    M = cv.moments(cnt[0])
    cx = int(M['m10']/M['m00'])  # centroid
    cy = int(M['m01']/M['m00'])
    _, radius = cv.minEnclosingCircle(cnt[0])

    return cx, cy, int(radius-little)


def contour_sort_area(img):
    '''
    Sort the contours by the area, from max to min.
    '''
    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # sort using key
    contours.sort(key=cv.contourArea, reverse=True)
    return contours


def cracks_extraction(img, num=1):
    '''
    Extract the cracks from the binary image (including some noises).
    Input: num, number of cracks to extract
    Output: 
        a list of cracks (max to min area)
    '''
    kernel = np.ones((3, 3), np.uint8)
    # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cnts = contour_sort_area(img)

    crack_poly = []
    for i in range(num):
        # diff = eps*cv.arcLength(cnts[i], True)
        # crack_poly.append(cv.approxPolyDP(cnts[i], diff, True))
        crack_poly.append(cnts[i])

    return crack_poly


def crack_geo_calc(c, cnt):
    '''
    Calculate the geometric information of the contour (cnt)
    Note: only valid if the crack doesn't contain center point and depth < r
    Input: 
        c = (cx, cy, r)
        cnt, the contour of crack
    Output: area, depth, side_length of the contour (pixel length)
    '''
    area = cv.contourArea(cnt)

    min_dis = c[2]
    max_ang = 0
    for p in cnt:
        # find the minimum distance between contour points and center
        dis = np.sqrt((p[0][0]-c[0])*(p[0][0]-c[0]) +
                      (p[0][1]-c[1])*(p[0][1]-c[1]))
        if(dis < min_dis):
            min_dis = dis

    # get the convex hull of the crack
    hull = cv.convexHull(cnt)
    # draw the crack on a blank plate
    # blank = np.zeros((736, 736), np.uint8)
    # cv.drawContours(blank, [hull], 0, 250, -1)
    # cv.imshow("Hull", blank)

    # find the central angle of the crack hull
    for p in hull:
        vec1 = [p[0][0]-c[0], p[0][1]-c[1]]
        for q in hull:
            vec2 = [q[0][0]-c[0], q[0][1]-c[1]]
            ang = find_angle(vec1, vec2)
            if(ang > max_ang):
                max_ang = ang

    return area, (c[2]-min_dis), c[2]*max_ang


def find_angle(vec1, vec2):
    '''
    Find the angle between vectors vec1 and vec2
    Input: vec1, vec2 = [x1, x2]
    '''
    unit_vec1 = vec1/np.linalg.norm(vec1)
    unit_vec2 = vec2/np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    return np.arccos(dot_product)


def main():
    cwd = os.getcwd()  # find the directories of the crack, and list the crack images
    folders = [dir for dir in os.listdir(cwd) if (
        os.path.isdir(dir) and str.isalnum(dir))]

    for f in folders:
        # loop the folders
        dir = os.path.join(cwd, f)
        imgs = os.listdir(dir)
        img_test = cv.imread(os.path.join(dir, imgs[0]), 0)  # image size
        crack_final = np.zeros(img_test.shape[:2], np.uint8)
        cx_stack = []
        cy_stack = []
        r_stack = []
        for img in imgs:
            # loop the images in the crack folder
            img_dir = os.path.join(dir, img)
            img_array = cv.imread(img_dir, 0)
            crack = image_filter(img_array, 12, 100)
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
        crack_num = 1
        crack_hub = cracks_extraction(crack_final, crack_num)
        for c in crack_hub:
            (area, depth, side_length) = crack_geo_calc(
                (np.mean(cx_stack), np.mean(cy_stack), np.mean(r_stack)), c)
            print("Crack-"+f+": area =", area*PIXEL*PIXEL, ", depth =",
                  depth*PIXEL, ", side length =", side_length*PIXEL)

            # draw the crack on a blank plate
            blank = np.zeros(img_test.shape[:2], np.uint8)
            cv.drawContours(blank, [c], 0, 250, -1)
            cv.imshow("Crack-"+f, blank)
    cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
