'''
Compute the crack geometry from a stack of crack images
Usage: 
    Put the script into the parent path of all crack images
'''

import heapq as pq
import numpy as np
import cv2 as cv
import os

PIXEL = 0.0061  # 1 pixel == 0.0061 mm


def image_filter(img, h, th_g):
    '''
    Filter the image, apply a global threshold to get it binarized.
    '''

    equ = cv.equalizeHist(img)  # apply global hist equalization
    gaus = cv.GaussianBlur(equ, (9, 9), 0)  # denoising using gaussian filter
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

    # sort using key (contour area)
    contours.sort(key=cv.contourArea, reverse=True)
    return contours


def cracks_extraction(img, c, num=1):
    '''
    Extract the cracks from the binary image (including some noises). 
    Input: 
        c = (cx, cy, r), contour circle of the cross-section
        num, number of cracks to extract
    Output: 
        a list of cracks (max to min area)
    '''
    kernel = np.ones((3, 3), np.uint8)
    # break thin bifurcations
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # fill small holes
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cnts = contour_sort_area(img)

    crack_poly = []
    k = 0
    for cnt in cnts:
        # check if the contour is a pore
        if(not is_pore(cnt, c)):
            crack_poly.append(cnt)
            k += 1
        if(k >= num):
            break

    return crack_poly


def is_pore(cnt, c):
    '''
    Determine of the contour is a internal pore
    Input:
        cnt, contour of the object
        c = (cx, cy, r), contour circle of the cross-section
    '''
    side_length = np.sqrt(cv.contourArea(cnt))
    M = cv.moments(cnt)
    centx = int(M['m10']/M['m00'])  # centroid
    centy = int(M['m01']/M['m00'])
    depth = c[2] - np.sqrt((centx-c[0])*(centx-c[0])+(centy-c[1])*(centy-c[1]))

    # internal pores criteria
    if(depth > 2*side_length and side_length < 0.1*c[2]):
        return True
    else:
        return False


def calc_crack_geo(c, cnt):
    '''
    Calculate the geometric information of the contour (cnt)
    Note: only valid if the crack doesn't contain center point and depth < r
    Input: 
        c = (cx, cy, r)
        cnt, the contour of crack
    Output: area, depth, side_length of the contour (pixel length)
    '''
    area = cv.contourArea(cnt)

    # find the minimum distance between contour points and center using heap queue
    dis_arr = []
    for p in cnt:
        dis = np.sqrt((p[0][0]-c[0])*(p[0][0]-c[0]) +
                      (p[0][1]-c[1])*(p[0][1]-c[1]))
        dis_arr.append(dis)
    dis_smallest = pq.nsmallest(int(len(dis_arr)/4), dis_arr)
    min_dis = np.mean(dis_smallest)

    # get the convex hull of the crack
    hull = cv.convexHull(cnt)
    # draw the crack on a blank plate
    # blank = np.zeros((736, 736), np.uint8)
    # cv.drawContours(blank, [hull], 0, 250, -1)
    # cv.imshow("Hull", blank)

    # find the central angle of the crack hull
    max_ang = 0
    for p in hull:
        vec1 = np.array([p[0][0]-c[0], p[0][1]-c[1]])
        for q in hull:
            vec2 = np.array([q[0][0]-c[0], q[0][1]-c[1]])
            ang = calc_angle(vec1, vec2)
            if(ang > max_ang):
                max_ang = ang

    return area, (c[2]-min_dis), c[2]*max_ang


def calc_angle(vec1, vec2):
    '''
    Determine the angle between vectors vec1 and vec2
    Input: vec1, vec2 are np.array
    '''
    cos = vec1.dot(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def rotate2y(cnt, c):
    '''
    Rotate the contour with cernter c, make its centroid on the y-axis
    Input:
        cnt, (a numpy ndarray), the contour 
        c, (cx, cy), the center coordinate
    '''
    M = cv.moments(cnt)
    centx = int(M['m10']/M['m00'])  # centroid of the contour
    centy = int(M['m01']/M['m00'])
    v1 = np.array([centx-c[0], centy-c[1]])
    v2 = np.array([0, 1])
    r_ang = calc_angle(v1, v2)
    if(centx-c[0] < 0):
        r_ang = -r_ang

    # rotate the contour
    # print(r_ang*180/np.pi)
    M = np.array([[np.cos(r_ang), -np.sin(r_ang)],
                  [np.sin(r_ang), np.cos(r_ang)]])  # rotation matrix
    # print(M)
    cnt_rot = []
    for p in cnt:
        p_rot = M.dot(np.array([p[0][0]-c[0], p[0][1]-c[1]]))
        cnt_rot.append([[int(p_rot[0]+c[0]), int(p_rot[1]+c[1])]])
    return np.array(cnt_rot)


def translate(cnt, c1, c2):
    '''
    Translate the contour from center c1 to center c2
    '''
    cnt_trans = []
    for p in cnt:
        cnt_trans.append(
            [[int(p[0][0]+c2[0]-c1[0]), int(p[0][1]+c2[1]-c1[0])]])
    return np.array(cnt_trans)


def main():
    crack_info = open('crack_info.txt', 'w')
    cwd = os.getcwd()  # find the directories of the crack, and list the crack images
    stepfolders = [dir for dir in os.listdir(cwd) if (
        os.path.isdir(dir))]

    crack_info.write("Crack: Area, Depth, Side length\n")
    # loop the loading step folders
    for stepf in stepfolders:
        step_dir = os.path.join(cwd, stepf)
        crackfolders = [dir for dir in os.listdir(step_dir)]

        # loop the cracks folders
        for crackf in crackfolders:
            crack_dir = os.path.join(step_dir, crackf)
            imgs = os.listdir(crack_dir)
            img_ref = cv.imread(os.path.join(
                crack_dir, imgs[0]), 0)  # image size
            crack_final = np.zeros(img_ref.shape[:2], np.uint8)
            cx_stack = []
            cy_stack = []
            r_stack = []

            # loop the crack images in the current folder
            for img in imgs:
                img_array = cv.imread(os.path.join(crack_dir, img), 0)
                crack = image_filter(img_array, 12, 98)
                # cv.imshow(img, crack)

                cx, cy, r = cross_section_contour(
                    img_array, 14)  # get the contour circle
                # create a circle mask, store the circle size
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
            crack_hub = cracks_extraction(crack_final, (np.mean(
                cx_stack), np.mean(cy_stack), np.mean(r_stack)), crack_num)
            if(len(crack_hub) == 0):
                crack_hub = np.array([])  # no crack detected
            for i, c in enumerate(crack_hub):
                (area, depth, side_length) = calc_crack_geo(
                    (np.mean(cx_stack), np.mean(cy_stack), np.mean(r_stack)), c)
                crack_info.write(stepf+"-"+crackf+"-"+str(i)+": "+str(area*PIXEL*PIXEL)+", " +
                                 str(depth*PIXEL)+", "+str(side_length*PIXEL)+"\n")
                print(stepf+"-"+crackf+"-"+str(i)+": "+str(area*PIXEL*PIXEL)+", " +
                      str(depth*PIXEL)+", "+str(side_length*PIXEL)+"\n")

                # rotate the crack on the y-axis, then translate
                c_rot = rotate2y(c, (np.mean(cx_stack), np.mean(cy_stack)))
                c_trans = translate(
                    c_rot, (np.mean(cx_stack), np.mean(cy_stack)), (img_ref.shape[0]/2, img_ref.shape[0]/2))
                # draw the crack on a blank plate
                blank = np.zeros(img_ref.shape[:2], np.uint8)
                cv.drawContours(blank, [c_trans], 0, 250, -1)
                cv.imwrite(stepf+"-"+crackf+"-"+str(i)+".png", blank)
    cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
