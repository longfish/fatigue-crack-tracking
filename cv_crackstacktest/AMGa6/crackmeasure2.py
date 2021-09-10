'''
Compute the crack geometry from a stack of crack images
'''

import heapq as pq
import numpy as np
import crackmeasure as cmeas
import cv2 as cv
import os

PIXEL = 0.0061  # 1 pixel == 0.0061 mm


def main():
    cwd = os.getcwd()
    crackfolders = [dir for dir in os.listdir(cwd) if (
        os.path.isdir(dir) and dir.isalnum())]

    print(crackfolders)
    # loop the crack folders
    for crackf in crackfolders:
        crack_dir = os.path.join(cwd, crackf)
        imgs = os.listdir(crack_dir)  # all crack images

        img_ref = cv.imread(os.path.join(
            crack_dir, imgs[0]), 0)  # image size
        crack_final = np.full(img_ref.shape[:2], 255, dtype=np.uint8)

        # loop the crack images in the current folder and add together
        for img in imgs:
            img_array = cv.imread(os.path.join(crack_dir, img), 0)
            img_array = cmeas.image_filter(img_array, 12, 100)

            # apply bitwise operation to obtain the crack object
            # crack_final = cv.bitwise_and(img_array, crack_final)
            crack_final = cv.bitwise_xor(img_array, crack_final)
            # crack_final = cv.min(img_array, crack_final)

        cv.imshow("Crack final"+crackf, crack_final)
    cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
