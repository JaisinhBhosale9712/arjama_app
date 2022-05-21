import datetime
import pdb
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from numba import cuda, jit
from numba import prange
import matplotlib.patches as patches


def modified_lambda_(image):
    image = image # Load your images# 84, multiply by 4 when blue, 123

    # ============================================================
    def draw_rectangel(x1, y1, x2, y2):
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0, 0),2)
        return image

    #image = Image.fromarray(image)
    # ==============================================================
    try:
        if image[0][1]*0==0:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    except:
        print("")
    w = image.shape[0] #define W and H
    h = image.shape[1]
    pix_flag = np.zeros((w * h, 5), dtype=np.int32)
    T = w
    L = 100


    @jit(nopython=True,parallel=True)
    def createpix(T,h,w,pix_flag):
        for y in prange(0, h-4):
            for x in prange(0, w-4):
                pix_flag[(T * y) + x, 0] = (T * y) + x  # set pixel id
                pix_flag[(T * y) + x, 1] = 0  # class
                pix_flag[(T * y) + x, 2] = x  # X position
                pix_flag[(T * y) + x, 3] = y  # Y position
                pix_flag[(T * y) + x, 4] = 0  # segment id
    f1time = []
    a1 = time.time()
    createpix(T,h,w,pix_flag)


    @jit(nopython=True)
    def checkpixels(image,T,h,w,pix_flag):
        L = 100
        for y in range(2, h - 4):
            for x in range(2, w - 4):

                if pix_flag[(T * y) + x, 4] == 0:
                    pix_flag[(T * y) + x, 4] = 1

                    RGB = image[y,x]
                    R, G, B = RGB

                    left = image[y, x-1]
                    up = image[y-1, x]
                    # down = image.getpixel((x,y+1))
                    leftUp = image[y-1, x-1]
                    rightUp = image[y-1, x+1]


                    if B < 200:

                        if left[2] < 200:
                            L = pix_flag[(T * y) + (x - 1), 1]
                            pix_flag[(T * y) + (x), 1] = L

                        elif up[2] < 200:
                            L = pix_flag[(T * (y - 1)) + (x), 1]
                            pix_flag[(T * y) + (x), 1] = L

                        elif leftUp[2] < 200:
                            L = pix_flag[(T * (y - 1)) + (x - 1), 1]
                            pix_flag[(T * y) + (x), 1] = L

                        elif rightUp[2] < 200:
                            L = pix_flag[(T * (y - 1)) + (x + 1), 1]
                            pix_flag[(T * y) + (x), 1] = L

                        else:
                            pix_flag[(T * y) + (x), 1] = L
                            # pix_flag[(T*y)+(x),4]=1

                    L = L + 1

    dat_list = []
    cluster_list = []
    checkpixels(np.array(image), T, h, w, pix_flag)


    def checkpoints(dat_list,cluster_list,pix_flag):
        for s in prange(0, len(pix_flag)):
            if pix_flag[s][1] != 0:
                dat_list.append(pix_flag[s])
                cluster_list.append(pix_flag[s][1])
    cluster_id = []

    checkpoints(dat_list, cluster_list, pix_flag)

    def filter_id(cluster_list, cluster_id):
        for x in cluster_list:
            if x not in cluster_id:
                cluster_id.append(x)
    filter_id(cluster_list, cluster_id)
    pos = np.zeros((len(cluster_id), 4), dtype=np.int32)

    @jit(nopython=True)
    def clusterform(cluster_id, pix_flag, pos):
        for c in prange(0, len(cluster_id)):
            c_id = cluster_id[c]
            for si in prange(0, len(pix_flag)):
                if pix_flag[si][1] == c_id:
                    pos[c][0] = pix_flag[si][2]
                    pos[c][1] = pix_flag[si][3]
                    break

    clusterform(np.array(cluster_id), pix_flag, pos)
    image = np.array(image)
    # -------------------------- plot rectangels -----------------
    for i in range(0, len(pos)):
        patch = draw_rectangel(pos[i][0] - 5, pos[i][1] - 5, pos[i][0] + 10, pos[i][1] + 10)

    x = []
    y = []
    intensity = []
    for each in pos:
        x.append(each[0])
        y.append(each[1])
        intensity.append(image[each[0], each[1]])
    df = pd.DataFrame(list(zip(x,y,intensity)), columns=["x","y","intensity"])
    return [image,len(pos),df]




