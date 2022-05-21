import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from time import time
from skimage.feature import plot_matches
from skimage.transform import pyramid_gaussian
import cv2
from utils import *
import matplotlib
import numpy as np
from scipy.signal import convolve2d
import sys

np.set_printoptions(threshold=sys.maxsize)
matplotlib.use('TKAgg')


# ---------------------------------------------------------


def FAST(img, N=9, threshold=0.15, nms_window=2):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16

    img = convolve2d(img, kernel, mode='same')

    cross_idx = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
    circle_idx = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                           [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])

    corner_img = np.zeros(img.shape)
    keypoints = []
    for y in range(3, img.shape[0] - 3):
        for x in range(3, img.shape[1] - 3):
            if 348 < x < 454 and 348 < y < 454:
                continue
            Ip = img[y, x]
            t = threshold * Ip if threshold < 1 else threshold
            # fast checking cross idx only
            if np.count_nonzero(Ip + t < img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3 or np.count_nonzero(
                    Ip - t > img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3:
                # detailed check -> full circle
                if np.count_nonzero(img[y + circle_idx[0, :], x + circle_idx[1, :]] >= Ip + t) >= N or np.count_nonzero(
                        img[y + circle_idx[0, :], x + circle_idx[1, :]] <= Ip - t) >= N:
                    # Keypoint [corner]
                    keypoints.append([x, y])  # Note: keypoint = [col, row]
                    corner_img[y, x] = np.sum(np.abs(Ip - img[y + circle_idx[0, :], x + circle_idx[1, :]]))

    # NMS - Non Maximal Suppression
    if nms_window != 0:
        fewer_kps = []
        for [x, y] in keypoints:
            window = corner_img[y - nms_window:y + nms_window + 1, x - nms_window:x + nms_window + 1]
            # v_max = window.max()
            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps)


def base(image):
    # ---------------------------------------------------------
    # Trying multi-scale
    N_LAYERS = 1
    DOWNSCALE = 2

    img1 = image

    img1 = Image.fromarray(img1)
    temp = img1.convert("L")
    gray1 = img1 = img1.convert('L')
    img1 = gray1 = np.array(img1)
    grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

    scales = [DOWNSCALE ** i for i in range(N_LAYERS)]
    print('scales: ', scales, '\n')
    features_img1 = img1
    kps1 = []
    ds1 = []
    for i in range(N_LAYERS):
        print('pyramid layer: ', i)
        print('scales[i]: ', scales[i])
        scale_coeff1 = (gray1.shape[1] / grays1[i].shape[1], gray1.shape[0] / grays1[i].shape[0])
        print('scale_coeff1: ', scale_coeff1)
        print('grays1[i] shape: ', grays1[i].shape)

        scale_kp1 = FAST(grays1[i], N=9, threshold=0.1, nms_window=3)
        print('kp1: ', len(scale_kp1))
        features_img1 = cv2.cvtColor(features_img1, cv2.COLOR_GRAY2RGB)
        for keypoint in scale_kp1:
            x0 = np.round(keypoint * scale_coeff1)[0] - 5 * scales[i]
            y0 = np.round(keypoint * scale_coeff1)[1] - 5 * scales[i]
            x1 = np.round(keypoint * scale_coeff1)[0] + 5 * scales[i]
            y1 = np.round(keypoint * scale_coeff1)[1] + 5 * scales[i]
            color1 = (0, 255, 0)
            features_img1 = cv2.rectangle(features_img1, (int(x0), int(y0)), (int(x1), int(y1)),
                                          (int(0), int(0), int(255)), 0)
    features_img1 = cv2.cvtColor(features_img1, cv2.COLOR_BGR2RGB)
    x=[]
    y=[]
    intensity=[]
    for each in scale_kp1:
        x.append(each[0])
        y.append(each[1])
        intensity.append(image[each[1],each[0]])
    df = pd.DataFrame(list(zip(x, y, intensity)), columns=["x","y","intensity"])
    return [features_img1, len(scale_kp1), df]


# base("../annotations/miss_mean")


# plt.figure()
# plt.imshow(features_img1)
# plt.show()

