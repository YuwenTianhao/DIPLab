import numpy as np
import random


def add_sp_noise(img: np.uint8, probability=0)->np.uint8:
    noisy_img = np.uint8(np.zeros(img.shape, np.uint8))
    th_high = 1 - probability/2
    th_low = probability/2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < th_low:
                noisy_img[i][j] = 0
            elif rdn > th_high:
                noisy_img[i][j] = 255
            else:
                noisy_img[i][j] = img[i][j]
    return noisy_img


def add_gauss_noise(img: np.uint8, mean=0, sigma=1) -> np.float32:
    noisy_img = np.float32(img) + np.random.normal(mean, sigma, (img.shape[0], img.shape[1]))
    noisy_img[noisy_img<0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img
