import numpy as np
import random
import warnings


# 要求指定pepper和salt的概率
def add_sp_noise(img: np.uint8, pepper_probability: float = 0, slat_probability: float = 0) -> np.uint8:
    noisy_img = np.uint8(np.zeros(img.shape, np.uint8))
    if pepper_probability == 0 and slat_probability == 0:
        warnings.warn('No noise will be added to the image.')
        return img
    if pepper_probability + slat_probability > 1:
        warnings.warn('Full image is noise')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < pepper_probability:
                noisy_img[i][j] = 0
            elif rdn > 1-slat_probability:
                noisy_img[i][j] = 255
            else:
                noisy_img[i][j] = img[i][j]
    return noisy_img


def add_gauss_noise(img: np.uint8, mean=0, sigma=1) -> np.uint8:
    noisy_img = np.float32(img) + np.random.normal(mean, sigma, (img.shape[0], img.shape[1]))
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return np.uint8(np.round(noisy_img))
    return noisy_img
