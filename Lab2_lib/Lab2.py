import numpy as np
import cv2
from noise import add_sp_noise
from noise import add_gauss_noise
from Lab1_lib.img_transform import float32img_output


def lab2(img_adress: str = 'imgs/lena.png'):
    img = cv2.imread(img_adress, cv2.IMREAD_GRAYSCALE)
    sp_img = add_sp_noise(img,0.1)
    gs_img = add_gauss_noise(img,mean=0,sigma=20)
    sp_mb = cv2.medianBlur(src=sp_img, ksize=3)
    # sp_blur = cv2.blur(src=sp_img, ksize=3)

    gs_mb = cv2.medianBlur(src=gs_img, ksize=3)
    # gs_blur = cv2.blur(src=gs_img, ksize=3)
    cv2.imshow('sp_mb',sp_mb)
    # cv2.imshow('sp_blur', float32img_output(sp_blur))
    cv2.imshow('gs_mb', gs_mb)
    # cv2.imshow('gs_blur', float32img_output(gs_blur))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lab2()
    # lab2(img_adress='imgs/cameraman.jpg')