import numpy as np
import cv2
from noise import add_sp_noise
from noise import add_gauss_noise


def search_best_param(img: np.uint8, noise_img: np.ndarray, blur_type: str = 'median') -> list:
    last_psnr = 0
    assert img.dtype == noise_img.dtype
    if blur_type == 'median':
        for ksize in range(1, 999, 2):
            blur_img = cv2.medianBlur(src=noise_img, ksize=ksize)
            psnr = cv2.PSNR(img, blur_img)
            if psnr < last_psnr:
                return ksize - 2
            last_psnr = psnr
        raise LookupError('Fail to find the best parameter!')
    elif blur_type == 'blur':
        for ksize in range(1, 999, 2):
            blur_img = cv2.blur(src=noise_img, ksize=(ksize, ksize))
            psnr = cv2.PSNR(img, blur_img)
            if psnr < last_psnr:
                return ksize - 2
            last_psnr = psnr
        raise LookupError('Fail to find the best parameter!')
    raise TypeError('No such blur_type!')


def show_best_img(img: np.uint8, filer_M_img: np.uint8, filer_B_img: np.uint8, extra_str: str = ''):
    psnr_m = cv2.PSNR(img, filer_M_img)
    psnr_b = cv2.PSNR(img, filer_B_img)
    if psnr_m > psnr_b:
        cv2.imshow(extra_str+'Median_Blur',filer_M_img)
    else:
        cv2.imshow(extra_str+'Blur',filer_B_img)

def lab2(img_adress: str = 'imgs/lena.png'):
    img = cv2.imread(img_adress, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img',img)
    # 椒盐噪声
    sp_img = add_sp_noise(img, 0.1, 0.1)
    cv2.imshow('sp_img', sp_img)
    ksize_m = search_best_param(img=img, noise_img=sp_img, blur_type='median')
    ksize_b = search_best_param(img=img, noise_img=sp_img, blur_type='blur')
    filer_M_img = cv2.medianBlur(src=sp_img, ksize=ksize_m)
    filer_B_img = cv2.blur(src=sp_img, ksize=(ksize_b, ksize_b))
    show_best_img(img,filer_M_img,filer_B_img,extra_str='slat_pepper ')
    # 高斯噪声
    gs_img = add_gauss_noise(img, mean=0, sigma=20)
    cv2.imshow('gs_img', gs_img)
    ksize_m = search_best_param(img=img, noise_img=gs_img, blur_type='median')
    ksize_b = search_best_param(img=img, noise_img=gs_img, blur_type='blur')
    filer_M_img = cv2.medianBlur(src=gs_img, ksize=ksize_m)
    filer_B_img = cv2.blur(src=gs_img, ksize=(ksize_b, ksize_b))
    show_best_img(img,filer_M_img,filer_B_img,extra_str='gauss ')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lab2()
    # lab2(img_adress='imgs/cameraman.jpg')
