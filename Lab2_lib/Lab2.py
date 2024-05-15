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


def show_best_img(img: np.uint8, filer_M_img: np.uint8, filer_B_img: np.uint8, extra_str: str = '') -> None:
    psnr_m = cv2.PSNR(img, filer_M_img)
    psnr_b = cv2.PSNR(img, filer_B_img)
    if psnr_m > psnr_b:
        cv2.imshow(extra_str + 'Median_Blur', filer_M_img)
    else:
        cv2.imshow(extra_str + 'Blur', filer_B_img)


# 按实验要求，更新了一个专门输出实验结果的函数，但是不建议作为API留用
def change_ksize_and_output(img: np.uint8, noise_img: np.uint8, starts_size: int, end_size: int):
    psnr_dict_mb = {'3': 0, '5': 0, '7': 0}
    psnr_dict_b = {'3': 0, '5': 0, '7': 0}
    assert img.dtype == noise_img.dtype
    if starts_size % 2 == 0 or end_size % 2 == 0:
        raise ValueError('Wrong kernel size input!')
    for ksize in range(starts_size, end_size + 1, 2):
        filter_M_img = cv2.medianBlur(src=noise_img, ksize=ksize)
        filter_B_img = cv2.blur(src=noise_img, ksize=(ksize, ksize))
        # cv2.imshow('ksize=' + str(ksize) + '_medianBlur', filter_M_img)
        # cv2.imshow('ksize=' + str(ksize) + '_blur', filter_B_img)
        psnr_dict_mb[str(ksize)] = cv2.PSNR(img, filter_M_img)
        psnr_dict_b[str(ksize)] = cv2.PSNR(img, filter_B_img)
    print('MedianBlur:' + str(psnr_dict_mb))
    print('Blur:' + str(psnr_dict_b))


def lab2(img_adress: str = 'imgs/lena.png'):
    img = cv2.imread(img_adress, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', img)
    # 椒盐噪声
    sp_img = add_sp_noise(img, 0.05, 0.05)
    cv2.imshow('sp_img', sp_img)
    # print('SP_img PSNR result')
    # change_ksize_and_output(img=img, noise_img=sp_img, starts_size=3, end_size=7)
    # 高斯噪声
    gs_img = add_gauss_noise(img, mean=0, sigma=20)
    cv2.imshow('gs_img', gs_img)
    # print('GS_img PSNR result')
    # change_ksize_and_output(img=img, noise_img=sp_img, starts_size=3, end_size=7)
    ksize_m = search_best_param(img=img, noise_img=sp_img, blur_type='median')
    ksize_b = search_best_param(img=img, noise_img=sp_img, blur_type='blur')
    filer_M_img = cv2.medianBlur(src=sp_img, ksize=ksize_m)
    filer_B_img = cv2.blur(src=sp_img, ksize=(ksize_b, ksize_b))
    show_best_img(img, filer_M_img, filer_B_img, extra_str='slat_pepper')

    ksize_m = search_best_param(img=img, noise_img=gs_img, blur_type='median')
    ksize_b = search_best_param(img=img, noise_img=gs_img, blur_type='blur')
    filer_M_img = cv2.medianBlur(src=gs_img, ksize=ksize_m)
    filer_B_img = cv2.blur(src=gs_img, ksize=(ksize_b, ksize_b))
    show_best_img(img, filer_M_img, filer_B_img, extra_str='gauss ')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lab2()
    # lab2(img_adress='imgs/cameraman.jpg')
