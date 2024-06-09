import cv2
import numpy as np

import Lab1_lib
from Lab1_lib.img_transform import img_transform
from Lab1_lib.img_transform import plot_psnr
from Lab1_lib.img_transform import comparePSNR
from Lab1_lib.img_transform import plt_spectrogram
import matplotlib.pyplot as plt
import time


# 对于压缩情况的任务
def trans_mission(img: np.uint8, compression_rate: float = 0):
    print('Compare in same compression rate')
    plt.subplot(2,3,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image',fontsize='small')
    plt.axis('off')
    plt.subplot(2,3,2)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image',fontsize='small')
    plt.axis('off')
    plt.subplot(2,3,3)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image',fontsize='small')
    plt.axis('off')
    figure = 4
    for trans_cate in Lab1_lib.TRANS_CATEGORY:
        cate = 'DCT'
        if trans_cate == Lab1_lib.TRANS_CATEGORY_DFT:
            cate = 'DFT'
        elif trans_cate == Lab1_lib.TRANS_CATEGORY_DHT:
            cate = 'DHT'
        plt.subplot(2,3,figure)
        trans_img = img_transform(img, trans_category=trans_cate, compression_rate=compression_rate)
        start_time = time.time()
        i_img = img_transform(trans_img, trans_category=trans_cate, reverse=True)
        end_time = time.time()
        print('Transform Category: '+cate)
        print('Use time :' + str(end_time - start_time) + 's')
        print('PSNR :' + str(cv2.PSNR(img, i_img)))
        plt.imshow(i_img,cmap='gray')
        plt.title(cate+'Inverse img',fontsize='small')
        plt.axis('off')
        figure += 1


def lab1(img_address: str = 'imgs/lena.png'):
    img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)
    cr = 0.95  # compression_rate
    psnr = 28  # 设置之后对比用的psnr

    # 无压缩情况绘制频谱图
    plt.figure(1)
    plt.suptitle('Spectrum')
    plt_spectrogram(img)
    plt.show()

    # 压缩情况绘图
    plt.figure(2)
    plt.suptitle('Inverse Image with compression rate: '+str(cr*100)+'%')
    trans_mission(img, compression_rate=cr)
    plt.show()
    # 比较同样PSNR下的压缩率
    print('comparePSNR:')
    comparePSNR(img=img, trans_category=Lab1_lib.TRANS_CATEGORY_DCT, psnr=psnr)
    comparePSNR(img=img, trans_category=Lab1_lib.TRANS_CATEGORY_DFT, psnr=psnr)
    comparePSNR(img=img, trans_category=Lab1_lib.TRANS_CATEGORY_DHT, psnr=psnr)


if __name__ == '__main__':
    lab1()
    # lab1(img_adress='imgs/cameraman.jpg')
