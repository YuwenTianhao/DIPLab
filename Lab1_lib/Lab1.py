import cv2
import numpy as np
from Lab1_lib.img_transform import img_transform
from Lab1_lib.img_transform import plot_psnr
from Lab1_lib.img_transform import comparePSNR
from Lab1_lib.img_transform import plt_spectrogram
import matplotlib.pyplot as plt
import time


def trans_mission(img: np.uint8, trans_category: str, compression_rate: float = 0):
    # 无压缩情况绘制频谱图
    trans_img = img_transform(img, trans_category=trans_category)
    plt_spectrogram(trans_img=trans_img)
    print(trans_category+' Result :')
    # 压缩情况
    trans_img = img_transform(img, trans_category=trans_category, compression_rate=compression_rate)
    start_time = time.time()
    i_img = img_transform(trans_img, trans_category=trans_category, reverse=True)
    end_time = time.time()
    print('Use time :' + str(end_time - start_time) + 's')
    print('PSNR :' + str(cv2.PSNR(img, i_img)))
    cv2.imshow(trans_category, i_img)


def lab1(img_adress: str = 'imgs/lena.png'):
    img = cv2.imread(img_adress, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('lena',img)
    cr = 0.95  # compression_rate
    psnr = 28  # 设置之后对比用的psnr
    # DCT
    trans_mission(img, 'DCT', compression_rate=cr)
    # DFT
    trans_mission(img, 'DFT', compression_rate=cr)
    # DHT
    trans_mission(img, 'DHT', compression_rate=cr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 比较同样PSNR下的压缩率
    comparePSNR(img=img, trans_category='DCT', psnr=psnr)
    comparePSNR(img=img, trans_category='DFT', psnr=psnr)
    comparePSNR(img=img, trans_category='DHT', psnr=psnr)
    # 隐藏任务：绘制 PSNR 与压缩率的图像，非常耗时间
    # dct_dict = plot_psnr(img, 'DCT')
    # dft_dict = plot_psnr(img, 'DFT')
    # dht_dict = plot_psnr(img, 'DHT')
    # plt.show()

if __name__ == '__main__':
    lab1()
    # lab1(img_adress='imgs/cameraman.jpg')
