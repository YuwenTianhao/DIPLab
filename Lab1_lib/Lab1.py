import cv2
from Lab1_lib.img_transform import img_transform
from Lab1_lib.img_transform import plot_psnr
from Lab1_lib.img_transform import comparePSNR
import matplotlib.pyplot as plt


def lab1(img_adress: str = 'imgs/lena.png'):
    img = cv2.imread(img_adress, cv2.IMREAD_GRAYSCALE)
    cr = 0.95  # compression_rate
    # DCT
    dct_img = img_transform(img, trans_category='DCT', compression_rate=cr)
    idct_img = img_transform(dct_img, trans_category='DCT', reverse=True)
    # DFT
    dft_img = img_transform(img, trans_category='DFT', compression_rate=cr)
    idft_img = img_transform(dft_img, trans_category='DFT', reverse=True)
    # DHT
    dht_img = img_transform(img, trans_category='DHT', compression_rate=cr)
    idht_img = img_transform(dht_img, trans_category='DHT', reverse=True)
    psnr = {'DCT': cv2.PSNR(img, idct_img), 'DFT': cv2.PSNR(img, idft_img), 'DHT': cv2.PSNR(img, idht_img)}
    # 显示原始图像和变换后的图像
    cv2.imshow('Original', img)
    cv2.imshow('iDCT_img', idct_img)
    cv2.imshow('iDFT_img', idft_img)
    cv2.imshow('iDHT_img', idht_img)
    print('PSNR result:')
    print(psnr)

    # 绘制 PSNR 与压缩率的图像，非常耗时间
    # dct_dict = plot_psnr(img, 'DCT')
    # dft_dict = plot_psnr(img, 'DFT')
    # dht_dict = plot_psnr(img, 'DHT')
    # 比较同样PSNR下的压缩率
    # comparePSNR(dct_dict=dct_dict,dft_dict=dft_dict,dht_dict=dht_dict,psnr=16)
    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lab1()
    # lab1(img_adress='imgs/cameraman.jpg')
