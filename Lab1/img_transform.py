import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

def img_transform(img: np.uint8, trans_category: str = 'DCT', reverse: bool = False,
                  compression_rate: float = 1.0) -> np.uint8:
    assert img.shape[0] == img.shape[1]
    if compression_rate > 1 or compression_rate < 0:
        raise ValueError('Wrong compression_rate!')
    # 将图像转换为 float32 类型
    img_float32 = np.float32(img)
    if trans_category == 'DCT':
        if not reverse:
            # 进行离散余弦变换
            dct_img = cv2.dct(img_float32)
            # cv2.imshow('DCT_beforeTH', dct_img)
            sorted_pixels = np.sort(dct_img.flatten())
            threshold_value = np.percentile(sorted_pixels, q=int(compression_rate * 100))
            dct_img[dct_img <= threshold_value] = 0
            # cv2.imshow('DCT', dct_img)
            return dct_img
        if reverse:
            idct_img = np.uint8(np.round(cv2.idct(img_float32)))  # img -> iDCT -> round() -> uint8 -> imshow
            return idct_img
    elif trans_category == 'DFT':
        if not reverse:
            # 傅里叶变换
            dft_img = cv2.dft(img_float32, flags=cv2.DFT_REAL_OUTPUT)
            dft_img_complex = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
            sorted_pixels = np.sort(dft_img.flatten())
            threshold_value = np.percentile(sorted_pixels, q=int(compression_rate * 100))
            # 遍历实部和虚部
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # 如果和小于阈值，将实部和虚部置为0
                    if dft_img[i, j] < threshold_value:
                        dft_img_complex[i, j, 0] = 0
                        dft_img_complex[i, j, 1] = 0
            return dft_img_complex
        if reverse:
            idft_img = cv2.idft(img_float32, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            idft_img = np.uint8(np.round(idft_img))
            return idft_img
    elif trans_category == 'DHT':
        Hadamard = hadamard(img.shape[0])
        if not reverse:
            dht_img = Hadamard @ img_float32 @ Hadamard
            sorted_pixels = np.sort(np.abs(dht_img).flatten())
            threshold_value = np.percentile(sorted_pixels, q=int(compression_rate * 100))
            dht_img[np.abs(dht_img) <= threshold_value] = 0
            return dht_img
        if reverse:
            idht_img = Hadamard@img_float32@ Hadamard/ img.shape[0]/img.shape[1]
            idht_img = np.uint8(np.round(idht_img))
            return idht_img

    raise TypeError('Not such transform category!')

def Lab1(img_adress:str='imgs/lena.png'):
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
    psnr = [cv2.PSNR(img, idct_img), cv2.PSNR(img, idft_img), cv2.PSNR(img, idht_img)]
    # 显示原始图像和变换后的图像
    cv2.imshow('Original', img)
    cv2.imshow('iDCT_img', idct_img)
    cv2.imshow('iDFT_img', idft_img)
    cv2.imshow('iDHT_img', idht_img)
    print(psnr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Lab1()
    # Lab1(img_adress='imgs/cameraman.jpg')