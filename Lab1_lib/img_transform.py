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
            sorted_pixels = np.sort(np.abs(dct_img).flatten())
            threshold_value = np.percentile(sorted_pixels, q=int(compression_rate * 100))
            dct_img[np.abs(dct_img) <= threshold_value] = 0
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
            idht_img = Hadamard @ img_float32 @ Hadamard / img.shape[0] / img.shape[1]
            idht_img = np.uint8(np.round(idht_img))
            return idht_img

    raise TypeError('Not such transform category!')


def maxPSNR(img: np.uint8) -> float:
    psnr_1 = cv2.PSNR(img, np.zeros(img.shape[0], img.shape[1]))
    psnr_2 = cv2.PSNR(img, 255 * np.ones(img.shape[0], img.shape[1]))
    return psnr_1 if psnr_1 > psnr_2 else psnr_2


def comparePSNR(dct_dict: dict, dft_dict: dict, dht_dict: dict, psnr: float = 0) -> int:
    dct_list = [key for key, value in dct_dict.items() if value < psnr]
    dft_list = [key for key, value in dft_dict.items() if value < psnr]
    dht_list = [key for key, value in dht_dict.items() if value < psnr]
    if len(dct_list) > 0:
        print(max(dct_list))
    else:
        print('all above')
    if len(dht_list) > 0:
        print(max(dct_list))
    else:
        print('all above')
    if len(dht_list) > 0:
        print(max(dct_list))
    else:
        print('all above')


def plot_psnr(img: np.uint8, trans_category: str = 'DCT', start: int = 1, stop: int = 85) -> dict:
    assert img.shape[0] == img.shape[1]
    if not trans_category in ['DCT', 'DFT', 'DHT']:
        raise TypeError('Not such transform category!')
    plt_x = np.array(range(start, stop, 1))
    plot_line = []
    for cr in range(start, stop, 1):
        cr = 1 - float(cr) / 100
        trans_img = img_transform(img, trans_category=trans_category, compression_rate=cr)
        itrans_img = img_transform(trans_img, trans_category=trans_category, reverse=True)
        plot_line.append(cv2.PSNR(img, itrans_img))
    plt.plot(plt_x, plot_line)
    return {key.tolist(): value for key, value in zip(plt_x, plot_line)}
