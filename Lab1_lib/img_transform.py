import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard


def img_transform(img: np.ndarray, trans_category: str = 'DCT', reverse: bool = False,
                  compression_rate: float = 0.0) -> np.uint8:
    assert img.shape[0] == img.shape[1]
    if compression_rate > 1 or compression_rate < 0:
        raise ValueError('Wrong compression_rate!')
    # 将图像转换为 float32 类型
    if not np.iscomplexobj(img):
        img_float32 = np.float32(img)
    if trans_category == 'DCT':
        if not reverse:
            # 进行离散余弦变换
            dct_img = cv2.dct(img_float32)
            # cv2.imshow('DCT_beforeTH', dct_img)
            sorted_pixels = np.sort(np.abs(dct_img).flatten())
            threshold_value = np.percentile(sorted_pixels, q=(compression_rate * 100))
            dct_img[np.abs(dct_img) <= threshold_value] = 0
            # cv2.imshow('DCT', dct_img)
            return dct_img
        if reverse:
            idct_img = np.uint8(np.round(cv2.idct(img_float32)))  # img -> iDCT -> round() -> uint8 -> imshow
            return idct_img
    elif trans_category == 'DFT':
        if not reverse:
            # 傅里叶变换
            dft_img = np.fft.fft2(img)
            sorted_pixels = np.sort(np.abs(dft_img).flatten())
            threshold_value = np.percentile(sorted_pixels, q=(compression_rate * 100))
            # 遍历实部和虚部
            dft_img[np.abs(dft_img) <= threshold_value] = 0
            return dft_img
        if reverse:
            idft_img = np.fft.ifft2(img)
            idft_img = np.abs(idft_img)
            idft_img = np.uint8(np.round(idft_img))
            return idft_img
    elif trans_category == 'DHT':
        Hadamard = hadamard(img.shape[0])
        if not reverse:
            dht_img = Hadamard @ img_float32 @ Hadamard
            sorted_pixels = np.sort(np.abs(dht_img).flatten())
            threshold_value = np.percentile(sorted_pixels, q=(compression_rate * 100))
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


def comparePSNR(img: np.uint8, trans_category: str = 'DCT', psnr: float = 0) -> None:
    # 在 0.9 -> 1.0 里寻找可能的点，步长为 0.001，可以自行修改
    for i in range(900, 1001):
        t_img = img_transform(img, trans_category, compression_rate=float(i / 1000))
        i_img = img_transform(t_img, trans_category, reverse=True)
        if cv2.PSNR(img, i_img) < psnr:
            print(trans_category+' non zero element:'+str(np.count_nonzero(t_img)))
            return
    return


def plt_spectrogram(trans_img: np.ndarray)->None:
    np.seterr(divide='ignore')
    fshift = np.fft.fftshift(trans_img)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return

def plot_psnr(img: np.uint8, trans_category: str = 'DCT', start: int = 0, stop: int = 100) -> dict:
    assert img.shape[0] == img.shape[1]
    if not trans_category in ['DCT', 'DFT', 'DHT']:
        raise TypeError('Not such transform category!')
    if start < 0 or stop > 100:
        raise ValueError
    plt_x = np.array(range(start, stop, 1))
    plot_line = []
    for cr in range(start, stop, 1):
        cr = float(cr) / 100
        trans_img = img_transform(img, trans_category=trans_category, compression_rate=cr)
        itrans_img = img_transform(trans_img, trans_category=trans_category, reverse=True)
        plot_line.append(cv2.PSNR(img, itrans_img))
    plt.plot(plt_x, plot_line)
    return {key.tolist(): value for key, value in zip(plt_x, plot_line)}


def float32img_output(f32_img: np.float32) -> np.uint8:
    return np.uint8(np.round(f32_img))
