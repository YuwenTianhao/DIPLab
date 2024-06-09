import cv2
import numpy as np
import matplotlib.pyplot as plt
from Lab2_lib.Lab2 import search_best_param
from Lab4_lib.operator import edge_detect_by_operator as edge_detect


def best_filter(img:np.uint8,noise_img:np.uint8)->np.uint8:
    assert img.dtype == noise_img.dtype
    m_size = search_best_param(img,noise_img,blur_type='median')
    best_median_img = cv2.medianBlur(src=noise_img, ksize=m_size)
    b_size = search_best_param(img,noise_img,blur_type='blur')
    best_blur_img = cv2.blur(src=noise_img, ksize=(b_size, b_size))
    if cv2.PSNR(img,best_median_img) > cv2.PSNR(img,best_blur_img):
        return best_median_img
    else:
        return best_blur_img


def hough_transform(img: cv2.UMat,index: int = 0):
    h_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40, param1=40, param2=20, minRadius=0,maxRadius=1000)
    circles = np.uint16(np.around(circles))
    if circles is not None:
        for circle in circles[0,:]:
            x,y,r=int(circle[0]),int(circle[1]),circle[2]
            #draw the outer circle
            cv2.circle(h_img,(x,y),r,(0,255,0),2)
    print('Hough_Transform Result saved as '+ str(index) +'_hough_img'+'.jpg')
    cv2.imwrite(str(index)+'_hough_img'+'.jpg',h_img)

def lab4(img_address: str = 'imgs/houghorg.bmp', *args):
    img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)
    noise_imgs = []
    if args:
        for address in args:
            assert isinstance(address,str)
            tem_img = cv2.imread(address,cv2.IMREAD_GRAYSCALE)
            tem_img = best_filter(img,tem_img)
            noise_imgs.append(tem_img)
            del tem_img
    if args:
        plt.figure()
        plt.subplot(1,3,1)
        plt.title('Original Image')
        plt.imshow(img,cmap='gray')
        plt.subplot(1,3,2)
        plt.title('GS Image')
        plt.imshow(noise_imgs[0],cmap='gray')
        plt.subplot(1,3,3)
        plt.title('SP Image')
        plt.imshow(noise_imgs[1],cmap='gray')
        plt.show()

    img_index = 0
    # 如果去掉下面两行会使得Hough圆检测结果显著变差
    noise_imgs[0] = cv2.imread(args[0],cv2.IMREAD_GRAYSCALE)
    noise_imgs[0] = cv2.blur(src=noise_imgs[0], ksize=(7, 7))
    for input_img in [img]+noise_imgs:
        r_re = edge_detect(input_img, operator='Roberts')
        s_re = edge_detect(input_img, operator='Sobel')
        l_re = edge_detect(input_img, operator='Laplacian')
        hough_transform(input_img,img_index)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(r_re,cmap='gray')
        plt.title(str(img_index)+'_Roberts')
        plt.subplot(1,3,2)
        plt.imshow(s_re,cmap='gray')
        plt.title(str(img_index)+'_Sobel')
        plt.subplot(1,3,3)
        plt.imshow(l_re,cmap='gray')
        plt.title(str(img_index)+'_Laplacian')
        plt.show()
        img_index += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lab4('imgs/houghorg.bmp','imgs/houghgau.bmp','imgs/houghsalt.bmp')