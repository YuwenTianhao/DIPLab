import numpy as np
import cv2

laplacian_kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

def edge_detect_by_operator(img:np.uint8,operator:str='Roberts')->np.uint8:
    if operator == 'Roberts':
        #Roberts算子
        kernel_x = np.array([[-1,0],[0,1]], dtype=int)
        kernel_y = np.array([[0,-1],[1,0]], dtype=int)
        x = cv2.filter2D(img, cv2.CV_8U, kernel_x)
        y = cv2.filter2D(img, cv2.CV_8U, kernel_y)
        #转uint8
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX,1,absY,1,0)
    elif operator == 'Sobel':
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)#垂直方向梯度
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)#水平方向梯度
        absX = cv2.convertScaleAbs(x) #转回原来的uint8格式，否则图像无法显示。
        absY = cv2.convertScaleAbs(y) #转回原来的uint8格式，否则图像无法显示。
        #两个方向的梯度加起来形成最终的梯度
        return cv2.addWeighted(absX, 1, absY, 1, 0)
    elif operator == 'Laplacian':
        return cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_8U,kernel=laplacian_kernel))