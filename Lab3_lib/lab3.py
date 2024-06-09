import cv2
import matplotlib.pyplot as plt


def lab3(img_address: str = 'imgs/shiyan3.bmp'):
    img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)
    plt.figure(1)
    figure = 1
    for filter_size in [3, 5, 7]:
        smoothed_image = cv2.medianBlur(img, filter_size)  # 根据实验需要调整滤波范围大小
        plt.subplot(1,3,figure)
        plt.imshow(smoothed_image, cmap='gray')
        plt.title(f'Smoothed Image with ksize {filter_size}',fontsize=6)
        plt.axis('off')
        figure += 1
    plt.show()

    cv2.medianBlur(src=img, ksize=3, dst=img)
    thr, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.figure(2)
    plt.imshow(otsu_img, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

    plt.figure(3)
    plt.suptitle('Opening')
    figure = 1
    for r in [1, 3, 5]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        eroded_image = cv2.erode(otsu_img, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        plt.subplot(1,3,figure)
        plt.imshow(dilated_image, cmap='gray')
        plt.title('Processed Image with r=' + str(r),fontsize = 6)
        plt.axis('off')
        figure += 1
    plt.show()


if __name__ == '__main__':
    lab3()
