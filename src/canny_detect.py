import cv2
import numpy as np

#定义了一个自适应阈值的canny边缘检测，在程序中也可以自行调试边缘检测的阈值
#0.78-0.92
def canny_detect(image, apertureSize=3, low_percent=0.78, high_percent=0.91):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the gradient using Sobel operators
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=apertureSize)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=apertureSize)
    
    # Compute the gradient magnitude
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Flatten the magnitude and calculate the thresholds based on percentiles
    flat_magnitude = magnitude.flatten()
    low_thresh = np.percentile(flat_magnitude, low_percent * 100)
    high_thresh = np.percentile(flat_magnitude, high_percent * 100)

    # Apply Canny edge detection with adaptive thresholds
    edges = cv2.Canny(image, low_thresh, high_thresh, apertureSize=apertureSize)
    
    return edges,sobel_x,sobel_y


if __name__ =="__main__":
    # 测试用例
    filename1 = r"E:\dataset\aamed_ellipse_datasets\Prasad Images - Dataset Prasad\images\033_0034.jpg"
    filename2 = r"E:\dataset\test\10_djh\500_L.BMP"
    image = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    #中值滤波
    img = cv2.medianBlur(image,3)
    edges,_,_ = canny_detect(img)
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    edges2 = cv2.bitwise_not(edges)
    # 结果展示
    cv2.imshow("Edges", edges2)
    cv2.imwrite("canny_edge.bmp",edges2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()