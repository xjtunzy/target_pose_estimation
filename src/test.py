import cv2 as cv
import numpy as np
import canny_detect
import edge_detect
import random
import math
import time
import os
import numpy as np
import processing
import draw

if __name__ =="__main__":
    file_path = r"E:\dataset\aamed_ellipse_datasets\Random Images - Dataset #1\imagenames.txt"
    image_names = []

    with open(file_path, "r", encoding="gbk") as f:
        for line in f:
            line = line.strip()  # 去除首尾空白字符
            if line.endswith((".jpg",".bmp")) and not line.startswith(("/", "'")):  # 过滤注释和路径
                image_names.append(line)

    print(f"共找到 {len(image_names)} 张图片：")
    print(image_names[:10])  # 打印前10个文件名示例

    recall = 0
    times = 0
    for img_name in image_names:
        img = cv.imread(f"E:\\dataset\\aamed_ellipse_datasets\\Random Images - Dataset #1\\images\\{img_name}")

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 1.0)
        canny,_,_ = canny_detect.canny_detect(blurred)
        canvas = 255*np.ones_like(img)  # 全白背景
        # 3. 将边缘位置设为灰色（128,128,128）
        edge_mask = (canny == 255)  # 找到边缘像素位置
        canvas[edge_mask] = [128, 128, 128]  # BGR格式的灰色
        # curved_arcs = processing.get_curved_arcs(img,img_name)
        # draw.draw_arcs(canvas,curved_arcs)
        # cv.imwrite(f"curved_arcs\\{img_name}",canvas)
        # cv.imwrite(f"canny_gaussian\\{img_name}",canny)
        ells = processing.get_ells(img)
        for i,ell in enumerate(ells):
            if ell ==None:continue
            cv.ellipse(canvas,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,255,0),2)
        cv.imwrite(f"ours2\\{img_name}",canvas)
