import cv2 as cv
import numpy as np
import canny_detect 
import edge_detect
import random
import math
import draw
import scipy
import sys
import ED
import ES
import EC
import ellipse_cluster
#f1:判断弧段的首尾是否相近
def arcs_h_e(e):
    ph = e[0]
    phx = ph[0]
    phy = ph[1]
    for i,p in enumerate(e):
        if i < len(e)/2:continue
        pex = p[0]
        pey = p[1]
        if abs(phx-pex)+abs(phy-pey)<50:
            return True
    return False
#f2:椭圆参数转换
def param_diversion_ell2ell(es):
    x = es[0][0]
    y = es[0][1]
    (width,height) = es[1]
    if width >= height:
        a = width / 2.0
        b = height / 2.0
        theta_deg = es[2]  # 已是顺时针
    else:
        a = height / 2.0
        b = width / 2.0
        theta_deg = es[2] - 90  # 长轴沿着 height，要校正成长轴的方向
    return x,y,a,b,theta_deg/180*math.pi
#f3:计算椭圆内点
def get_inlner_points(ell,e):
    num = 0
    for p in e:
        x = p[0]
        y = p[1]
        F = ((x*math.cos(ell[4])-ell[0]*math.cos(ell[4])+y*math.sin(ell[4])-ell[1]*math.sin(ell[4])))**2/(ell[2]**2)+((-x*math.sin(ell[4])+ell[0]*math.sin(ell[4])+y*math.cos(ell[4])-ell[1]*math.cos(ell[4])))**2/(ell[3]**2)
        if abs(F-1)<0.1:
            num +=1
    return num
#f4:计算椭圆内对应掩码的均值
def get_ellipse_mask_mean(mask, ellipse_param):

    x, y, axis1, axis2, angle = ellipse_param

    # 创建一个与掩码大小一致的空白图像
    ellipse_mask = np.zeros_like(mask, dtype=np.uint8)

    # 在该空图上绘制一个填充的椭圆（值为255）
    cv.ellipse(
        ellipse_mask,
        center=(int(x), int(y)),
        axes=(int(axis1), int(axis2)),
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1  # 填充
    )

    # 提取掩码中处于椭圆内的像素值
    values_in_ellipse = mask[ellipse_mask == 255]

    if values_in_ellipse.size == 0:
        return 0.0
    else:
        return float(np.mean(values_in_ellipse))
#计算斜率
def get_slope(p1,p2):
    p1x = p1[0]
    p2x = p2[0]
    p1y = p1[1]
    p2y = p2[1]
    if p1x == p2x:
        return float('inf')
    else:
        return (p2y-p1y)/(p2x-p1x)
#根据斜率计算两条直线之间的夹角
def get_included_angle(k1,k2):
    # 如果两条线都垂直
    if math.isinf(k1) and math.isinf(k2):
        return 0.0

    # 如果其中一条是垂直线
    if math.isinf(k1):
        return 90.0 - math.degrees(math.atan(abs(k2)))
    if math.isinf(k2):
        return 90.0 - math.degrees(math.atan(abs(k1)))

    # 一般情况：使用 tan(θ) = |(k1 - k2)/(1 + k1*k2)|
    if (1 + k1 * k2) == 0:
        return 90.0  # 直角相交

    tan_theta = abs((k1 - k2) / (1 + k1 * k2))
    theta_rad = math.atan(tan_theta)
    theta_deg = math.degrees(theta_rad)
    return theta_deg
if __name__ == "__main__":
    filename  = r"..\img_test\1.jpg"
    img = cv.imread(filename)
    #提取图像中黑色的部分
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 定义黑色的 HSV 范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 200])  # 调整 V 值以扩大/缩小范围
    # 创建掩码
    mask = cv.inRange(hsv, lower_black, upper_black)
    cv.imwrite(f"..\\img_of_process\\black_mask.bmp",mask)
    # 显示结果
    #cv.imshow('Black Region', mask)
    #cv.waitKey(0)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 1.0)
    canny,_,_ = canny_detect.canny_detect(gray)
    canvas1 = 255*np.ones_like(img)
    canvas2 = 255*np.ones_like(img)
    canvas3 = 255*np.ones_like(img)
    canvas4 = 255*np.ones_like(img)
    canvas5 = 255*np.ones_like(img)
    canvas6 = 255*np.ones_like(img)
    canvas7 = 255*np.ones_like(img)
    _iminlength = 150
    contours = ED.find_contours(canny,_iminlength)
    draw.draw_arcs(canvas6,contours)
    cv.imwrite(f"..\\img_of_process\\contours.bmp",canvas6)
    cv.imwrite(f"..\\img_of_process\\canny.bmp",canny)
    #过滤掉没有首尾链接的弧段
    arcs1 = []
    for e in contours:
        p1 = e[0]
        p2 = e[len(e)-1]
        if arcs_h_e(e):
            arcs1.append(e)
    draw.draw_arcs(canvas1,arcs1)
    cv.imwrite(f"..\\img_of_process\\contours2.bmp",canvas1)
    #椭圆拟合
    ells = []
    for e in arcs1:
        es = cv.fitEllipse(np.array(e))
        ell = param_diversion_ell2ell(es)
        x_c,y_c,A,B,_ = ell
        #计算椭圆得分
        num2 = get_inlner_points(ell,e)
        num2 = get_inlner_points(ell,e)
        #print(f"num2: {num2}")
        R_c1 = num2/len(e)
        R_c2 = num2/(math.pi*(3*(A+B)-math.sqrt((3*A+B)*(A+3*B))))
        R_jude = 0.5*R_c1+0.5*R_c2
        print(f"评价: {R_jude}")
        if R_jude<0.8:continue
        ells.append(ell)
    for i,ell in enumerate(ells):
        #print(f"ell: {ell}")
        if ell ==None:continue
        cv.ellipse(img,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(255,100,0),2)
        cv.circle(img,(int(ell[0]),int(ell[1])),3,(0,0,255),-1)
    cv.imwrite("..\\img_of_process\\ans.bmp",img)
    #区分出是圆还是圆环
    circle = []
    cirque = []
    for e in ells:
        #查询中心附近是否为黑
        m = get_ellipse_mask_mean(mask,e)
        if m<30:
            #圆
            circle.append(e)
        else:
            cirque.append(e)
    for i,ell in enumerate(circle):
        #print(f"ell: {ell}")
        if ell ==None:continue
        cv.ellipse(img,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(255,100,0),2)
        cv.circle(img,(int(ell[0]),int(ell[1])),3,(0,0,255),-1)
    for i,ell in enumerate(cirque):
        #print(f"ell: {ell}")
        if ell ==None:continue
        cv.ellipse(img,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,100,255),2)
        cv.circle(img,(int(ell[0]),int(ell[1])),3,(0,0,255),-1)
    
    #计算得到，聚类之后的中心
    circle_center = []
    cirque_center = []
    for e in circle:
        center = (e[0],e[1])
        circle_center.append(center)
    while cirque:
        r0 = cirque.pop(0)
        cirque_center.append([r0[0],r0[1]])
        to_remove = []
        for i,ri in enumerate(cirque):
            label = 1
            num = 0
            while label:
                for j,rj in enumerate(cirque_center):
                    if abs(ri[0]-rj[0])<5 and abs(ri[1]-rj[1])<5:
                        label = 0
                        rj[0] = (ri[0]+rj[0])/2
                        rj[1] = (ri[1]+rj[1])/2
                        to_remove.append(ri)
                        break
                    else:
                        num += 1
                if label and num == len(cirque_center):
                    to_remove.append(ri)
                    cirque_center.append([ri[0],ri[1]])
                    label = 0
        for idx in to_remove:
            cirque.remove(idx)
    print(f"circle: {circle_center}")
    print(f"cirque: {cirque_center}")
    #特征规则，进行排序
    #s1：寻找两组三个共线的圆环
    #s1.1:计算两两之间的斜率
    slopes = []
    for i,ci in enumerate(cirque_center):
        print(f"ci: {ci}")
        cv.putText(img, str(i), (int(ci[0]) + 5, int(ci[1]) - 5),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=2)
        for j,cj in enumerate(cirque_center):
            if i>=j :continue
            slope = get_slope(ci,cj)
            slopes.append([i,j,slope])
    #s1.1':测试
    for i in slopes:
        print(i)
    #s1.2:根据斜率找到共线的圆环
        #根据共线规则进行聚类
    cirquet = []
    while slopes:
        s0 = slopes.pop(0)
        cirquet.append(s0)
        to_remove = []
        for i,ri in enumerate(slopes):
            label = 1
            num = 0
            while label :
                for j,rj in enumerate(cirquet):
                    aij = get_included_angle(ri[2],rj[2])
                    #print(f"aij: {aij}")
                    if aij<1 and (ri[0] in rj or ri[1] in rj) :
                        label = 0
                        if ri[0] not in rj:
                            rj.append(ri[0])
                        if ri[1] not in rj:
                            rj.append(ri[1])
                        to_remove.append(ri)
                        break
                    else:
                        num+=1
                if label and num==len(cirquet):
                    to_remove.append(ri)
                    cirquet.append(ri)
                    label = 0
        for idx in to_remove:
            slopes.remove(idx)
    print(f"cirquet: {cirquet}  len of cirquet: {len(cirquet)}")
    #测试用例为4个共线
    cirque1 = []
    cirque2 = []
    to_remove = []
    for i in cirquet:
        if len(i)==3:
            to_remove.append(i)
    for idx in to_remove:
        cirquet.remove(idx)
    print(f"cirque1: {cirque1} \n cirquet: {cirquet}")
    cv.imwrite("..\\img_of_process\\ans.bmp",img)
    print("over")
    #cv.imshow("org",canny)
