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
import argparse
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
        axes=(int(axis1*6/10), int(axis2*6/10)),
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
#判断线段和射线有没有交点
def has_intersection(seg_p1, seg_p2, ray_origin, ray_direction):

    # 转换为向量形式
    p = np.array(seg_p1)
    r = np.array(seg_p2) - np.array(seg_p1)
    q = np.array(ray_origin)
    s = np.array(ray_direction) - np.array(ray_origin)
    r_cross_s = np.cross(r, s)
    q_minus_p = q - p
    q_minus_p_cross_r = np.cross(q_minus_p, r)

    # 共线但不平行
    if r_cross_s == 0:
        return False

    t = np.cross((q - p), s) / r_cross_s
    u = np.cross((q - p), r) / r_cross_s

    # t 在 [0, 1] 表示在线段上；u >= 0 表示在射线上
    return  t >= 0 and t <= 1 and u >= 0
#计算点到直线的距离
def point_to_line_distance(point, line_p1, line_p2):
    # 转为 numpy 向量
    p0 = np.array(point)
    p1 = np.array(line_p1)
    p2 = np.array(line_p2)
    # 计算线段方向向量和点差向量
    line_vec = p2 - p1
    point_vec = p0 - p1
    # 叉积模长 / 线段长度，得到垂直距离
    area = np.abs(np.cross(line_vec, point_vec))
    length = np.linalg.norm(line_vec)
    if length == 0:
        raise ValueError("line_p1 and line_p2 cannot be the same point.")
    distance = area / length
    return distance
#判断射线有没有交点
def ray_ray_intersect(a1, a2, b1, b2):
    #1为起点
    # 转为 numpy 向量
    a1 = np.array(a1, dtype=np.float64)
    a2 = np.array(a2, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)

    # 射线方向向量
    r = a2 - a1
    s = b2 - b1

    r_cross_s = np.cross(r, s)

    if r_cross_s == 0:
        # 平行或重合
        return False

    # 解 t 和 u
    diff = b1 - a1
    t = np.cross(diff, s) / r_cross_s
    u = np.cross(diff, r) / r_cross_s

    # 射线的参数 t 和 u 要 >= 0 才表示在方向上
    return t >= 0 and u >= 0
#判断射线和直线有没有交点
def ray_line_intersect(ray_origin, ray_direction_point, line_p1, line_p2):
    """
    判断射线与直线是否相交（二维）
    :param ray_origin: 射线起点 (x1, y1)
    :param ray_direction_point: 射线方向点 (x2, y2)
    :param line_p1: 直线上的点1 (x3, y3)
    :param line_p2: 直线上的点2 (x4, y4)
    :return: (bool, intersection_point) 有交点返回 True 和交点坐标，否则返回 False 和 None
    """
    # 向量化
    p = np.array(ray_origin, dtype=float)
    r = np.array(ray_direction_point, dtype=float) - p

    q = np.array(line_p1, dtype=float)
    s = np.array(line_p2, dtype=float) - q

    r_cross_s = np.cross(r, s)
    print(f"r: {r}\ns: {s}")
    if np.isclose(r_cross_s, 0):
        # 平行（包括重合）情况
        return False

    diff = q - p
    t = np.cross(diff, s) / r_cross_s  # 射线的参数

    # 只要 t >= 0，说明交点在射线方向上
    if t >= 0:
        intersection_point = p + t * r
        return True
    else:
        return False
#判断两个线段有没有交点
def intersection(start1, end1, start2, end2):
        def inside(x1, y1, x2, y2, xk, yk):
            # 若与 x 轴平行，只需要判断 x 的部分
            # 若与 y 轴平行，只需要判断 y 的部分
            # 若为普通线段，则都要判断
            return (x1 == x2 or min(x1, x2) <= xk <= max(x1, x2)) and (y1 == y2 or min(y1, y2) <= yk <= max(y1, y2))
        
        def update(ans, xk, yk):
            # 将一个交点与当前 ans 中的结果进行比较
            # 若更优则替换
            return [xk, yk] if not ans or [xk, yk] < ans else ans
        
        x1, y1 = start1
        x2, y2 = end1
        x3, y3 = start2
        x4, y4 = end2

        ans = None
        # 判断 (x1, y1)~(x2, y2) 和 (x3, y3)~(x4, y3) 是否平行
        if (y4 - y3) * (x2 - x1) == (y2 - y1) * (x4 - x3):
            # 若平行，则判断 (x3, y3) 是否在「直线」(x1, y1)~(x2, y2) 上
            if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1):
                if inside(x1, y1, x2, y2, x3, y3):
                    ans = update(ans, x3, y3)
                if inside(x1, y1, x2, y2, x4, y4):
                    ans = update(ans, x4, y4)
                if inside(x3, y3, x4, y4, x1, y1):
                    ans = update(ans, x1, y1)
                if inside(x3, y3, x4, y4, x2, y2):
                    ans = update(ans, x2, y2)
        else:
            t1 = (x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / ((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1))
            t2 = (x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / ((x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3))
            # 判断 t1 和 t2 是否均在 [0, 1] 之间
            if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
                ans = [x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1)]

        return ans
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只修改图片编号")
    parser.add_argument("--img_id", type=str, required=True, help="图片编号，如 8 表示使用 8.jpg")
    args = parser.parse_args()

    # 构造完整文件路径
    filename = rf"..\img_test\{args.img_id}.jpg"
    print(f"加载图像: {filename}")
    img = cv.imread(filename)
    #提取图像中黑色的部分
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 定义黑色的 HSV 范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])  # 调整 V 值以扩大/缩小范围
    # 创建掩码
    mask = cv.inRange(hsv, lower_black, upper_black)
    cv.imwrite(f"..\\img_of_process\\black_mask.png",mask)
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
    cv.imwrite(f"..\\img_of_process\\contours.png",canvas6)
    cv.imwrite(f"..\\img_of_process\\canny.png",canny)
    #过滤掉没有首尾链接的弧段
    arcs1 = []
    for e in contours:
        p1 = e[0]
        p2 = e[len(e)-1]
        if arcs_h_e(e):
            arcs1.append(e)
    draw.draw_arcs(canvas1,arcs1)
    cv.imwrite(f"..\\img_of_process\\contours2.png",canvas1)
    #椭圆拟合
    ells = []
    for e in arcs1:
        es = cv.fitEllipse(np.array(e))
        ell = param_diversion_ell2ell(es)
        x_c,y_c,A,B,_ = ell
        #计算椭圆得分
        num2 = get_inlner_points(ell,e)
        #print(f"num2: {num2}")
        R_c1 = num2/len(e)
        R_c2 = num2/(math.pi*(3*(A+B)-math.sqrt((3*A+B)*(A+3*B))))
        R_jude = 0.5*R_c1+0.5*R_c2
        #print(f"评价: {R_jude}")
        if R_jude<0.8:continue
        ells.append(ell)
    for i,ell in enumerate(ells):
        #print(f"ell: {ell}")
        if ell ==None:continue
        cv.ellipse(img,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(255,100,0),2)
        cv.circle(img,(int(ell[0]),int(ell[1])),3,(0,0,255),-1)
    cv.imwrite("..\\img_of_process\\ans.png",img)
    #区分出是圆还是圆环
    circle = []
    cirque = []
    #
    for e in ells:
        m = get_ellipse_mask_mean(mask,e)
        #print(f"椭圆内对应的掩码值： {m}")
        if m<15:
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
        center = [e[0],e[1]]
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
        #print(f"ci: {ci}")
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
    # for i in slopes:
    #     print(i)
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
    #保存下共线的部分
    to_remove = []
    for i in cirquet:
        if len(i)==3:
            to_remove.append(i)  #没有共线
    for idx in to_remove:
        cirquet.remove(idx)
    print(f"cirquet: {cirquet}")
    #测试
    L1 = []
    L2 = []
    L3 = []
    for i in cirquet:
        idx1c = cirque_center[i[0]]
        idx2c = cirque_center[i[1]]
        idx3c = cirque_center[i[3]]
        cv.line(img,(int(idx1c[0]),int(idx1c[1])),(int(idx3c[0]),int(idx3c[1])),(0,0,255),2)
        #判断射线和线段有没有交点
        if has_intersection(circle_center[0],circle_center[1],idx1c,idx3c) or has_intersection(circle_center[0],circle_center[1],idx3c,idx1c):
            L1 = i
            cirquet.remove(i)
    L2 = cirquet[0]
    for i in range(0,7):
        if i in L1:continue
        if i in L2:continue
        L3.append(i)
    print(f"L1: {L1}\nL2: {L2}\nL3: {L3}")

    #对L1进行编号(2,3,4)
    dis_L1_t = []
    c1 = cirque_center[L1[0]]
    dis1 = point_to_line_distance(c1,circle_center[0],circle_center[1])
    dis_L1_t.append((dis1,L1[0]))
    c2 = cirque_center[L1[1]]
    dis2 = point_to_line_distance(c2,circle_center[0],circle_center[1])
    dis_L1_t.append((dis2,L1[1]))
    c3 = cirque_center[L1[3]]
    dis3 = point_to_line_distance(c3,circle_center[0],circle_center[1])
    dis_L1_t.append((dis3,L1[3]))
    dis_L1_t.sort(key=lambda x :x[0])
    print(f"dis: {dis_L1_t}")
    for i,e in enumerate(dis_L1_t):
        idx = e[1]
        cirque_center[idx].append(i+2)
        cv.putText(img, str(i+2), (int(cirque_center[idx][0]) - 20, int(cirque_center[idx][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    
    #对剩下的两个圆环进行编号
    if ray_line_intersect(cirque_center[L3[0]],cirque_center[L3[1]],circle_center[0],circle_center[1]):
        cirque_center[L3[0]].append(1)
        cirque_center[L3[1]].append(6)
        cv.putText(img, str(1), (int(cirque_center[L3[0]][0]) - 20, int(cirque_center[L3[0]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
        cv.putText(img, str(6), (int(cirque_center[L3[1]][0]) - 20, int(cirque_center[L3[1]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    else:
        cirque_center[L3[1]].append(1)
        cirque_center[L3[0]].append(6)
        cv.putText(img, str(1), (int(cirque_center[L3[1]][0]) - 20, int(cirque_center[L3[1]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
        cv.putText(img, str(6), (int(cirque_center[L3[0]][0]) - 20, int(cirque_center[L3[0]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    #对L2进行编号（0，2，5）
    c1 = cirque_center[L2[0]]
    c2 = cirque_center[L2[3]]
    for i in cirque_center:
        if len(i)==2:continue
        if i[2]==1:c_idx_1 = [i[0],i[1]]
    if intersection(c1,c_idx_1,(cirque_center[L1[0]][0],cirque_center[L1[0]][1]),(cirque_center[L1[3]][0],cirque_center[L1[3]][1]))==None:
        cirque_center[L2[0]].append(0)
        cirque_center[L2[3]].append(5)
        cv.putText(img, str(0), (int(cirque_center[L2[0]][0]) - 20, int(cirque_center[L2[0]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
        cv.putText(img, str(5), (int(cirque_center[L2[3]][0]) - 20, int(cirque_center[L2[3]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    else:
        cirque_center[L2[0]].append(5)
        cirque_center[L2[3]].append(0)
        cv.putText(img, str(5), (int(cirque_center[L2[0]][0]) - 20, int(cirque_center[L2[0]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
        cv.putText(img, str(0), (int(cirque_center[L2[3]][0]) - 20, int(cirque_center[L2[3]][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    #对圆进行编号
    c1 = circle_center[0]
    c2 = circle_center[1]
    for i in cirque_center:
        if len(i)==2:continue
        if i[2]==1:c_idx_1 = [i[0],i[1]]
        if i[2]==5:c_idx_5 = [i[0],i[1]]
    if ray_ray_intersect(c_idx_1,c_idx_5,c1,c2):
        circle_center[0].append(7)
        circle_center[1].append(8)
        cv.putText(img, str(7), (int(circle_center[0][0]) - 20, int(circle_center[0][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
        cv.putText(img, str(8), (int(circle_center[1][0]) - 20, int(circle_center[1][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    else:
        circle_center[0].append(8)
        circle_center[1].append(7)
        cv.putText(img, str(8), (int(circle_center[0][0]) - 20, int(circle_center[0][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
        cv.putText(img, str(7), (int(circle_center[1][0]) - 20, int(circle_center[1][1]) + 20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(0, 255, 0),
                thickness=3)
    print(f"cirque: {cirque_center}")
    cv.imwrite(f"..\\img_of_process\\ans_{args.img_id}.png",img)
    print("over")
    #cv.imshow("org",canny)
