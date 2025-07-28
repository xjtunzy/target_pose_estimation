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
import matlab.engine
eng = matlab.engine.start_matlab()

'''
processing说明：
rdp多边形逼近处理
交点切割处理
'''
corh_select = 1
cnc_select = not(corh_select)
#计算向量之间的夹角
def  get_angle(l1,l2):
    #首先计算向量叉乘
    l_c = np.cross(l1,l2)
    s = np.sign(l_c)
    #计算点乘
    l_d = np.dot(l1,l2)
    cos_theta = l_d / (np.linalg.norm(l1) * np.linalg.norm(l2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if s>0:return theta
    elif s==0:return 0
    else:return -theta
# 计算点到线段的垂直距离
def perpendicular_distance(pt, line_start, line_end):
    # 使用点到直线的距离公式
    x1, y1 = line_start
    x2, y2 = line_end
    x0, y0 = pt
    
    # 计算直线的两个向量
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if line_length == 0:
        return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    
    # 计算点到直线的距离
    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length

# 递归 RDP 算法，存储点的索引
def rdp(points, indices, epsilon, result, result_indices):
    if len(points) < 2:
        return
    
    # 找到最大距离点
    max_dist = 0
    index = 0
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            index = i

    if max_dist > epsilon:
        # 将曲线分为两部分
        left_points = points[:index + 1]
        right_points = points[index:]

        left_indices = indices[:index + 1]
        right_indices = indices[index:]

        # 递归处理左右两部分
        left_result = []
        left_result_indices = []
        right_result = []
        right_result_indices = []

        rdp(left_points, left_indices, epsilon, left_result, left_result_indices)
        rdp(right_points, right_indices, epsilon, right_result, right_result_indices)

        # 合并结果
        result.extend(left_result[:-1])
        result_indices.extend(left_result_indices[:-1])

        result.extend(right_result)
        result_indices.extend(right_result_indices)
    else:
        # 只保留开始和结束点
        result.append(points[0])
        result_indices.append(indices[0])

        result.append(points[-1])
        result_indices.append(indices[-1])

def get_ellipse(e,ch=1):
        score = 0
        x = []
        y = []
        for p in e:
            p_x = p[0]
            p_y = p[1]
            x.append(p_x)
            y.append(p_y)
        x = np.array(x)
        y = np.array(y)
        D = np.column_stack([x**2,x*y,y**2,x,y,np.ones_like(x)])
        S = D.T @ D
        #构造约束矩阵
        C = np.zeros((6,6))
        C[0,2] = C[2,0] = 2
        C[1,1] = -1
        #求解广义特征值
        S = np.array(S, dtype=np.float64)
        C = np.array(C, dtype=np.float64)
        ans = eng.eig(S,C,nargout=2)
        #print(ans)
        geval,gevec = ans[1],ans[0]
        #print(geval)
        #print(gevec)
        geval = np.array(geval)
        gevec = np.array(gevec)
        idx = np.where((geval>0)&(~np.isinf(geval)))
            #print(f"geval:{geval}")
            #print(idx[0])
        print(f"len of idx: {len(idx[0])}")
        if len(idx[0]) == 0 :
            print("无法找到正特征值") 
            return None
        a,b,c,d,f,g = gevec[:,idx[0][0]]
        b = b/2
        d = d/2
        f = f/2
        if a*c-b**2 <= 0: 
            print("正特征值对应的不是椭圆")
            return None
        devin = b**2 - a*c
        x1 = (c*d-b*f)/devin
        y1 = (a*f-b*d)/devin
        #cv.circle(canvas,(int(x1),int(y1)),2,(0,0,255),-1)
        if (a-c)**2+4*b**2<=0:return None
        #if math.sqrt((a-c)**2+4*b**2)-(a+c)<=0:return None
        devin1 = (devin)*(math.sqrt((a-c)**2+4*b**2)-(a+c))
        #if -math.sqrt((a-c)**2+4*b**2)-(a+c)<=0:return None
        devin2 = (devin)*(-math.sqrt((a-c)**2+4*b**2)-(a+c))
        print(f"devin1: {devin1}  devin2: {devin2}")
        if 2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g)/devin1<=0:
            print("A轴求解出错")
            return None
        A = math.sqrt(2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g)/devin1)
        if 2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g)/devin2<=0:
            print("B轴求解出错")
            return None
        B = math.sqrt(2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g)/devin2)
        if not a==c:
            if b== 0:
                if a<c:
                    theta = 0
                else:
                    theta = math.pi/2
            else:
                if a<c:
                    theta = 0.5*math.atan((2*b)/(a-c))
                else:
                    theta = 0.5*math.atan((2*b)/(a-c)) + math.pi/2
        if ch==1:return x1,y1,A,B,theta
        return a,b,c,d,f,g

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

def get_intersection_point2(l11,l12,l21,l22):
    #直线1
    dx1 = l11[0]-l12[0]
    dy1 = l11[1]-l12[1]
    if(dx1==0): 
        slope1 = None
    else:
        slope1 = dy1/dx1
    #直线2
    dx2 = l21[0]-l22[0]
    dy2 = l21[1]-l22[1]
    if(dx2==0): 
        slope2 = None
    else:
        slope2 = dy2/dx2
    if slope1==None and not slope2==None:
        p_x = l11[0]
        p_y = slope2*(p_x-l21[0])+l21[1]
    elif not slope1==None and slope2==None:
        p_x = l21[0]
        p_y = slope1*(p_x-l11[0])+l11[1]
    elif slope1==slope2:
        return None
    else:
         p_x = (slope1*l11[0]-l11[1]-slope2*l21[0]+l21[1])/(slope1-slope2)
         p_y = slope1*(p_x-l11[0])+l11[1]
    return (p_x,p_y)
def get_ell_score(ell,G_i):
    len_all = 0
    num = 0
    print(f"len of g: {len(G_i)}")
    for k in range(len(G_i)-1):
        e = G_i[k+1]
        len_all += len(e)
        for p in e:
            x = p[0]
            y = p[1]
            F = ((x*math.cos(ell[4])-ell[0]*math.cos(ell[4])+y*math.sin(ell[4])-ell[1]*math.sin(ell[4])))**2/(ell[2]**2)+((-x*math.sin(ell[4])+ell[0]*math.sin(ell[4])+y*math.cos(ell[4])-ell[1]*math.cos(ell[4])))**2/(ell[3]**2)
            if abs(F-1)<0.1:
                num +=1 
    return num/len_all
def get_inlner_points(ell,e):
    num = 0
    print(f"ell: {ell}")
    for p in e:
        x = p[0]
        y = p[1]
        F = ((x*math.cos(ell[4])-ell[0]*math.cos(ell[4])+y*math.sin(ell[4])-ell[1]*math.sin(ell[4])))**2/(ell[2]**2)+((-x*math.sin(ell[4])+ell[0]*math.sin(ell[4])+y*math.cos(ell[4])-ell[1]*math.cos(ell[4])))**2/(ell[3]**2)
        if abs(F-1)<0.1:
            num +=1
    print(f"num: {num}")
    return num
def get_dis_of_2points(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def get_s_of_triangle(p1,p2,p3):
    #使用海伦公式
    dis1 = get_dis_of_2points(p1,p2)
    dis2 = get_dis_of_2points(p1,p3)
    dis3 = get_dis_of_2points(p2,p3)
    dis = (dis1+dis2+dis3)/2
    print(f"dis1: {dis1} dis2: {dis2} dis3: {dis3} dis: {dis} uuu: {(dis-dis1)*(dis-dis2)*(dis-dis3)}")
    s = dis*math.sqrt((dis-dis1)*(dis-dis2)*(dis-dis3))
    return s
def get_cr(P):
    #p1-p-p3
    s1 = get_s_of_triangle(P[0],P[2],P[4])
    #p2-p-p4
    s2 = get_s_of_triangle(P[1],P[3],P[4])
    #p2-p-p3
    s3 = get_s_of_triangle(P[1],P[2],P[4])
    #p1-p-p4
    s4 = get_s_of_triangle(P[0],P[3],P[4])
    if s1 == 0 or s2 ==0 or s3==0 or s4==0:return None
    cr = (s1*s2)/(s3*s4)
    return cr
def get_curved_arcs(img,name):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    canny,_,_ = canny_detect.canny_detect(gray)
    contours = edge_detect.detect_edges(canny)
    
    arc_corner_point=[] #存放rdp算法计算出来的角点
    arc_index=[]#存放rdp算法计算出来的角点在原数组中的序号
    for arc in contours:
        arc_p = []
        arc_pid = []
        rdp(arc,range(len(arc)),1.5,arc_p,arc_pid)
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        # for p in arc_p:
        #     cv.circle(canvas2,p,2,random_color,-1)
        # for i in range(len(arc_p)-1):
        #     cv.line(canvas2,arc_p[i],arc_p[i+1],random_color,1)
        arc_corner_point.append(arc_p)
        arc_index.append(arc_pid)
    #print(f"length of arcs:{len(arcs)}  length of arc:{len(arc_corner_point)}")
    #cv.imwrite("img_of_process\\rdp.bmp",canvas2)
    arcs2=[]
    #角点切割
    _theta = 50
    _iminlength = 15
    #print("角点切割........")
    for n,e in enumerate(arc_corner_point):
        e_line = []
        for i in range(len(e)-1):
            #cv.circle(canvas3,e[0],3,(255,0,0),-1) #图像中弧段的开始使用蓝色表示
            # cv.line(canvas4,e[i],e[i+1],(random.randint(0,255),random.randint(0,255),random.randint(0,255)),2)
            l = np.array([e[i+1][0]-e[i][0],e[i+1][1]-e[i][1]])
            e_line.append(l)
        #print(f"e_line: {e_line}")
        e_theta = []
        for j in range(len(e_line)-1):
            t = get_angle(e_line[j],e_line[j+1])
            t = t/math.pi*180
            e_theta.append(t)
        if len(e_theta)==0:continue
        #构造布尔序列
        bool_sequence = []
        t_0 = e_theta[0]
        for t in e_theta:
            b = 0
            if abs(t+t_0)<(abs(t)+abs(t_0)):b=1
            bool_sequence.append(b)
        #构造长度序列
        c_idx = arc_index[n]
        len_sequence = []
        for i in range(len(e)):
            if i==0:continue
            length = c_idx[i]-c_idx[i-1]
            len_sequence.append(length)
        #计算角点坐标
        break_idx = []
        for k in range(len(bool_sequence)):
            #首先判断转角是否过大
            t = e_theta[k]
            if abs(t)>_theta: 
                break_idx.append(k+1)
                continue
            #判断旋转角
            b = bool_sequence[k]
            if b==1:
                b1 = bool_sequence[k-1]
                b2 = 1
                if k<len(bool_sequence)-1:
                    b2 = bool_sequence[k+1]
                if b1 ==0:
                    break_idx.append(k+1)
                    continue
                if b2==0:
                    break_idx.append(k+1)
            #其次需要判断相邻角点之间弧段的长度是否满足要求
            _multiple = 3.5
            len1 = len_sequence[k]
            len2 = len_sequence[k+1]
            len_max = max(len1,len2)
            len_min = min(len1,len2)
            if len_max>_multiple*len_min :
                break_idx.append(k+1)       
        #弧段
        edge = contours[n]
    
        if len(break_idx)==0:
            if len(edge)>=_iminlength:arcs2.append(edge)
        if len(break_idx)==1:
            edge1 = edge[:c_idx[break_idx[0]]]
            edge2 =  edge[c_idx[break_idx[0]]:]
            if len(edge1)>=_iminlength:arcs2.append(edge1)
            if len(edge2)>=_iminlength:arcs2.append(edge2) 
            continue
        for count,idx in enumerate(break_idx):
            edge_t = None
            if count == 0:
                edge_t = edge[:c_idx[idx]]
            elif count == len(break_idx)-1:
                edge_t1 = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                if len(edge_t1)>_iminlength:arcs2.append(edge_t1)
                edge_t = edge[c_idx[idx]:]
                #draw.draw_edge3(canvas2,edge_t,(0,0,0))
            else:
                edge_t = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
            if len(edge_t)<_iminlength:continue
            if not edge_t==None: arcs2.append(edge_t)    
    #需要继续过滤掉一些较直的弧段
    arcs3 = []
    #采用三角不等式
    for e in arcs2:
        eh = e[0]
        em = e[len(e)//2]
        ee = e[len(e)-1]
        dis_h_m = math.sqrt((eh[0]-em[0])**2+(eh[1]-em[1])**2)
        dis_h_e = math.sqrt((eh[0]-ee[0])**2+(eh[1]-ee[1])**2)
        dis_e_m = math.sqrt((ee[0]-em[0])**2+(ee[1]-em[1])**2)
        #定义的曲率
        t = 1-dis_h_e/(dis_h_m+dis_e_m)
        _t = 0.01
        if t<_t:continue
        arcs3.append(e)
    return arcs3

def get_ells(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 1.0)
    canvas1 = 255*np.ones_like(img)
    canvas2 = 255*np.ones_like(img)
    canvas3 = 255*np.ones_like(img)
    canvas4 = 255*np.ones_like(img)
    canvas5 = 255*np.ones_like(img)
    canvas6 = 255*np.ones_like(img)
    canny,_,_ = canny_detect.canny_detect(gray)
    #canny,_ = EC.mark_edge_canny(gray)
    contours = edge_detect.detect_edges(canny)
    draw.draw_arcs(canvas1,contours)
    cv.imwrite(f"img_of_process\\arcs.bmp",canvas1)
    contours = ED.find_contours(canny,15)
    draw.draw_arcs(canvas6,contours)
    cv.imwrite(f"img_of_process\\contours.bmp",canvas6)
    cv.imwrite(f"img_of_process\\canny.bmp",canny)
    #保存图片：所有边缘点
    
    
    
    arc_corner_point=[] #存放rdp算法计算出来的角点
    arc_index=[]#存放rdp算法计算出来的角点在原数组中的序号
    #使用自编的rdp算法
    print("多边形逼近........")
    for arc in contours:
        arc_p = []
        arc_pid = []
        #rdp(arc,range(len(arc)),1.5,arc_p,arc_pid)
        arc_p,arc_pid = ES.simplify_rdp_with_indices(arc)
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for p in arc_p:
            cv.circle(canvas2,p,2,random_color,-1)
        for i in range(len(arc_p)-1):
            cv.line(canvas2,arc_p[i],arc_p[i+1],random_color,1)
        arc_corner_point.append(arc_p)
        arc_index.append(arc_pid)
    #print(f"length of arcs:{len(arcs)}  length of arc:{len(arc_corner_point)}")
    cv.imwrite("img_of_process\\rdp.bmp",canvas2)
    arcs2=[]
    #角点切割
    _theta = 55
    _iminlength = 15
    print("角点切割........")
    for n,e in enumerate(arc_corner_point):
        e_line = []
        print(f"n: {n}")
        for i in range(len(e)-1):
            cv.circle(canvas3,e[0],3,(255,0,0),-1) #图像中弧段的开始使用蓝色表示
            # cv.line(canvas4,e[i],e[i+1],(random.randint(0,255),random.randint(0,255),random.randint(0,255)),2)
            l = np.array([e[i+1][0]-e[i][0],e[i+1][1]-e[i][1]])
            e_line.append(l)
        #print(f"e_line: {e_line}")
        e_theta = []
        for j in range(len(e_line)-1):
            t = get_angle(e_line[j],e_line[j+1])
            t = t/math.pi*180
            e_theta.append(t)
        print(f"e_theta: {e_theta}")
        if len(e_theta)==0:continue
        #构造布尔序列
        bool_sequence = []
        t_0 = e_theta[0]
        for t in e_theta:
            b = 0
            if abs(t+t_0)<(abs(t)+abs(t_0)):b=1
            bool_sequence.append(b)
        print(f"bool_sequence: {bool_sequence}")
        #构造长度序列
        c_idx = arc_index[n]
        len_sequence = []
        for i in range(len(e)):
            if i==0:continue
            length = c_idx[i]-c_idx[i-1]
            len_sequence.append(length)
        print(f"len_sequence: {len_sequence}")
        #计算角点坐标
        break_idx = []
        for k in range(len(bool_sequence)):
            #首先判断转角是否过大
            t = e_theta[k]
            if abs(t)>_theta: 
                break_idx.append(k+1)
                continue
            #判断旋转角
            b = bool_sequence[k]
            
            if b==1:
                b1 = bool_sequence[k-1]
                b2 =1
                if k<len(bool_sequence)-1:
                    b2 = bool_sequence[k+1]
                if b1 ==0:
                    break_idx.append(k+1)
                    continue
                if b2==0:
                    break_idx.append(k+1)
            #其次需要判断相邻角点之间弧段的长度是否满足要求
            _multiple = 3.5
            len1 = len_sequence[k]
            len2 = len_sequence[k+1]
            len_max = max(len1,len2)
            len_min = min(len1,len2)
            if len_max>_multiple*len_min :
                print("here")
                break_idx.append(k+1)       
        print(f"break_index: {break_idx}")  
        #弧段
        edge = contours[n]
        
        for bb in break_idx:
            cv.circle(canvas3,e[bb],3,(0,0,255),-1)
        cv.circle(canvas3,edge[len(edge)-1],3,(0,255,0),-1)
        if len(break_idx)==0:
            if len(edge)>=_iminlength:arcs2.append(edge)
        if len(break_idx)==1:
            edge1 = edge[:c_idx[break_idx[0]]]
            edge2 =  edge[c_idx[break_idx[0]]:]
            if len(edge1)>=_iminlength:arcs2.append(edge1)
            if len(edge2)>=_iminlength:arcs2.append(edge2) 
            continue
        for count,idx in enumerate(break_idx):
            print(f"count: {count}")
            edge_t = None
            if count == 0:
                edge_t = edge[:c_idx[idx]]
            elif count == len(break_idx)-1:
                edge_t1 = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                if len(edge_t1)>_iminlength:arcs2.append(edge_t1)
                edge_t = edge[c_idx[idx]:]
                #draw.draw_edge3(canvas2,edge_t,(0,0,0))
            else:
                edge_t = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
            if len(edge_t)<_iminlength:continue
            if not edge_t==None: arcs2.append(edge_t)    
    draw.draw_arcs(canvas3,arcs2)
    cv.imwrite("img_of_process\\arcs2.bmp",canvas3)

    #需要继续过滤掉一些较直的弧段
    arcs3 = []
    #采用三角不等式
    for e in arcs2:
        eh = e[0]
        em = e[len(e)//2]
        ee = e[len(e)-1]
        dis_h_m = math.sqrt((eh[0]-em[0])**2+(eh[1]-em[1])**2)
        dis_h_e = math.sqrt((eh[0]-ee[0])**2+(eh[1]-ee[1])**2)
        dis_e_m = math.sqrt((ee[0]-em[0])**2+(ee[1]-em[1])**2)
        #定义的曲率
        t = 1-dis_h_e/(dis_h_m+dis_e_m)
        print(f"t: {t}")
        _t = 0.01
        if t<_t:continue
        arcs3.append(e)
    draw.draw_arcs(canvas4,arcs3)
    cv.imwrite("img_of_process\\curved_arc.bmp",canvas4)
    # for i in range(len(arcs2)):
    #     print(f"i: {i}")
    #     draw.draw_edge3(canvas4,arcs2[i],(0,0,0))
    #     cv.imshow("1",canvas4)
    #     cv.waitKey(0)

    #对所有的弧段采用B2AC方法拟合椭圆
    #首先对所有弧段按照长度进行排序
    arcs3.sort(key = lambda x:len(x),reverse= True)

    seto = []
    while arcs3:
        a_i = arcs3.pop(0)
        G = [a_i]
        to_remove = []
        for a_j in arcs3:
            num = 0
            consistent = True
            a_j_h = a_j[0]
            a_j_m = a_j[len(a_j)//2]
            a_j_e = a_j[len(a_j)-1]
            for a_k in G:
                a_k_h = a_k[0]
                a_k_m = a_k[len(a_k)//2]
                a_k_e = a_k[len(a_k)-1]
                #判断是否有交点
                inter1 = intersection(a_k_h,a_k_e,a_k_m,a_j_m)
                inter2 = intersection(a_j_h,a_j_e,a_k_m,a_j_m)
                #print(f"inter1: {inter1} inter2: {inter2}")
                if inter1== None or inter2 == None:
                    consistent = False
                    continue
                else:
                   #判断弦是否有交点
                #    cv.line(canvas4,a_k_m,a_j_m,(0,0,255),2)
                #    cv.line(canvas4,a_k_h,a_k_e,(255,0,255),2)
                #    cv.line(canvas4,a_j_h,a_j_e,(0,0,0),2)
                    inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                    offset = 1
                    while inter3==None:
                        a_k_h = a_k[offset]
                        inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                        offset +=1
                    #计算弦交比
                    cop = lambda p1,p2 :math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
                    pq = cop(inter3,a_j_h)
                    pr = cop(inter3,a_j_e)
                    ps = cop(inter3,a_k_h)
                    pt = cop(inter3,a_k_e)
                    print(f"pq: {pq} pr: {pr} ps: {ps} pt: {pt}")
                    offset2 = 1
                    while pq==0:
                       a_j_h = a_j[offset]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       pq = cop(inter3,a_j_h)
                       offset2 += 1
                    offset3 = 2
                    while pr==0:
                       a_j_e = a_j[len(a_j)-offset3]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       pr = cop(inter3,a_j_e)
                       offset3 += 1 
                    offset4 = 1
                    while ps==0:
                       a_k_h = a_k[offset4]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       ps = cop(inter3,a_k_h)
                       offset4 +=1   
                    offset5 = 2
                    while pt==0:
                       a_k_e = a_k[len(a_k)-offset5]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       pt = cop(inter3,a_k_e)
                       offset5 += 1  
                    uj = a_j_e[1] - a_j_h[1]
                    vj = a_j_h[0] - a_j_e[0]
                    wj = (a_j_h[1] - a_j_e[1])*a_j_h[0]-(a_j_h[0] - a_j_e[0])*a_j_h[1]
                    uk = a_k_e[1] - a_k_h[1]
                    vk = a_k_h[0] - a_k_e[0]
                    wk = (a_k_h[1] - a_k_e[1])*a_k_h[0]-(a_k_h[0] - a_k_e[0])*a_k_h[1]
                    es = get_ellipse(a_j+a_k,2)
                    if es==None:
                        consistent = False
                        break
                    a = es[0]
                    b = es[1]
                    c = es[2]
                    #交比
                    
                    ppp = (pq*pr)/(ps*pt)
                    rhs = ((uj**2+vj**2)/(uk**2+vk**2))*((a*vj**2-b*uj*vj+c*uj**2)/(a*vk**2-b*uk*vk+c*uk**2))
                    print(f"ddd: {abs(ppp-rhs)}")
                    if abs(ppp-rhs)<0.5:
                        num +=1
                    else:
                        consistent = False
                        break
            if consistent and num == len(G):
                G.append(a_j)
                to_remove.append(a_j)
        for a in to_remove:
            arcs3.remove(a)
        seto.append(G)
    ell_ans =[]
    for g in seto:
        #print(f"len of g: {len(g)}")
        e_all = []
        for e in g:
            e_all +=e
        ell = get_ellipse(e_all)
        if ell==None:
            ell_ans.append(ell)
            continue
        #需要去验证椭圆是否具有较高的置信度
        #第一步计算这个椭圆的交比
        P = []
        num = 0
        A = ell[2]
        B = ell[3]
        x_c = ell[0]
        y_c = ell[1]
        the = ell[4]
        gap = 2*math.pi//5
        for i in range(5):
            p_x = (A*math.cos(the))*math.cos(i*gap)+(-B*math.sin(the))*math.sin(i*gap)+x_c
            p_y = (A*math.sin(the))*math.cos(i*gap)+(B*math.cos(the))*math.sin(i*gap)+y_c
            P.append([p_x,p_y])
        cr0 = get_cr(P)
        for pp in e_all:
            P[4] = pp
            cr_pp = get_cr(P)
            #print(f"cr_dis: {abs(cr_pp-cr0)}")
            if abs(cr_pp-cr0)<1:
                num +=1
        #计算出来的内点率
        R_c = num/(math.pi*(3*(A+B)-math.sqrt((3*A+B)*(A+3*B))))
        num2 = get_inlner_points(ell,e_all)
        R_c2 = num2/(math.pi*(3*(A+B)-math.sqrt((3*A+B)*(A+3*B))))
        print(f"内点率1: {R_c} 内点率2: {R_c2}")
        #if R_c<0.4:ell=None
        if R_c2<0.4:ell=None
        #计算张角
        label_len = 1
        if len(g)>1:label_len =0
        an_A = 0
        for e in g:
            e_h = e[0]
            e_m = e[len(e)//2]
            e_e = e[len(e)-1]
            the1 = abs(get_angle([(e_h[0]-e_m[0]),(e_h[1]-e_m[1])],[(e_e[0]-e_m[0]),(e_e[1]-e_m[1])]))
            the2 = abs(get_angle([(e_h[0]-x_c),(e_h[1]-y_c)],[(e_e[0]-x_c),(e_e[1]-y_c)]))
            print(f"the1: {the1} the2: {the2}")
            if the1 > math.pi/2:
                an_A += the2
            else:
                the2 = (math.pi*2-the2)
                an_A += the2
        print(f"label: {label_len} an_A: {an_A}")
        if label_len and an_A<math.pi*0.9:ell=None
        ell_ans.append(ell)
    return ell_ans
if __name__ == "__main__":
    filename1 = r"E:\dataset\aamed_ellipse_datasets\Random Images - Dataset #1\images\im8237.jpg"
    filename2 = r"E:\dataset\aamed_ellipse_datasets\Prasad Images - Dataset Prasad\images\027_0003.jpg"
    filename3 = r"E:\dataset\aamed_ellipse_datasets\Prasad Images - Dataset Prasad\images\033_0034.jpg"
    filename4 = r"E:\dataset\test\test\ellipse6.bmp"
    img = cv.imread(filename1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 1.0)
    # medianBlur_ksize = 3
    # gray = cv.medianBlur(gray,medianBlur_ksize)
    canvas1 = 255*np.ones_like(img)
    canvas2 = 255*np.ones_like(img)
    canvas3 = 255*np.ones_like(img)
    canvas4 = 255*np.ones_like(img)
    canvas5 = 255*np.ones_like(img)
    canvas6 = 255*np.ones_like(img)
    canvas7 = 255*np.ones_like(img)
    canny,_,_ = canny_detect.canny_detect(gray)
    # 3. 将边缘位置设为灰色（128,128,128）
    edge_mask = (canny == 255)  # 找到边缘像素位置
    canvas7[edge_mask] = [128, 128, 128]  # BGR格式的灰色
    #canny,_ = EC.mark_edge_canny(gray)
    contours = edge_detect.detect_edges(canny)
    draw.draw_arcs(canvas1,contours)
    cv.imwrite(f"img_of_process\\arcs.bmp",canvas1)
    contours = ED.find_contours(canny,15)
    draw.draw_arcs(canvas6,contours)
    cv.imwrite(f"img_of_process\\contours.bmp",canvas6)
    cv.imwrite(f"img_of_process\\canny.bmp",canny)
    #保存图片：所有边缘点
    
    
    arc_corner_point=[] #存放rdp算法计算出来的角点
    arc_index=[]#存放rdp算法计算出来的角点在原数组中的序号
    #使用自编的rdp算法
    print("多边形逼近........")
    for arc in contours:
        arc_p = []
        arc_pid = []
        rdp(arc,range(len(arc)),1.5,arc_p,arc_pid)
        arc_p,arc_pid = ES.simplify_rdp_with_indices(arc)
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for p in arc_p:
            cv.circle(canvas2,p,2,random_color,-1)
        for i in range(len(arc_p)-1):
            cv.line(canvas2,arc_p[i],arc_p[i+1],random_color,1)
        arc_corner_point.append(arc_p)
        arc_index.append(arc_pid)
    #print(f"length of arcs:{len(arcs)}  length of arc:{len(arc_corner_point)}")
    cv.imwrite("img_of_process\\rdp.bmp",canvas2)
    arcs2=[]
    #角点切割
    _theta = 45
    _iminlength = 15
    print("角点切割........")
    for n,e in enumerate(arc_corner_point):
        e_line = []
        print(f"n: {n}")
        for i in range(len(e)-1):
            cv.circle(canvas3,e[0],3,(255,0,0),-1) #图像中弧段的开始使用蓝色表示
            # cv.line(canvas4,e[i],e[i+1],(random.randint(0,255),random.randint(0,255),random.randint(0,255)),2)
            l = np.array([e[i+1][0]-e[i][0],e[i+1][1]-e[i][1]])
            e_line.append(l)
        #print(f"e_line: {e_line}")
        e_theta = []
        for j in range(len(e_line)-1):
            t = get_angle(e_line[j],e_line[j+1])
            t = t/math.pi*180
            e_theta.append(t)
        print(f"e_theta: {e_theta}")
        if len(e_theta)==0:continue
        #构造布尔序列
        bool_sequence = []
        t_0 = e_theta[0]
        for t in e_theta:
            b = 0
            if abs(t+t_0)<(abs(t)+abs(t_0)):b=1
            bool_sequence.append(b)
        print(f"bool_sequence: {bool_sequence}")
        #构造长度序列
        c_idx = arc_index[n]
        len_sequence = []
        for i in range(len(e)):
            if i==0:continue
            length = c_idx[i]-c_idx[i-1]
            len_sequence.append(length)
        print(f"len_sequence: {len_sequence}")
        #计算角点坐标
        break_idx = []
        for k in range(len(bool_sequence)):
            #首先判断转角是否过大
            t = e_theta[k]
            if abs(t)>_theta: 
                break_idx.append(k+1)
                continue
            #判断旋转角
            b = bool_sequence[k]
            
            if b==1:
                b1 = bool_sequence[k-1]
                b2 =1
                if k<len(bool_sequence)-1:
                    b2 = bool_sequence[k+1]
                if b1 ==0:
                    break_idx.append(k+1)
                    continue
                if b2==0:
                    break_idx.append(k+1)
            #其次需要判断相邻角点之间弧段的长度是否满足要求
            _multiple = 3.5
            len1 = len_sequence[k]
            len2 = len_sequence[k+1]
            len_max = max(len1,len2)
            len_min = min(len1,len2)
            if len_max>_multiple*len_min :
                print("here")
                break_idx.append(k+1)       
        print(f"break_index: {break_idx}")  
        #弧段
        edge = contours[n]
        
        for bb in break_idx:
            cv.circle(canvas3,e[bb],3,(0,0,255),-1)
        cv.circle(canvas3,edge[len(edge)-1],3,(0,255,0),-1)
        if len(break_idx)==0:
            if len(edge)>=_iminlength:arcs2.append(edge)
        if len(break_idx)==1:
            edge1 = edge[:c_idx[break_idx[0]]]
            edge2 =  edge[c_idx[break_idx[0]]:]
            if len(edge1)>=_iminlength:arcs2.append(edge1)
            if len(edge2)>=_iminlength:arcs2.append(edge2) 
            continue
        for count,idx in enumerate(break_idx):
            print(f"count: {count}")
            edge_t = None
            if count == 0:
                edge_t = edge[:c_idx[idx]]
            elif count == len(break_idx)-1:
                edge_t1 = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                if len(edge_t1)>_iminlength:arcs2.append(edge_t1)
                edge_t = edge[c_idx[idx]:]
                #draw.draw_edge3(canvas2,edge_t,(0,0,0))
            else:
                edge_t = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
            if len(edge_t)<_iminlength:continue
            if not edge_t==None: arcs2.append(edge_t)    
    draw.draw_arcs(canvas3,arcs2)
    cv.imwrite("img_of_process\\arcs2.bmp",canvas3)

    #需要继续过滤掉一些较直的弧段
    arcs3 = []
    #采用三角不等式
    for e in arcs2:
        eh = e[0]
        em = e[len(e)//2]
        ee = e[len(e)-1]
        dis_h_m = math.sqrt((eh[0]-em[0])**2+(eh[1]-em[1])**2)
        dis_h_e = math.sqrt((eh[0]-ee[0])**2+(eh[1]-ee[1])**2)
        dis_e_m = math.sqrt((ee[0]-em[0])**2+(ee[1]-em[1])**2)
        #定义的曲率
        t = 1-dis_h_e/(dis_h_m+dis_e_m)
        print(f"t: {t}")
        _t = 0.01  #原文中给出的范围为0.01-0.04
        if t<_t:continue
        arcs3.append(e)
    draw.draw_arcs(canvas4,arcs3)
    cv.imwrite("img_of_process\\curved_arc.bmp",canvas4)
    # for i in range(len(arcs2)):
    #     print(f"i: {i}")
    #     draw.draw_edge3(canvas4,arcs2[i],(0,0,0))
    #     cv.imshow("1",canvas4)
    #     cv.waitKey(0)

    #对所有的弧段采用B2AC方法拟合椭圆
    #首先对所有弧段按照长度进行排序
    arcs3.sort(key = lambda x:len(x),reverse= True)

    seto = []
    while arcs3:
        a_i = arcs3.pop(0)
        G = [a_i]
        to_remove = []
        for a_j in arcs3:
            num = 0
            consistent = True
            a_j_h = a_j[0]
            a_j_m = a_j[len(a_j)//2]
            a_j_e = a_j[len(a_j)-1]
            for a_k in G:
                a_k_h = a_k[0]
                a_k_m = a_k[len(a_k)//2]
                a_k_e = a_k[len(a_k)-1]
                #判断是否有交点
                inter1 = intersection(a_k_h,a_k_e,a_k_m,a_j_m)
                inter2 = intersection(a_j_h,a_j_e,a_k_m,a_j_m)
                #print(f"inter1: {inter1} inter2: {inter2}")
                if inter1== None or inter2 == None:
                    consistent = False
                    continue
                else:
                    #判断弦是否有交点
                #    cv.line(canvas4,a_k_m,a_j_m,(0,0,255),2)
                #    cv.line(canvas4,a_k_h,a_k_e,(255,0,255),2)
                #    cv.line(canvas4,a_j_h,a_j_e,(0,0,0),2)
                    inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                    offset = 1
                    while inter3==None:
                        a_k_h = a_k[offset]
                        inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                        offset +=1
                    #计算弦交比
                    cop = lambda p1,p2 :math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
                    pq = cop(inter3,a_j_h)
                    pr = cop(inter3,a_j_e)
                    ps = cop(inter3,a_k_h)
                    pt = cop(inter3,a_k_e)
                    print(f"pq: {pq} pr: {pr} ps: {ps} pt: {pt}")
                    offset2 = 1
                    while pq==0:
                       a_j_h = a_j[offset]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       pq = cop(inter3,a_j_h)
                       offset2 += 1
                    offset3 = 2
                    while pr==0:
                       a_j_e = a_j[len(a_j)-offset3]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       pr = cop(inter3,a_j_e)
                       offset3 += 1 
                    offset4 = 1
                    while ps==0:
                       a_k_h = a_k[offset4]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       ps = cop(inter3,a_k_h)
                       offset4 +=1   
                    offset5 = 2
                    while pt==0:
                       a_k_e = a_k[len(a_k)-offset5]
                       inter3 = get_intersection_point2(a_j_h,a_j_e,a_k_h,a_k_e)
                       pt = cop(inter3,a_k_e)
                       offset5 += 1 
                    uj = a_j_e[1] - a_j_h[1]
                    vj = a_j_h[0] - a_j_e[0]
                    wj = (a_j_h[1] - a_j_e[1])*a_j_h[0]-(a_j_h[0] - a_j_e[0])*a_j_h[1]
                    uk = a_k_e[1] - a_k_h[1]
                    vk = a_k_h[0] - a_k_e[0]
                    wk = (a_k_h[1] - a_k_e[1])*a_k_h[0]-(a_k_h[0] - a_k_e[0])*a_k_h[1]
                    es = get_ellipse(a_j+a_k,2)
                    if es==None:
                        consistent = False
                        break
                    a = es[0]
                    b = es[1]
                    c = es[2]
                    #交比
                    
                    ppp = (pq*pr)/(ps*pt)
                    rhs = ((uj**2+vj**2)/(uk**2+vk**2))*((a*vj**2-b*uj*vj+c*uj**2)/(a*vk**2-b*uk*vk+c*uk**2))
                    print(f"ddd: {abs(ppp-rhs)}")
                    if abs(ppp-rhs)<0.3:
                        num +=1
                    else:
                        consistent = False
                        break
            if consistent and num == len(G):
                G.append(a_j)
                to_remove.append(a_j)
        for a in to_remove:
            arcs3.remove(a)
        seto.append(G)
    ell_ans =[]
    ell_jude = []
    print(f"len of seto: {len(seto)}")
    for g in seto:
        #print(f"len of g: {len(g)}")
        e_all = []
        for e in g:
            e_all +=e
        ell = get_ellipse(e_all)
        if ell==None:
            ell_ans.append(ell)
            continue
        #需要去验证椭圆是否具有较高的置信度
        #第一步计算这个椭圆的交比
        P = []
        num = 0
        A = ell[2]
        B = ell[3]
        x_c = ell[0]
        y_c = ell[1]
        the = ell[4]
        # gap = 2*math.pi//5
        # for i in range(5):
        #     p_x = (A*math.cos(the))*math.cos(i*gap)+(-B*math.sin(the))*math.sin(i*gap)+x_c
        #     p_y = (A*math.sin(the))*math.cos(i*gap)+(B*math.cos(the))*math.sin(i*gap)+y_c
        #     P.append([p_x,p_y])
        # cr0 = get_cr(P)
        # for pp in e_all:
        #     P[4] = pp
        #     cr_pp = get_cr(P)
        #     #print(f"cr_dis: {abs(cr_pp-cr0)}")
        #     if abs(cr_pp-cr0)<1:
        #         num +=1
        # #计算出来的内点率
        # R_c = num/(math.pi*(3*(A+B)-math.sqrt((3*A+B)*(A+3*B))))
        num2 = get_inlner_points(ell,e_all)
        R_c2 = num2/(math.pi*(3*(A+B)-math.sqrt((3*A+B)*(A+3*B))))
        print(f"内点率2: {R_c2}")
        ell_jude.append(R_c2)
        #if R_c<0.4:ell=None
        if R_c2<0.4:ell=None
        #计算张角
        label_len = 1
        if len(g)>1:label_len =0
        an_A = 0
        for e in g:
            e_h = e[0]
            e_m = e[len(e)//2]
            e_e = e[len(e)-1]
            the1 = abs(get_angle([(e_h[0]-e_m[0]),(e_h[1]-e_m[1])],[(e_e[0]-e_m[0]),(e_e[1]-e_m[1])]))
            the2 = abs(get_angle([(e_h[0]-x_c),(e_h[1]-y_c)],[(e_e[0]-x_c),(e_e[1]-y_c)]))
            print(f"the1: {the1} the2: {the2}")
            if the1 > math.pi/2:
                an_A += the2
            else:
                the2 = (math.pi*2-the2)
                an_A += the2
        print(f"label: {label_len} an_A: {an_A}")
        #if label_len and an_A<math.pi*0.9:ell=None
        ell_ans.append(ell)
    # for g in G:
    #     print(f"g1: {g[0]}")
    #椭圆聚类
    #首先按照内点率2进行排序
    ell_after_cluster = []
    while ell_ans:
        ell0 = ell_ans.pop(0)
        ell_after_cluster.append(ell0)
        to_remove = []
        for i,e1 in enumerate(ell_ans):
            label = 1
            num = 0
            while label:
                for j,e2 in enumerate(ell_after_cluster):
                    #判断两个椭圆是否满足聚类要求
                    d = ellipse_cluster.ell_cluster(e1,e2)
                    if d:
                        label = 0
                        s1 = ell_jude[i]
                        s2 = ell_jude[j]
                        if s1<=s2:to_remove.append(e1)
                        else:
                            ell_after_cluster[j] = e1
                            to_remove.append(e1)
                        break
                    else:
                        num +=1
                if label and num==len(ell_after_cluster):
                    to_remove.append(e1)
                    ell_after_cluster.append(e1)
                    label = 0
        for ee in to_remove:
            ell_ans.remove(ee)
    for i,ell in enumerate(ell_after_cluster):
        print(f"ell: {ell}")
        if ell ==None:continue
            #cv.ellipse(canvas4,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,255,0),1)
        #计算椭圆得分
        #score = get_ell_score(ell,G[i])
        #print(f"score: {score}")
        #if score <0.95: continue
        cv.ellipse(canvas4,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,255,0),2)
        cv.ellipse(canvas7,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,255,0),2)
    cv.imwrite("img_of_process\\curved_arc_and_ells.bmp",canvas4)
    cv.imwrite("img_of_process\\ell_ans.bmp",canvas7)
    eng.quit()










