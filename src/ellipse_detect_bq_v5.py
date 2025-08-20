#说明：对弯曲弧段的总的旋转角度加以限制，去除弧段聚类和拟合后的椭圆聚类的过程
import numpy as np
import math
import cv2 as cv
import ED
import ES
import EC
import draw
import canny_detect
import random
import ellipse_cluster
#f1:计算向量之间的夹角
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
#f2:计算两个线段的交点
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
#f3: rdp算法
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
#f4: 计算直线的交点
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
#f5: 计算CNC
def get_CNC(q11,q12,q21,q22,q31,q32):
    p1 = get_intersection_point2(q11,q12,q31,q32)
    p2 = get_intersection_point2(q11,q12,q21,q22)
    p3 = get_intersection_point2(q31,q32,q21,q22)
    if p1 == None or p2 == None or p3 == None: return 0
    cnc_pq = 1
    # 首先计算q11、q12
    det_p = p1[0] * p2[1] - p1[1] * p2[0]
    if det_p==0:
        cnc_pq = 1
    else:    
        x1 = (q11[0] * p2[1] - q11[1] * p2[0]) / det_p
        y1 = (-q11[0] * p1[1] + q11[1] * p1[0]) / det_p
        x2 = (q12[0] * p2[1] - q12[1] * p2[0]) / det_p
        y2 = (-q12[0] * p1[1] + q12[1] * p1[0]) / det_p
        if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
            return 0
        cnc_pq = (y1 * y2) / (x1 * x2) * cnc_pq
    

    # 计算q21、q22
    det_p = p2[0] * p3[1] - p2[1] * p3[0]
    if det_p==0:
        pass
    else:
        x1 = (q21[0] * p3[1] - q21[1] * p3[0]) / det_p
        y1 = (-q21[0] * p2[1] + q21[1] * p2[0]) / det_p
        x2 = (q22[0] * p3[1] - q22[1] * p3[0]) / det_p
        y2 = (-q22[0] * p2[1] + q22[1] * p2[0]) / det_p
        if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
            return 0
        cnc_pq = (y1 * y2) / (x1 * x2) * cnc_pq

    # 计算q31、q32
    det_p = p3[0] * p1[1] - p3[1] * p1[0]
    if det_p==0:
        pass
    else:
        x1 = (q31[0] * p1[1] - q31[1] * p1[0]) / det_p
        y1 = (-q31[0] * p3[1] + q31[1] * p3[0]) / det_p
        x2 = (q32[0] * p1[1] - q32[1] * p1[0]) / det_p
        y2 = (-q32[0] * p3[1] + q32[1] * p3[0]) / det_p
        if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
            return 0
        cnc_pq = (y1 * y2) / (x1 * x2) * cnc_pq
    #print(f"cnc: {cnc_pq}")
    return cnc_pq
#f6: 椭圆参数转换函数(5个参数)
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
#f7: 椭圆参数转换函数(二次标准型)
def param_diversion(es):
    x = es[0][0]
    y = es[0][1]
    (width,height) = es[1]
    if width >= height:
        A = width / 2.0
        B = height / 2.0
        theta_deg = es[2]  # 已是顺时针
    else:
        A = height / 2.0
        B = width / 2.0
        theta_deg = es[2] - 90  # 长轴沿着 height，要校正成长轴的方向
    an = theta_deg/180*math.pi
    C = math.cos(an)
    S = math.sin(an)
    a = (B**2)*C**2+(A**2)*S**2 #x**2
    b = (B**2)*(2*C*S)+(A**2)*(-2*S*C)#xy
    #y**2
    c = (B**2)*(S**2)+(A**2)*(C**2)
    #x
    d = (B**2)*(-2*x*C**2-2*C*S*y)+(A**2)*(-2*x*S**2+2*S*C*y)
    #y
    e = (B**2)*(-2*y*S**2-2*C*S*x)+(A**2)*(-2*y*C**2+2*S*C*x)
    #1
    f = (B**2)*(C**2*x**2+S**2*y**2+2*C*S*x*y)+(A**2)*(S**2*x**2+C**2*y**2-2*C*S*x*y)-A**2*B**2
    
    return a,b,c,d,e,f
#f8: 计算两个直线中垂线的交点
def get_intersection_perpendicular_bisector(l1h,l1e,l2h,l2e):
    dx1 = l1h[0]-l1e[0]
    if dx1==0:
        inter_y1 = (l1h[1]+l1e[1])/2
    else:
        dy1 = l1h[1]-l1e[1]
        k1 = dy1/dx1
    dx2 = l2h[0]-l2e[0]
    if dx2==0:
        inter_y2 = (l2h[1]+l2e[1])/2
    else:
        dy2 = l2h[1]-l2e[1]
        k2 = dy2/dx2
    if dx1==0 and dx2==0:
        return None
    elif dx1==0:
        inter_x = (inter_y1-l2e[1])/k2+l2e[0]
        inter_y = inter_y1
    elif dx2==0:
        inter_x = (inter_y2-l1e[1])/k1+l1e[0]
        inter_y = inter_y2
    elif k1==k2:
        return [(l1h[0]+l1e[0])/2,(l1h[1]+l1e[1])/2]  
    else:
        inter_x = ((k1*l1e[0]+k2*l2e[0])+l2e[1]-l1e[1])/(k1-k2)
        inter_y = k1*(inter_x-l1e[0])+l1e[1]
    return [inter_x,inter_y]
#f9: 计算拟合出来的椭圆的内点率
def get_inlner_points(ell,e):
    num = 0
    for p in e:
        x = p[0]
        y = p[1]
        F = ((x*math.cos(ell[4])-ell[0]*math.cos(ell[4])+y*math.sin(ell[4])-ell[1]*math.sin(ell[4])))**2/(ell[2]**2)+((-x*math.sin(ell[4])+ell[0]*math.sin(ell[4])+y*math.cos(ell[4])-ell[1]*math.cos(ell[4])))**2/(ell[3]**2)
        if abs(F-1)<0.1:
            num +=1
    return num
#f10: 判断两个弧段的首尾是否非常相近
def jude_head_end_of_arcs(e1h,e1e,e2h,e2e):
    j1 = 0
    j2 = 0
    j3 = 0
    j4 = 0
    dis1 = max(e1h[0]-e2h[0],e1h[1]-e2h[1])
    dis2 = max(e1h[0]-e2e[0],e1h[1]-e2e[1])
    dis3 = max(e1e[0]-e2h[0],e1e[1]-e2h[1])
    dis4 = max(e1e[0]-e2e[0],e1e[1]-e2e[1])
    if dis1<5:j1 = 1
    if dis2<5:j2 = 1
    if dis3<5:j3 = 1
    if dis4<5:j4 = 1
    return j1+j2+j3+j4

#测试
def get_select_arc(img):
    canvas1 = 255*np.ones_like(img)
    canvas2 = 255*np.ones_like(img)
    canvas3 = 255*np.ones_like(img)
    canvas4 = 255*np.ones_like(img)
    canvas5 = 255*np.ones_like(img)
    canvas6 = 255*np.ones_like(img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),1)
    #,low_percent=0.78,high_percent=0.92
    canny,DX,DY = canny_detect.canny_detect(gray)
    edge_mask = (canny == 255)  # 找到边缘像素位置
    canvas6[edge_mask] = [128, 128, 128]  # BGR格式的灰色
    cv.imwrite(f"jcj_bq_v5\\canny1.bmp",canny)
    # canny,_ = EC.mark_edge_canny(gray)
    # cv.imwrite(f"jcj_bq_v3\\canny2.bmp",canny)
    _minlen = 150
    #多边形逼近的方法得到弯曲弧段
    arcs = ED.find_contours(canny,_minlen)
    draw.draw_arcs(canvas1,arcs)
    cv.imwrite(f"jcj_bq_v5\\arcs.bmp",canvas1)
    arc_corner_point=[] #存放rdp算法计算出来的角点
    arc_index=[]#存放rdp算法计算出来的角点在原数组中的序号
    #使用自编的rdp算法
    print("多边形逼近........")
    for arc in arcs:
        arc_p = []
        arc_pid = []
        rdp(arc,range(len(arc)),5,arc_p,arc_pid)
        #arc_p,arc_pid = ES.simplify_rdp_with_indices(arc)
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for p in arc_p:
            cv.circle(canvas2,p,2,random_color,-1)
        for i in range(len(arc_p)-1):
            cv.line(canvas2,arc_p[i],arc_p[i+1],random_color,1)
        arc_corner_point.append(arc_p)
        arc_index.append(arc_pid)
    #print(f"length of arcs:{len(arcs)}  length of arc:{len(arc_corner_point)}")
    cv.imwrite("jcj_bq_v5\\rdp.bmp",canvas2)
    arcs2=[]
    arcs2_spin = []
    #角点切割
    _theta = 45  #这个后面再继续测试
    _iminlength = _minlen
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
            _multiple = 3
            len1 = len_sequence[k]
            len2 = len_sequence[k+1]
            len_max = max(len1,len2)
            len_min = min(len1,len2)
            if len_max>_multiple*len_min :
                #print("here")
                break_idx.append(k+1)       
        print(f"break_index: {break_idx}")  
        #弧段
        edge = arcs[n]
        
        for bb in break_idx:
            cv.circle(canvas3,e[bb],3,(0,0,255),-1)
        cv.circle(canvas3,edge[len(edge)-1],3,(0,255,0),-1)
        if len(break_idx)==0:
            if len(edge)>=_iminlength:
                spin_all = abs(sum(e_theta))
                arcs2_spin.append(spin_all)
                arcs2.append(edge)
        if len(break_idx)==1:
            edge1 = edge[:c_idx[break_idx[0]]]
            edge2 =  edge[c_idx[break_idx[0]]:]
            spin_all1 = abs(sum(e_theta[:break_idx[0]]))
            spin_all2 = abs(sum(e_theta[break_idx[0]+1:]))
            if len(edge1)>=_iminlength:
                arcs2.append(edge1)
                arcs2_spin.append(spin_all1)
            if len(edge2)>=_iminlength:
                arcs2.append(edge2) 
                arcs2_spin.append(spin_all2)
            continue
        for count,idx in enumerate(break_idx):
            #print(f"count: {count}")
            edge_t = None
            if count == 0:
                edge_t = edge[:c_idx[idx]]
                spin_all = abs(sum(e_theta[:idx]))
            elif count == len(break_idx)-1:
                edge_t1 = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                spin_all1 = abs(sum(e_theta[break_idx[count-1]+1:idx]))
                if len(edge_t1)>_iminlength:
                    arcs2.append(edge_t1)
                    arcs2_spin.append(spin_all1)
                edge_t = edge[c_idx[idx]:]
                spin_all = abs(sum(e_theta[idx+1:]))
                #draw.draw_edge3(canvas2,edge_t,(0,0,0))
            else:
                edge_t = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                spin_all = abs(sum(e_theta[break_idx[count-1]+1:idx]))
            if len(edge_t)<_iminlength:continue
            if not edge_t==None: 
                arcs2.append(edge_t) 
                arcs2_spin.append(spin_all)   
    draw.draw_arcs(canvas3,arcs2)
    cv.imwrite("jcj_bq_v5\\arcs2.bmp",canvas3)

    #过滤掉一些较短较直的弧段
    arcs3 = []
    for i,e in enumerate(arcs2):
        e_spin = arcs2_spin[i]
        print(e_spin)
        eh = e[0]
        em = e[len(e)//2]
        ee = e[len(e)-1]
        dis1 = math.sqrt((eh[0]-ee[0])**2+(eh[1]-ee[1])**2)
        dis2 = math.sqrt((eh[0]-em[0])**2+(eh[1]-em[1])**2)
        dis3 = math.sqrt((ee[0]-em[0])**2+(ee[1]-em[1])**2)
        t = 1 - dis1/(dis2+dis3)
        if t<0.1: continue
        if e_spin<245: continue
        arcs3.append(e)
    #canvas4
    draw.draw_arcs(canvas4,arcs3)
    cv.imwrite("jcj_bq_v5\\curved_arcs.bmp",canvas4)
    #弧段聚类
    arcs3.sort(key = lambda x:len(x),reverse=True)#降序排列

    ells = []
    #对筛选出来的弧段进行椭圆拟合
    for e in arcs3:
        ell = param_diversion_ell2ell(cv.fitEllipse(np.array(e)))
        ells.append(ell)
    
    #测试
    for ell in ells:
        cv.ellipse(img,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,100,255),2)
    cv.imwrite("jcj_bq_v5\\all_ells.bmp",img)

    #判断出弧段的极性
    r_dis = []
    for i,ell in enumerate(ells):
        cx = ell[0]
        cy = ell[1]
        edge = arcs3[i]
        num = 0
        r = 0
        for p in edge:
            #print(f"p: {p}")
            dis = math.sqrt((p[0]-cx)**2+(p[1]-cy)**2)
            r += dis
            num += 1
        r_average = r/num
        if r_average<50:
            r_dis.append(None)
            continue
        print(f"r_avergae: {r_average}")
        r_dis.append(r_average)
    print(f"r_dis: {r_dis}")
    min_r_dis = 2048
    min_idx = None
    for i,r in enumerate(r_dis):
        if r == None:continue
        if r<min_r_dis:
            min_r_dis = r
            min_idx = i
    if min_idx == None:return canny,None,ells
    return canny,arcs3[min_idx],ells
    
if __name__ == "__main__":
    filename1 = r"C:\Users\34991\Documents\WeChat Files\wxid_49d84dg7g43b22\FileStorage\File\2025-08\djh\Connector.bmp"
    filename2 = r"F:\test5_14\L1_1.bmp"
    img = cv.imread(filename1)
    #创建白色画布
    canvas1 = 255*np.ones_like(img)
    canvas2 = 255*np.ones_like(img)
    canvas3 = 255*np.ones_like(img)
    canvas4 = 255*np.ones_like(img)
    canvas5 = 255*np.ones_like(img)
    canvas6 = 255*np.ones_like(img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),1)
    #,low_percent=0.78,high_percent=0.92
    canny,DX,DY = canny_detect.canny_detect(gray)
    edge_mask = (canny == 255)  # 找到边缘像素位置
    canvas6[edge_mask] = [128, 128, 128]  # BGR格式的灰色
    cv.imwrite(f"..\\jcj_bq_v5\\canny1.bmp",canny)
    # canny,_ = EC.mark_edge_canny(gray)
    # cv.imwrite(f"jcj_bq_v3\\canny2.bmp",canny)
    _minlen = 150
    #多边形逼近的方法得到弯曲弧段
    arcs = ED.find_contours(canny,_minlen)
    draw.draw_arcs(canvas1,arcs)
    cv.imwrite(f"..\\jcj_bq_v5\\arcs.bmp",canvas1)
    arc_corner_point=[] #存放rdp算法计算出来的角点
    arc_index=[]#存放rdp算法计算出来的角点在原数组中的序号
    #使用自编的rdp算法
    print("多边形逼近........")
    for arc in arcs:
        arc_p = []
        arc_pid = []
        rdp(arc,range(len(arc)),5,arc_p,arc_pid)
        #arc_p,arc_pid = ES.simplify_rdp_with_indices(arc)
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for p in arc_p:
            cv.circle(canvas2,p,2,random_color,-1)
        for i in range(len(arc_p)-1):
            cv.line(canvas2,arc_p[i],arc_p[i+1],random_color,1)
        arc_corner_point.append(arc_p)
        arc_index.append(arc_pid)
    #print(f"length of arcs:{len(arcs)}  length of arc:{len(arc_corner_point)}")
    cv.imwrite("..\\jcj_bq_v5\\rdp.bmp",canvas2)
    arcs2=[]
    arcs2_spin = []
    #角点切割
    _theta = 45  #这个后面再继续测试
    _iminlength = _minlen
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
            _multiple = 3
            len1 = len_sequence[k]
            len2 = len_sequence[k+1]
            len_max = max(len1,len2)
            len_min = min(len1,len2)
            if len_max>_multiple*len_min :
                #print("here")
                break_idx.append(k+1)       
        print(f"break_index: {break_idx}")  
        #弧段
        edge = arcs[n]
        
        for bb in break_idx:
            cv.circle(canvas3,e[bb],3,(0,0,255),-1)
        cv.circle(canvas3,edge[len(edge)-1],3,(0,255,0),-1)
        if len(break_idx)==0:
            if len(edge)>=_iminlength:
                spin_all = abs(sum(e_theta))
                arcs2_spin.append(spin_all)
                arcs2.append(edge)
        if len(break_idx)==1:
            edge1 = edge[:c_idx[break_idx[0]]]
            edge2 =  edge[c_idx[break_idx[0]]:]
            spin_all1 = abs(sum(e_theta[:break_idx[0]-1]))
            spin_all2 = abs(sum(e_theta[break_idx[0]:]))
            if len(edge1)>=_iminlength:
                arcs2.append(edge1)
                arcs2_spin.append(spin_all1)
            if len(edge2)>=_iminlength:
                arcs2.append(edge2) 
                arcs2_spin.append(spin_all2)
            continue
        for count,idx in enumerate(break_idx):
            #print(f"count: {count}")
            edge_t = None
            if count == 0:
                edge_t = edge[:c_idx[idx]]
                spin_all = abs(sum(e_theta[:idx-1]))
            elif count == len(break_idx)-1:
                edge_t1 = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                spin_all1 = abs(sum(e_theta[break_idx[count-1]:idx-1]))
                if len(edge_t1)>_iminlength:
                    arcs2.append(edge_t1)
                    arcs2_spin.append(spin_all1)
                edge_t = edge[c_idx[idx]:]
                spin_all = abs(sum(e_theta[idx:]))
                #draw.draw_edge3(canvas2,edge_t,(0,0,0))
            else:
                edge_t = edge[c_idx[break_idx[count-1]]:c_idx[idx]]
                spin_all = abs(sum(e_theta[break_idx[count-1]:idx-1]))
            if len(edge_t)<_iminlength:continue
            if not edge_t==None: 
                arcs2.append(edge_t) 
                arcs2_spin.append(spin_all)   
    draw.draw_arcs(canvas3,arcs2)
    cv.imwrite("..\\jcj_bq_v5\\arcs2.bmp",canvas3)

    #过滤掉一些较短较直的弧段
    arcs3 = []
    for i,e in enumerate(arcs2):
        e_spin = arcs2_spin[i]
        print(e_spin)
        eh = e[0]
        em = e[len(e)//2]
        ee = e[len(e)-1]
        dis1 = math.sqrt((eh[0]-ee[0])**2+(eh[1]-ee[1])**2)
        dis2 = math.sqrt((eh[0]-em[0])**2+(eh[1]-em[1])**2)
        dis3 = math.sqrt((ee[0]-em[0])**2+(ee[1]-em[1])**2)
        t = 1 - dis1/(dis2+dis3)
        if t<0.1: continue
        if e_spin<245: continue  #旋转角度限制
        arcs3.append(e)
    #canvas4
    draw.draw_arcs(canvas4,arcs3)
    cv.imwrite("jcj_bq_v5\\curved_arcs.bmp",canvas4)
    #弧段聚类
    arcs3.sort(key = lambda x:len(x),reverse=True)#降序排列

    ells = []
    #对筛选出来的弧段进行椭圆拟合
    for e in arcs3:
        ell = param_diversion_ell2ell(cv.fitEllipse(np.array(e)))
        ells.append(ell)
    
    #测试
    for ell in ells:
        cv.ellipse(img,(int(ell[0]),int(ell[1])),(int(ell[2]),int(ell[3])),ell[4]/math.pi*180,0,360,(0,100,255),2)
    cv.imwrite("..\\jcj_bq_v5\\all_ells.bmp",img)

    #判断出弧段的极性
    r_dis = []
    for i,ell in enumerate(ells):
        cx = ell[0]
        cy = ell[1]
        edge = arcs3[i]
        num = 0
        r = 0
        for p in edge:
            #print(f"p: {p}")
            dis = math.sqrt((p[0]-cx)**2+(p[1]-cy)**2)
            r += dis
            num += 1
        r_average = r/num
        if r_average<50:
            r_dis.append(None)
            continue
        print(f"r_avergae: {r_average}")
        r_dis.append(r_average)
    print(f"r_dis: {r_dis}")
    min_r_dis = 2048
    min_idx = None
    for i,r in enumerate(r_dis):
        if r == None:continue
        if r<min_r_dis:
            min_r_dis = r
            min_idx = i
    e_select = arcs3[min_idx]
    draw.draw_edge(canvas6,e_select,(0,0,0))
    cv.imwrite("..\\jcj_bq_v5\\select_arcs.bmp",canvas6)