import math

def ell_cluster(e1,e2):
    if e1==None or e2==None:return 0
    e1_x = e1[0]
    e1_y = e1[1]
    e1_A = e1[2]
    e1_B = e1[3]
    e1_an = e1[4]
    e2_x = e2[0]
    e2_y = e2[1]
    e2_A = e2[2]
    e2_B = e2[3]
    e2_an = e2[4]
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    if math.sqrt((e1_x-e2_x)**2+(e1_y-e2_y)**2)<min(e1_B,e2_B)*0.1:
        d1 = 1
    #else:print("中心位置不满足聚类要求")
    if abs(e1_A-e2_A)<max(e1_A,e2_A)*0.1:d2 = 1
    #else:print("长轴不满足聚类要求")
    if abs(e1_B-e2_B)<max(e1_B,e2_B)*0.1:d3 = 1
    #else:print("短轴不满足聚类要求")
    if (e1_B/e1_A)<0.9 and (e2_B/e2_A)<0.9:
        if abs(e1_an-e2_an)/math.pi<0.1:d4 = 1
        #else:print("旋转角不满足聚类要求")
    return d1 and d2 and d3 and d4

