import numpy as np
import cv2 as cv
import random
import canny_detect as cd
import edge_detect as ed

def draw_edge(img,edge,color):
    for p in edge:
        x = p[0]
        y = p[1]
        img[y,x] = color

def draw_arcs(img,arcs):
    for edge in arcs:
        color = np.array([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
        draw_edge(img,edge,color)

#采用连线的方式画弧段
def draw_edge3(img,edge,color):
    #print(type(tuple(color)))
    for i in range(len(edge)):
        if i == len(edge)-1:break
        cv.line(img,edge[i],edge[i+1],color,2)    
#采用连线的方式画弧段
def draw_edge2(img,edge,color):
    #print(type(tuple(color)))
    for i in range(len(edge)):
        cv.circle(img,edge[i],2,color,-1)
        if i == len(edge)-1:break
        cv.line(img,edge[i],edge[i+1],color,1)

def draw_arcs_line(img,arcs):
    for edge in arcs:
        color = tuple(map(int, [random.randint(0, 255) for _ in range(3)]))
        draw_edge2(img,edge,color)

if __name__ =="__main__":
    filename1 = r"E:\dataset\test\test\ellipse1.bmp"
    img = cv.imread(filename1)
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    #创建白色画布
    canvas = 255*np.ones((height,width,3),dtype=np.uint8)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    canny,_,_ = cd.canny_detect(gray)

    #连接边缘点
    arcs = ed.detect_edges(canny)
    draw_arcs(canvas,arcs)

    print(f"len of arcs: {len(arcs)}")
    cv.imshow("canvas",canvas)
    cv.imshow("img",img)
    cv.imshow("canny",canny)
    cv.waitKey(0)