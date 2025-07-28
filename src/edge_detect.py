import numpy as np
import cv2 as cv
import canny_detect
import random
# 八邻域方向
directions = [
    (1, 0), (1, -1), (0, -1), (-1, -1),
    (-1, 0), (-1, 1), (0, 1), (1, 1)
]

# Python 版的边缘检测函数
def detect_edges(src, iminlength=10):
    if len(src.shape) > 2:
        raise ValueError("输入图像必须是灰度图像")
    
    # 克隆图像
    edge = src.copy()
    arc = []
    rows, cols = edge.shape

    for j in range(rows - 1):
        for i in range(cols - 1):
            b_pt = (i, j)
            c_pt = (i, j)
            
            # 判断是否是边缘点
            if edge[c_pt[1], c_pt[0]] == 255:
                edge_t = []
                edge_t.append(c_pt)
                edge[c_pt[1], c_pt[0]] = 0
                curr_d = 0
                tra_flag = False

                while not tra_flag:
                    for count in range(8):
                        curr_d = (curr_d + 8) % 8  # 确保方向索引在 0-7 之间
                        direction = directions[curr_d]

                        c_pt = (b_pt[0] + direction[0], b_pt[1] + direction[1])
                        # 检查边界条件
                        if 0 <= c_pt[0] < cols and 0 <= c_pt[1] < rows:
                            if edge[c_pt[1], c_pt[0]] == 255:
                                curr_d -= 2
                                edge_t.append(c_pt)
                                edge[c_pt[1], c_pt[0]] = 0
                                b_pt = c_pt
                                break
                        curr_d += 1

                    # 如果没有找到下一个边缘点
                    if count == 7:
                        curr_d = 0
                        tra_flag = True
                        if len(edge_t) > iminlength:
                            arc.append(edge_t)

    return arc


#可以找到所有的点，但是不保证顺序
def detect_edges_all(src,iminlength = 10):
    directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    edge = src.copy()
    h, w = edge.shape
    arc = []

    for i in range(w):
        for j in range(h):
            if edge[j, i] == 255:  # 如果当前点为轮廓点
                contour = []
                seeds = [(i, j)]
                edge[j, i] = 0  # 访问过的点设为 0

                while seeds:
                    x, y = seeds.pop(0)
                    contour.append((x, y))

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < w and 0 <= ny < h and edge[ny, nx] == 255:
                            edge[ny, nx] = 0
                            seeds.append((nx, ny))
                if(len(contour)>iminlength):
                    arc.append(contour)

    return arc


if __name__ =="__main__":
    filename = r"E:\dataset\Prasad Images - Dataset Prasad\images\016_0059.jpg"
    img = cv.imread(filename)
    #创建白色画布
    canvas = 255*np.ones((img.shape[0],img.shape[1],3),dtype=np.uint8)
    #转成灰度图像
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #中值滤波
    medianBlur_ksize = 3
    ray = cv.medianBlur(gray,medianBlur_ksize)
    #自适应canny边缘检测
    canny = canny_detect.canny_detect(gray)
    #连接边缘点
    arcs = detect_edges(canny)
    print(f"找到的边的数量：{len(arcs)}")
    #测试：
    for segement in arcs:
    # 生成随机 BGR 颜色
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #print(random_color)
        #print(segement)
        for p in segement:
            x = p[0]
            y = p[1]
            canvas[y,x] = random_color
    

    cv.imshow("edges",canvas)
    cv.imwrite("edges.jpg",canvas)
    cv.waitKey(0)
    cv.destroyAllWindows