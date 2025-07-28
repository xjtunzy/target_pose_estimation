import numpy as np
import cv2

def find_contour_swapped(edge, x, y, len_threshold, edge_contours):
    H, W = edge.shape

    # 方向定义仍然是 x<->y 互换的
    clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    anti_wise  = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]

    one_contour = [(x, y)]
    edge[y, x] = 0  

    move_x, move_y = x, y
    while True:
        is_end = True
        for dx, dy in clock_wise:
            nx, ny = move_x + dx, move_y + dy
            if 0 <= ny < H and 0 <= nx < W and edge[ny, nx]:
                edge[ny, nx] = 0
                one_contour.append((nx, ny))
                move_x, move_y = nx, ny
                is_end = False
                break
        if is_end:
            break

    # 反向追踪
    one_contour_opp = []
    move_x, move_y = x, y
    while True:
        is_end = True
        for dx, dy in anti_wise:
            nx, ny = move_x + dx, move_y + dy
            if 0 <= ny < H and 0 <= nx < W and edge[ny, nx]:
                edge[ny, nx] = 0
                one_contour_opp.append((nx, ny))
                move_x, move_y = nx, ny
                is_end = False
                break
        if is_end:
            break

    if len(one_contour) + len(one_contour_opp) > len_threshold:
        if one_contour_opp:
            one_contour_opp.reverse()
            full_contour = one_contour_opp + one_contour
            edge_contours.append(full_contour)
        else:
            edge_contours.append(one_contour)

def find_contours(edge, len_threshold):
    """
    edge: 二值图像 (numpy array of shape (H, W), dtype=uint8)
    return: list of list of (x, y) tuples, 注意 x/y 是调换后的坐标
    """
    edge_map = edge.copy()
    H, W = edge_map.shape
    edge_contours = []

    edge_map[0, :] = 0
    edge_map[-1, :] = 0
    edge_map[:, 0] = 0
    edge_map[:, -1] = 0

    for j in range(1, W):
        for i in range(1, H):
            if edge_map[i, j]:
                edge_map[i, j] = 0
                if edge_map[i + 1, j - 1] and edge_map[i + 1, j] and edge_map[i + 1, j + 1]:
                    continue
                find_contour_swapped(edge_map, j, i, len_threshold, edge_contours)  # 注意 j,i → x,y

    return edge_contours
