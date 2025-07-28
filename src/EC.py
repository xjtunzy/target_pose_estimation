import numpy as np
import cv2

def compute_threshold(dx, dy):
    mag_grad = np.abs(dx.astype(np.float32)) + np.abs(dy.astype(np.float32))
    max_grad = np.max(mag_grad)

    NUM_BINS = 80
    percent_of_pixels_not_edges = 0.93
    threshold_ratio = 0.3

    bin_size = int(np.floor(max_grad / NUM_BINS + 0.5)) + 1
    bin_size = max(bin_size, 1)

    bins = np.zeros(NUM_BINS, dtype=int)
    for val in mag_grad.flatten():
        bin_index = int(val) // bin_size
        if bin_index < NUM_BINS:
            bins[bin_index] += 1

    total = 0
    target = mag_grad.size * percent_of_pixels_not_edges
    threshold_high = 0
    while total < target and threshold_high < NUM_BINS:
        total += bins[threshold_high]
        threshold_high += 1
    threshold_high *= bin_size

    total = 0
    target *= threshold_ratio
    threshold_low = 0
    while total < target and threshold_low < NUM_BINS:
        total += bins[threshold_low]
        threshold_low += 1
    threshold_low *= bin_size

    return threshold_low, threshold_high

def mark_edge_canny(image):
    dx = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_16S, 0, 1)

    threshold_low, threshold_high = compute_threshold(dx, dy)
    edge = cv2.Canny(dx, dy, threshold_low, threshold_high)

    edge = edge.astype(np.uint8)
    direction = np.zeros((*image.shape, 2), dtype=np.float32)

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            if edge[i, j] != 0:
                gx, gy = dx[i, j], dy[i, j]
                length = np.hypot(gx, gy)
                if length != 0:
                    direction[i, j] = [gx / length, gy / length]
                edge[i, j] = 255
            else:
                edge[i, j] = 0

    return edge, direction

def canny_with_vector(image, min_length=10):
    edge_map, direction = mark_edge_canny(image)
    edge_vector = []  # placeholder, your actual edge contour collection logic goes here
    # Collecting(edge_map, edge_vector, min_length)  # needs your implementation
    return direction, edge_map, edge_vector