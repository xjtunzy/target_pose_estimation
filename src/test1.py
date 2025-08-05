import cv2

img = cv2.imread(r"C:\Users\34991\Desktop\0b088c9ab5be49088477ece9ec3f337d.jpg")

# 保存为不同压缩质量的 JPEG 文件
cv2.imwrite('output_90.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # 质量高
cv2.imwrite('output_50.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])  # 中等质量
cv2.imwrite('output_10.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 2])  # 很模糊
