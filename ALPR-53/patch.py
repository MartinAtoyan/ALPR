import cv2
import numpy
# 1280x720
original_image = cv2.imread("lion.jpg")
image_copy = original_image.copy()

image_height, image_width, channels = original_image.shape
M = 119
N = 255

x1 = 0
y1 = 0

for y in range(0, image_height, M):
    for x in range(0, image_width, N):

        if image_height - y < M or image_width - x < N:
            break

        y1 = y + M
        x1 = x + N

        if x1 >= image_width and y1 >= image_height:
            y1 = image_height - 1
            x1 = image_width - 1
            tiles = image_copy[y:y+M, x:x+N]
            cv2.imwrite(f"{x}_{y}.jpg", tiles)
            cv2.rectangle(original_image, (x, y), (x1, y1), (0, 255, 0), 1)

        elif y1 > image_height:
            y1 = image_height - 1
            tiles = image_copy[y:y+M, x:x+N]
            cv2.imwrite(f"{x}_{y}.jpg", tiles)
            cv2.rectangle(original_image, (x, y), (x1, y1), (0, 255, 0), 1)

        elif x1 > image_width:
            x1 = image_width - 1
            tiles = image_copy[y:y+M, x:x+N]
            cv2.imwrite(f"{x}_{y}.jpg", tiles)
            cv2.rectangle(original_image, (x, y), (x1, y1), (0, 255, 0), 1)

        else:
            tiles = image_copy[y:y+M, x:x+N]
            cv2.imwrite(f"{x}_{y}.jpg", tiles)
            cv2.rectangle(original_image, (x,y), (x1, y1), (0, 255, 0), 1)

cv2.imshow("Patched image", original_image)
cv2.imwrite("patched.jpg", original_image)
cv2.waitKey()
cv2.destroyAllWindows()
