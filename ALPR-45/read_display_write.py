import cv2
import numpy

image = cv2.imread("mountain.jpg", cv2.IMREAD_COLOR)
gray_image = cv2.imread("mountain.jpg", cv2.IMREAD_GRAYSCALE)
image_unchanged = cv2.imread("mountain.jpg", cv2.IMREAD_UNCHANGED)

cv2.imshow("Original Image", image)
cv2.imshow("Gray Scale Image", gray_image)
cv2.imshow("Unchanged Image", image_unchanged)

cv2.imwrite("mountain.png", image)
cv2.imwrite("gray_mountain.png", gray_image)
cv2.imwrite("mountain_unchanged.png", image_unchanged)

written_image = cv2.imread("mountain.png")
written_gray_image = cv2.imread("gray_mountain.png")
written_unchanged_image = cv2.imread("mountain_unchanged.png")

cv2.imshow("Written Image", written_image)
cv2.imshow("Written gray image", written_gray_image)
cv2.imshow("Written unchanged image", written_unchanged_image)

cv2.waitKey(0)
cv2.destroyAllWindows()