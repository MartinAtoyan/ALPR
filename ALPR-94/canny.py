import cv2
import numpy

image = cv2.imread("tiger.jpg")
image_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)

edges = cv2.Canny(image_blur, threshold1=100, threshold2=200)
cv2.imshow("Canny", edges)
cv2.waitKey()
cv2.destroyAllWindows()