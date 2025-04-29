import cv2
import numpy

image = cv2.imread("tiger.jpg", flags=0)
image_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)

sobel_x = cv2.Sobel(image_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobel_y = cv2.Sobel(image_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobel_xy = cv2.Sobel(image_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

concatenated = numpy.concatenate((sobel_x, sobel_y, sobel_xy), axis=1)
cv2.imshow("Sobel X, Sobel Y, Sobel XY", concatenated)
cv2.waitKey()
cv2.destroyAllWindows()