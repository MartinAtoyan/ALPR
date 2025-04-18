import cv2
import numpy

original_image = cv2.imread("tiger.jpg")

scale_up_x = 2
scale_up_y = 1.5

scale_down_x = 0.8
scale_down_y = 0.8

resized_down_image = cv2.resize(original_image, None, fx=scale_down_x, fy=scale_down_y, interpolation=cv2.INTER_LINEAR)
resized_up_image = cv2.resize(original_image, None, fx=scale_up_x, fy=scale_up_y, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Original image", original_image)
cv2.waitKey()
cv2.imshow("Resized down image", resized_down_image)
cv2.waitKey()
cv2.imshow("Resized up image", resized_up_image)
cv2.waitKey()

cv2.destroyAllWindows()