import cv2
import numpy

original_image = cv2.imread("tiger.jpg")

resize_down_width = 360
resize_down_height = 202
resize_down = (resize_down_width, resize_down_height)
down_image = cv2.resize(original_image, resize_down, interpolation=cv2.INTER_LINEAR)

resize_up_width = 1080
resize_up_height = 608
resize_up = (resize_up_width, resize_up_height)
up_image = cv2.resize(original_image, resize_up, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Original image", original_image)
cv2.waitKey()
cv2.imshow("Resized down image", down_image)
cv2.waitKey()
cv2.imshow("Resized up image", up_image)
cv2.waitKey()

cv2.destroyAllWindows()

