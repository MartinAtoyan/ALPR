import cv2
import numpy

original_image = cv2.imread("lion.jpg")

cropped_image = original_image[150:500, 300:1000]
cv2.imshow("cropped", cropped_image)
cv2.imwrite("cropped_image.jpg", cropped_image)
cv2.waitKey()

cv2.destroyAllWindows()