from math import radians

import cv2
import numpy

original_image = cv2.imread("mountain.jpg")
white_image = cv2.imread("white.jpg")
image_copy = original_image.copy()

height, width = original_image.shape[:2]
PointA_x = (width // 5) * 2
PointA_y = (height // 4) * 2

PointB_x = (width // 5) * 4
PointB_y = (height // 4) * 3

PointA = (PointA_x, PointA_y)
PointB = (PointB_x, PointB_y)

circle_center_1 = (600, 1800)
circle_center_2 = (4900, 1800)
radius = 300

rect_start = (PointA_x - 100, PointA_y - 900)
rect_end = (PointB_x - 100, PointB_y - 900)

center_ellipse = (800, 1800)
ellipse_radius = 200
axis = (100, 50)

text = "Testing drawing skills in OpenCV"
org = (2900, 800)

cv2.line(image_copy, PointA, PointB, (0, 255, 0), 3, lineType=cv2.LINE_AA)
cv2.circle(image_copy, circle_center_1, radius, (255, 255, 0), thickness=4, lineType=cv2.LINE_AA)
cv2.circle(image_copy, circle_center_2, radius, (255, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
cv2.rectangle(image_copy, rect_start, rect_end, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
cv2.ellipse(image_copy, center_ellipse, axis, 1, 0, 360, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
cv2.putText(image_copy, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 0, 0),
            lineType=cv2.LINE_AA, thickness=3)
concatenated = numpy.concatenate((original_image, image_copy), axis=1)

cv2.imshow("Original and with line images", concatenated)
cv2.waitKey()
cv2.destroyAllWindows()
