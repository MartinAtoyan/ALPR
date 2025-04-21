import cv2
import numpy

cube_light = cv2.imread("cubes_light.png")
cube_dark = cv2.imread("cubes_dark.png")

lightLAB = cv2.cvtColor(cube_light, cv2.COLOR_BGR2LAB)
darkLAB = cv2.cvtColor(cube_dark, cv2.COLOR_BGR2LAB)

# cv2.imshow("LAB", lightLAB)
# cv2.waitKey(0)
# cv2.imshow("LAB", darkLAB)
# cv2.waitKey(0)
#
YCrCb_light = cv2.cvtColor(cube_light, cv2.COLOR_BGR2YCrCb)
YCrCb_dark = cv2.cvtColor(cube_dark, cv2.COLOR_BGR2YCrCb)
# cv2.imshow("LAB", YCrCb_light)
# cv2.waitKey(0)
# cv2.imshow("LAB", YCrCb_dark)
# cv2.waitKey(0)

lightHSV = cv2.cvtColor(cube_light, cv2.COLOR_BGR2HSV)
darkHSV = cv2.cvtColor(cube_dark, cv2.COLOR_BGR2HSV)
# cv2.imshow("LAB", lightHSV)
# cv2.waitKey(0)
# cv2.imshow("LAB", darkHSV)
# cv2.waitKey(0)

bgr = [40, 158, 6]
thresh = 124

minBGR = numpy.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = numpy.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
maskBGR = cv2.inRange(cube_light, minBGR, maxBGR)
res_bgr = cv2.bitwise_and(cube_light, cube_light, mask=maskBGR)

hsv_color = cv2.cvtColor(numpy.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
minHSV = numpy.array([hsv_color[0] - thresh, hsv_color[1] - thresh, hsv_color[2] - thresh])
maxHSV = numpy.array([hsv_color[0] + thresh, hsv_color[1] + thresh, hsv_color[2] + thresh])
maskHSV = cv2.inRange(lightHSV, minHSV, maxHSV)
res_hsv = cv2.bitwise_and(cube_light, cube_light, mask=maskHSV)

ycb_color = cv2.cvtColor(numpy.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]
minYCB = numpy.array([ycb_color[0] - thresh, ycb_color[1] - thresh, ycb_color[2] - thresh])
maxYCB = numpy.array([ycb_color[0] + thresh, ycb_color[1] + thresh, ycb_color[2] + thresh])
maskYCB = cv2.inRange(YCrCb_light, minYCB, maxYCB)
res_ycb = cv2.bitwise_and(cube_light, cube_light, mask=maskYCB)

lab_color = cv2.cvtColor(numpy.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
minLAB = numpy.array([lab_color[0] - thresh, lab_color[1] - thresh, lab_color[2] - thresh])
maxLAB = numpy.array([lab_color[0] + thresh, lab_color[1] + thresh, lab_color[2] + thresh])
maskLAB = cv2.inRange(lightLAB, minLAB, maxLAB)
res_lab = cv2.bitwise_and(cube_light, cube_light, mask=maskLAB)

cv2.imshow("BGR", res_bgr)
cv2.waitKey()
cv2.imshow("HSV", res_hsv)
cv2.waitKey()
cv2.imshow("YCB", res_ycb)
cv2.waitKey()
cv2.imshow("LAB", res_lab)
cv2.waitKey()

cv2.destroyAllWindows()