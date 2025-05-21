import cv2
import numpy

# image = cv2.imread("Screenshot from 2025-05-21 18-10-52.png")
image = cv2.imread("Screenshot from 2025-05-21 18-11-37.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.bitwise_not(gray_image)

thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = numpy.column_stack(numpy.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

print(angle)

# if angle < -45:
#     angle = -(90 + angle)
# elif angle >= 0:
#     angle = angle
# elif -45 <= angle < 0:
#     angle = -angle
#

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

print(angle)


(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

concatenated = numpy.concatenate((image, rotated), 1)
cv2.imshow("BOTH", concatenated)
cv2.waitKey(0)
cv2.destroyAllWindows()