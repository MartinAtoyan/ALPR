import cv2
import numpy

original_image = cv2.imread("tiger.jpg")

scale_up = 1.5

resized_interpolation_liner = cv2.resize(
                                original_image, None, fx=scale_up,
                                fy=scale_up, interpolation=cv2.INTER_LINEAR)
resized_interpolation_area = cv2.resize(
                                original_image, None, fx=scale_up,
                                fy=scale_up, interpolation=cv2.INTER_AREA)
resized_interpolation_nearest = cv2.resize(
                                    original_image, None, fx=scale_up,
                                    fy=scale_up, interpolation=cv2.INTER_NEAREST)

vertical_con = numpy.concatenate(
            (resized_interpolation_nearest,
                    resized_interpolation_area,
                    resized_interpolation_liner), axis=1)

cv2.imshow("Concatenated image for differences", vertical_con)
cv2.waitKey()

cv2.destroyAllWindows()