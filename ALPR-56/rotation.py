import cv2
import numpy

original_image = cv2.imread("images.jpeg")
image_height, image_width = original_image.shape[:2]

center = (image_width / 2, image_height / 2)

rotate_matrix = cv2.getRotationMatrix2D(center, angle=-45, scale=1)
rotated_image = cv2.warpAffine(
                    original_image, M=rotate_matrix,
                    dsize=(image_width, image_height))

concatenated = numpy.concatenate((original_image, rotated_image), axis=1)
cv2.imshow("Original and Rotated", concatenated)
cv2.waitKey()
