import cv2
import numpy

original_image = cv2.imread("images.jpeg")
image_height, image_width = original_image.shape[:2]

tx = image_width / 4
ty = image_height / 4

matrix = numpy.array([[1, 0, tx],
                      [0, 1, ty]],
                     dtype=numpy.float32)

translated_image = cv2.warpAffine(original_image, M=matrix, dsize=(image_width, image_height))
concatenated = numpy.concatenate((original_image, translated_image), axis=1)
cv2.imshow("Original and translated", concatenated)
cv2.waitKey()
cv2.destroyAllWindows()
