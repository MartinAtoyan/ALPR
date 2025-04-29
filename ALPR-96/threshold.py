import cv2
import numpy

image = cv2.imread("input_image.jpg", cv2.IMREAD_GRAYSCALE)

thresh = 127
maxValue = 255

# All values less than thresh are invisible
_, threshold = cv2.threshold(image, thresh, maxValue, cv2.THRESH_BINARY)
concatenated = numpy.concatenate((image, threshold), axis=1)
cv2.imshow("Original and threshold", concatenated)
cv2.waitKey(0)

# Inverse of first threshold
_, threshold_inv = cv2.threshold(image, thresh, maxValue, cv2.THRESH_BINARY_INV)
concatenated = numpy.concatenate((image, threshold_inv), axis=1)
cv2.imshow("Original and threshold", concatenated)
cv2.waitKey(0)

# All values greater than thresh are set to thresh value, all values less than thresh are unchanged.
_, threshold_trunc = cv2.threshold(image, thresh, maxValue, cv2.THRESH_TRUNC)
concatenated_trunc = numpy.concatenate((image, threshold_trunc), axis=1)
cv2.imshow("Original and threshold", concatenated_trunc)
cv2.waitKey(0)

# Visible numbers only greater thresh
_, threshold_zero = cv2.threshold(image, thresh, maxValue, cv2.THRESH_TOZERO)
concatenated_trunc = numpy.concatenate((image, threshold_zero), axis=1)
cv2.imshow("Original and threshold", concatenated_trunc)
cv2.waitKey(0)

# Visible only numbers which thresh border greater than thresh, (ketgic)
_, threshold_inv_zero = cv2.threshold(image, thresh, maxValue, cv2.THRESH_TOZERO_INV)
concatenated_trunc = numpy.concatenate((image, threshold_inv_zero), axis=1)
cv2.imshow("Original and threshold", concatenated_trunc)
cv2.waitKey()

cv2.destroyAllWindows()
