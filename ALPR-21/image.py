import cv2
import numpy

image = cv2.imread("wood.jpg")

kernel_matrix = numpy.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]])

identify = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_matrix)

concatenated = numpy.concatenate((image, identify), axis=1)
cv2.imshow("Original and filter2D", concatenated)
cv2.waitKey()
cv2.imwrite("identify.jpg", identify)

kernel_2 = numpy.ones((5, 5), numpy.float32) / 25
conv_image = cv2.filter2D(image, -1, kernel_2)
concatenated_conv = numpy.concatenate((image, conv_image), axis=1)
cv2.imshow("Original and filter2D", concatenated_conv)
cv2.waitKey()
cv2.imwrite("blured_wood.jpg", conv_image)

wood_blur = cv2.blur(image, ksize=(5, 5))
concatenated_blur = numpy.concatenate((image, conv_image), axis=1)
cv2.imshow("Original and blur", concatenated_blur)
cv2.waitKey()
cv2.imwrite("blured_wood_2.jpg", wood_blur)

gaussian_blur = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=0, sigmaY=0)
concatenated_gaussian = numpy.concatenate((image, gaussian_blur), axis=1)
cv2.imshow("Original and Gaussian blur", concatenated_gaussian)
cv2.waitKey()
cv2.imwrite("gaussian_blur.jpg", gaussian_blur)

median_blur = cv2.medianBlur(image, ksize=5)
concatenated_median = numpy.concatenate((image, median_blur), axis=1)
cv2.imshow("Original and median blur", concatenated_median)
cv2.waitKey()
cv2.imwrite("median_blur.jpg", median_blur)

kernel_3 = numpy.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

sharpen = cv2.filter2D(image, -1, kernel_3)
concatenated_sharpen = numpy.concatenate((image, sharpen), axis=1)
cv2.imshow("original and sharpen", concatenated_sharpen)
cv2.waitKey()
cv2.imwrite("sharpen.jpg", sharpen)

bilateral_filter = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
concatenated_bilateral = numpy.concatenate((image, bilateral_filter), axis=1)
cv2.imshow("Original and bilateral", concatenated_bilateral)
cv2.waitKey()
cv2.imwrite("bilateral.jpg", bilateral_filter)

cv2.destroyAllWindows()