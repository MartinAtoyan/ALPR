import cv2
import numpy as np

image = cv2.imread("BlobTest.jpg", cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 200

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.87

params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

image_with_keypoints = cv2.drawKeypoints(image_bgr, keypoints, np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
