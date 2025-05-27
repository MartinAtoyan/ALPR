import numpy
import cv2

image = cv2.imread('test.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Cleaned', cleaned)
cv2.waitKey(0)

cv2.imshow('Original', image)
cv2.waitKey(0)

cv2.destroyAllWindows()