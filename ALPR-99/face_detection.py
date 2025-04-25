import cv2
import numpy

def show_detection(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=5)
    return img

image = cv2.imread("face.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

classifier_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
classifier_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces_alt2 = classifier_alt2.detectMultiScale(gray)
faces_default = classifier_default.detectMultiScale(gray)

_, faces_haar_alt2 = cv2.face.getFacesHAAR(image, "haarcascade_frontalface_alt2.xml")
# faces_haar_alt2 = numpy.squeeze(faces_haar_alt2) /book example doesn't work
# it's a GPT version
faces_haar_alt2 = faces_haar_alt2.reshape(-1, 4)

_, faces_haar_default = cv2.face.getFacesHAAR(image, "haarcascade_frontalface_default.xml")
# faces_haar_default = numpy.squeeze(faces_haar_default)
# This also
faces_haar_default = faces_haar_default.reshape(-1, 4)

image_haar_alt2 = show_detection(image.copy(), faces_haar_alt2)
image_haar_default = show_detection(image.copy(), faces_haar_default)

image_faces_alt2 = show_detection(image.copy(), faces_alt2)
image_faces_default = show_detection(image.copy(), faces_default)

cv2.imshow("Face detected", image_faces_alt2)
cv2.waitKey()
cv2.imshow("Face detected", image_faces_default)
cv2.waitKey()

cv2.imshow("Face detected", image_haar_alt2)
cv2.waitKey()
cv2.imshow("Face detected", image_haar_default)
cv2.waitKey()

cv2.destroyAllWindows()
