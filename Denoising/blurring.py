import os
import cv2
import numpy as np

folder_path = "/Users/picsartacademy/Desktop/croppedimgs"

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    median = cv2.medianBlur(gray, 3)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    concatenated_blur = np.concatenate((gaussian, median, bilateral), axis=1)

    cv2.imshow('Gaussian | Median | Bilateral', concatenated_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# best result is Median