import os
import cv2
import numpy as np

folder_path = "/Users/picsartacademy/Desktop/croppedimgs"

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Laplacian(gray, ksize=3, ddepth=cv2.CV_16S)
    filtered_image = cv2.convertScaleAbs(edges)
    cv2.imshow(filename, filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

