import os
import cv2
import numpy as np

folder_path = "/Users/picsartacademy/Desktop/croppedimgs"

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blured = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blured, 50, 150)
    cv2.imshow(filename, edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# best result for license plates