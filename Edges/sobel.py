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
    # No need to do blur process with sobel, it decrease image quality
    sobel_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    concatenated_sobel = np.concatenate((sobel_x, sobel_y), axis=1)
    cv2.imshow(filename, concatenated_sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



