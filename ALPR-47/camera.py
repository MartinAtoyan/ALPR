import cv2
import os

camera = cv2.VideoCapture(0)

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
fps = 20

output = cv2.VideoWriter('output.avi',
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                         fps, frame_size)

while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        output.write(frame)
        cv2.imshow('Recording', frame)
        k = cv2.waitKey(1)
        if k == 113:
            break
    else:
        print("Camera disconnected")
        break

camera.release()
output.release()
cv2.destroyAllWindows()
