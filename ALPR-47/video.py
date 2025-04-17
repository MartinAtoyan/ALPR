import cv2
import numpy

video = cv2.VideoCapture("video.mov")

if video.isOpened():
    fps = int(video.get(5))
    frame_count = int(video.get(7))
    print(f"FPS: {fps}, Frame count: {frame_count}")
else:
    print("Error with opening video")

while video.isOpened():
    ret, frame = video.read()
    if ret:
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(20)
        if k == 113:
            break
    else:
        break

video.release()
cv2.destroyAllWindows()