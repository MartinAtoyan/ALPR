import cv2
import numpy

video = cv2.VideoCapture("video.mov")

if video.isOpened():
    fps = int(video.get(5))
    frame_count = int(video.get(7))
    print(f"FPS: {fps}, Frame count: {frame_count}")
else:
    print("Error with opening video")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter("output_with_text.avi", fourcc, fps, (width, height))

while video.isOpened():
    ret, frame = video.read()
    if ret:
        cv2.imshow("Frame", frame)

        cv2.putText(frame, "Lorem Ispum", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("written video", frame)
        out.write(frame)

        k = cv2.waitKey(20)
        if k == 113:
            break
    else:
        break

video.release()
cv2.destroyAllWindows()
