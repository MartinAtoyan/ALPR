import cv2
import numpy
import time


def start_recording():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")

    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    center = (int(frame_width / 2), int(frame_height / 2))

    output = cv2.VideoWriter('edited_video.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             20, frame_size)

    is_flipped_horizontal = False
    is_flipped_vertical = False
    flip_horizontal_type = 0
    flip_type = 1
    is_rotated = 0
    is_grayscale = False
    is_HSV = False
    is_LAB = False
    is_blur = False
    is_filtered = False
    is_drawn = False
    timer = True

    start_time = time.time()

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error reading from camera.")
            break

        if is_flipped_horizontal:
            frame = cv2.flip(frame, flip_type)

        if is_flipped_vertical:
            frame = cv2.flip(frame, flip_horizontal_type)

        if is_rotated != 0:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle=is_rotated, scale=1)
            frame = cv2.warpAffine(frame, rotation_matrix, frame_size)

        if is_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_HSV:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if is_LAB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        if is_blur:
            frame = cv2.GaussianBlur(frame, (11, 11), 0)  # (11, 11) to see difference between original and blured

        if is_filtered:
            frame = cv2.medianBlur(frame, 7)

        if is_drawn:
            frame = cv2.rectangle(frame, (20, 20), (1260, 700), (0, 0, 255), 3)  # 1280 x 720 pixels

        if timer:
            duration = time.time() - start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            text = f"{minutes:02d}:{seconds:02d}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            cv2.rectangle(frame, (25, 25), (35 + text_size[0], 65), (0, 0, 0), -1)
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        face = classifier.detectMultiScale(gray)
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA)

        output.write(frame)
        cv2.imshow("Recording", frame)

        key = cv2.waitKey(5) & 0xFF

        if key == ord("h"):
            is_flipped_horizontal = not is_flipped_horizontal

        elif key == ord("v"):
            is_flipped_vertical = not is_flipped_vertical

        elif key == ord("r"):
            is_rotated = is_rotated - 90

        elif key == ord("g"):
            is_grayscale = not is_grayscale

        elif key == ord("s"):
            is_HSV = not is_HSV

        elif key == ord("l"):
            is_LAB = not is_LAB

        elif key == ord("b"):
            is_blur = not is_blur

        elif key == ord("n"):
            is_filtered = not is_filtered

        elif key == ord("d"):
            is_drawn = not is_drawn

        elif key == ord("t"):
            timer = not timer

        elif key == ord('q'):
            break

    camera.release()
    output.release()

if __name__ == "__main__":
    start_recording()
    cv2.destroyAllWindows()