import os
import cv2
import numpy as np
import math

folder_path = "/Users/picsartacademy/Desktop/croppedimgs"

def get_rotated_dimensions(w, h, angle_deg):
    angle_rad = math.radians(abs(angle_deg))
    cos_angle = abs(math.cos(angle_rad))
    sin_angle = abs(math.sin(angle_rad))

    new_w = int(w * cos_angle + h * sin_angle)
    new_h = int(w * sin_angle + h * cos_angle)

    return new_w, new_h

def select_license_plate_line(lines_lst, image_shape, debug=False):
    if not lines_lst:
        return None

    h, w = image_shape[:2]
    image_center_y = h // 2
    image_center_x = w // 2

    line_scores = []

    for i, line in enumerate(lines_lst):
        x1, y1, x2, y2 = line

        length = math.hypot(x2 - x1, y2 - y1)
        y_diff = abs(y1 - y2)
        x_diff = abs(x2 - x1)

        angle = abs(math.degrees(math.atan2(y_diff, x_diff))) if x_diff > 0 else 90
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        score = 0

        if angle <= 5:
            score += 100
        elif angle <= 10:
            score += 80
        elif angle <= 15:
            score += 60
        elif angle <= 25:
            score += 30

        rel_length = length / w
        if rel_length >= 0.4:
            score += 50
        elif rel_length >= 0.25:
            score += 35
        elif rel_length >= 0.15:
            score += 20
        else:
            score += 5

        dist_from_center_y = abs(center_y - image_center_y) / h
        if dist_from_center_y >= 0.3:
            score += 30
        elif dist_from_center_y >= 0.1:
            score += 20
        else:
            score += 15

        hor_span = x_diff / w
        if hor_span >= 0.3:
            score += 25
        elif hor_span >= 0.2:
            score += 15
        elif hor_span >= 0.1:
            score += 10

        if angle > 45:
            score -= 50

        dist_from_center_x = abs(center_x - image_center_x) / w
        if dist_from_center_x <= 0.2:
            score += 15

        line_scores.append({
            'line': line,
            'score': score,
            'angle': angle,
            'length': length,
        })

    line_scores.sort(key=lambda x: x['score'], reverse=True)

    if debug:
        print("Top scoring lines:")
        for i, info in enumerate(line_scores[:5]):
            print(f"  Rank {i+1}: Score={info['score']} | Angle={info['angle']:.2f}° | Length={info['length']:.1f}")

    return line_scores[0]['line'] if line_scores else None


for filename in os.listdir(folder_path):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Could not load: {filename}")
        continue

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        print(f"No lines detected in {filename}")
        continue

    lines_lst = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines_lst.append([x1, y1, x2, y2])

    best_line = select_license_plate_line(lines_lst, image.shape, debug=True)

    if best_line:
        x1, y1, x2, y2 = best_line
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle = math.degrees(angle_rad)
    else:
        print(f"⚠️ No suitable horizontal line found in {filename}")
        angle = 0

    new_w, new_h = get_rotated_dimensions(w, h, angle)
    new_center = (new_w // 2, new_h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    M[0, 2] += new_center[0] - center[0]
    M[1, 2] += new_center[1] - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    max_display = 1200
    if max(new_w, new_h) > max_display:
        scale = max_display / max(new_w, new_h)
        display_size = (int(new_w * scale), int(new_h * scale))
        rotated_display = cv2.resize(rotated, display_size)
        original_display = cv2.resize(image, display_size)
    else:
        rotated_display = rotated
        original_display = cv2.resize(image, (new_w, new_h))

    concatenated = np.concatenate((rotated_display, original_display), axis=1)

    cv2.imshow("Original vs Rotated", concatenated)
    print(f"File: {filename}")
    print(f"Original size: {w}x{h}")
    print(f"New size:{new_w}x{new_h}")
    print(f"Rotation:{angle:.2f}°")

    cv2.waitKey(0)

cv2.destroyAllWindows()
