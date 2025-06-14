import math
import time
import cv2

def  maintain_aspect_ratio_resize(image,  width=None,  height=None,  inter=cv2.INTER_AREA):
    dim =  None
    (h, w)  = image.shape[:2]
    if width is  None  and height is  None:
        return image

    if width is  None:
        r = height /  float(h)
        dim =  (int(w * r), height)
    else:
        r = width /  float(w)
        dim =  (width,  int(h * r))

    return cv2.resize(image, dim,  interpolation=inter)

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


def clear_text(string: str):
    return (string.replace("/", "").replace(",", "").
            replace(".", "").replace("-", "").
            replace(" ", "").replace("|", "").replace("-", "").lower())


start_time = time.time()
expected_plates = {}


def image_name_in_expected_plates(image_name_func, detected_text, highest_prob, results_func):
        results_func.append({
            "image_name": image_name_func,
            "detected": detected_text,
            "confidence": highest_prob
            # "CER": (levenshtein_distance(expected_plate_func, detected_text) / max(1, len(expected_plate_func)))
        })

#
def levenshtein_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost)

    return dp[m][n]
