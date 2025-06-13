import os
import cv2
import csv
import math
import time
import easyocr
import numpy as np
from pathlib import Path

from img_proc_final.extra_func import get_rotated_dimensions, select_license_plate_line, clear_text, \
    levenshtein_distance

def crop_image_x_axis(image, crop_percentage):
    h, w = image.shape[:2]

    crop_pixels = int(w * crop_percentage)

    if crop_pixels * 2 >= w:
        return image

    cropped = image[:, crop_pixels:w - crop_pixels]

    return cropped

def image_name_in_expected_plates(image_name, detected_text, expected_plates, results):
    if image_name in expected_plates:
        expected_plate = expected_plates[image_name]
        cer = levenshtein_distance(expected_plate, detected_text) / max(1, len(expected_plate))
        results.append({
            "image_name": image_name,
            "expected": expected_plate,
            "detected": detected_text,
            "match": expected_plate == detected_text,
            "CER": cer
        })


def enhance_image_for_ocr(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    kernel = np.ones((2, 2), np.uint8)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return enhanced


def process_license_plate_image(image_path, reader):
    filename = Path(image_path).name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Could not load: {filename}")
        return filename, None, ""

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_median = cv2.medianBlur(image_gray, 3)

    edges = cv2.Canny(image_median, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    angle = 0
    if lines is not None:
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

        best_line = select_license_plate_line(lines_lst, image.shape, debug=False)

        if best_line:
            x1, y1, x2, y2 = best_line
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle = math.degrees(angle_rad)

    if angle != 0:
        new_w, new_h = get_rotated_dimensions(w, h, angle)
        new_center = (new_w // 2, new_h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        M[0, 2] += new_center[0] - center[0]
        M[1, 2] += new_center[1] - center[1]

        rotated = cv2.warpAffine(image, M, (new_w, new_h))
    else:
        rotated = image.copy()
        new_w, new_h = w, h

    max_display = 1200
    if max(new_w, new_h) > max_display:
        scale = max_display / max(new_w, new_h)
        display_size = (int(new_w * scale), int(new_h * scale))
        rotated_display = cv2.resize(rotated, display_size)
    else:
        rotated_display = rotated.copy()

    rotated_gray = cv2.cvtColor(rotated_display, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image_for_ocr(rotated_gray)
    cropped = crop_image_x_axis(enhanced, 0.07)

    upscaled = cv2.resize(cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    result = reader.readtext(upscaled, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if not result:
        return filename, upscaled, ""

    plate = ""
    for bbox, text, prob in result:
        if prob > 0.5:
            plate += text

    cleaned_text = clear_text(plate)
    return filename, upscaled, cleaned_text


def main():
    folder_path = "/Users/picsartacademy/Downloads/cropped_plates_1000/images/eu/images"
    csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-150/number_plate_1000.csv"

    expected_plates = {}
    with open(csv_file, 'r', newline='') as csvfile:
        reader_csv = csv.reader(csvfile)
        for row in reader_csv:
            if row:
                parts = row[0].split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    plate_value = parts[1].replace(" ", "").lower()
                    expected_plates[image_name] = plate_value

    reader = easyocr.Reader(['en'])
    results = []
    undetected_images = []
    folder = Path(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    start_time = time.time()
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    for image_path in image_files:
        filename, processed, raw_concatenated_text = process_license_plate_image(image_path, reader)

        if not raw_concatenated_text:
            undetected_images.append(filename)
            continue

        image_name_in_expected_plates(filename, raw_concatenated_text, expected_plates, results)

    output_file = "final_1.txt"
    with open(output_file, 'w', newline='') as f:
        f.write("Final results\n")
        f.write("-" * 60 + "\n")
        f.write("Filename | Expected Plate | Detected Plate | Match | CER\n")
        f.write("-" * 60 + "\n")

        for item in results:
            match_str = "✓" if item["match"] else "✗"
            f.write(
                f"{item['image_name']} | {item['expected']} | {item['detected']} | {match_str} | {item['CER']:.3f}\n")

        f.write(f"\nSUMMARY:\n")
        f.write("=" * 30 + "\n")

        total_processed = len(results)
        total_matches = sum(1 for item in results if item["match"])

        f.write(f"Total images processed: {total_processed}\n")

        if total_processed > 0:
            accuracy = (total_matches / total_processed) * 100
            avg_cer = sum(item["CER"] for item in results) / total_processed

            f.write(f"Successful matches: {total_matches} ({accuracy:.2f}%)\n")
            f.write(f"Average CER: {avg_cer:.3f} ({avg_cer * 100:.2f}%)\n")
            f.write(
                f"Failed matches: {total_processed - total_matches} ({((total_processed - total_matches) / total_processed) * 100:.2f}%)\n")

            if undetected_images:
                f.write(f"Undetected images: {len(undetected_images)}\n")

        f.write(f"\nProcessing time: {(time.time() - start_time):.2f} seconds\n")

        if undetected_images:
            f.write(f"\nImages with no detected text ({len(undetected_images)}):\n")
            for img in undetected_images:
                f.write(f"  - {img}\n")

    return results, undetected_images


if __name__ == "__main__":
    results, undetected_images = main()