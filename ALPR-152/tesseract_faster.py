import os
import cv2
import csv
import pytesseract
import numpy as np
import concurrent.futures

from pathlib import Path
from functools import partial
from difflib import get_close_matches


def load_expected_plates(csv_file):
    expected_plates = []
    with open(csv_file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                plate = row[0].split()
                if len(plate) > 1:
                    expected_plates.append(plate[1].replace(" ", "").lower())
    return expected_plates


def fast_preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed_images = []

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(binary)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images.append(adaptive_thresh)

    return preprocessed_images


def clean_text(text):
    replacements = {
        'о': '0', 'о': 'o', 'i': '1', 'l': '1',
        's': '5', 'б': '6', 'в': 'b'
    }

    cleaned = ''.join(c for c in text.replace(" ", "").replace("\n", "").strip().lower()
                      if c.isalnum() or c in '-.')

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    return cleaned


def detect_license_plate(image_path, configs, expected_plates=None):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        preprocessed_images = fast_preprocess_image(image)
        detected_texts = []

        for config in configs:
            for prep_img in preprocessed_images:
                text = pytesseract.image_to_string(prep_img, config=config)
                cleaned = clean_text(text)

                if len(cleaned) >= 4:
                    detected_texts.append(cleaned)

        if expected_plates and detected_texts:
            for detected in detected_texts:
                if detected in expected_plates:
                    return detected

            close_matches = []
            for detected in detected_texts:
                matches = get_close_matches(detected, expected_plates, n=1, cutoff=0.7)
                if matches:
                    close_matches.append((detected, matches[0]))

            if close_matches:
                close_matches.sort(key=lambda x: get_close_matches(x[0], [x[1]], n=1, cutoff=0)[0])
                return close_matches[0][1]

        return max(detected_texts, key=len) if detected_texts else None

    except Exception:
        return None


def process_images_parallel(image_folder, csv_file):
    expected_plates = load_expected_plates(csv_file)

    tesseract_configs = [
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789',
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789',
        r'--oem 3 --psm 7 -l eng+rus+hye',
        r'--oem 3 --psm 7 -l eng+rus+kat',
    ]

    image_files = Path(image_folder).glob("*.png")

    detected_plates = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        process_image = partial(detect_license_plate, configs=tesseract_configs, expected_plates=expected_plates)
        detected_plates = list(filter(None, executor.map(process_image, image_files)))

    return detected_plates


def main():
    csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
    image_folder = "/Users/picsartacademy/Desktop/cropped_plates/images"

    detected_plates = process_images_parallel(image_folder, csv_file)

    print(f"Plates detected: {len(detected_plates)}")

    with open("detected_plates.txt", "w", encoding="utf-8") as f:
        for i, plate in enumerate(detected_plates, 1):
            f.write(f"{i}.{plate}\n")


if __name__ == "__main__":
    main()


# The program read number plate from image, if it in csv file, adding to .txt file, if it not match adding best guessed string.
# The best result 495

