import cv2
import csv
import pytesseract
from pathlib import Path

csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
image_folder = Path("/Users/picsartacademy/Desktop/cropped_plates/images")

expected_plates = []
with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        plate = row[0].split()
        expected_plates.append(plate[1].replace(" ", "").lower())

# detected_plates = []
# for image_path in sorted(image_folder.glob("*.png")):
#     image = cv2.imread(str(image_path))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     text = pytesseract.image_to_string(image)
#     if len(text) > 4:
#         detected_plates.append(text.replace(" ", "").lower())
#
# for detected_plate in detected_plates:
#     if detected_plate in expected_plates:
#         print(f"{detected_plate} is in expected plates")
#     else:
#         print(f"{detected_plate} is not in expected plates")

custom_config = (
    r'--oem 3 --psm 7 '
    r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    r'abcdefghijklmnopqrstuvwxyz'
    r'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    r'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    r'0123456789'
)

detected_plates = []

for image_path in sorted(image_folder.glob("*.png")):
    image = cv2.imread(str(image_path))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)  # Denoise
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    text = pytesseract.image_to_string(thresh, config=custom_config, lang='eng+rus')
    cleaned_text = text.replace(" ", "").replace("\n", "").strip().lower()

    if len(cleaned_text) > 4:
        detected_plates.append(cleaned_text)

for detected_plate in detected_plates:
    if detected_plate in expected_plates:
        print(f"{detected_plate} is in expected plates")
    else:
        print(f"{detected_plate} is NOT in expected plates")

print(len(detected_plates)) # 347
print(len(expected_plates)) # 499

# Tesseract doesnt work well with armenian number plates, but good in georgian.