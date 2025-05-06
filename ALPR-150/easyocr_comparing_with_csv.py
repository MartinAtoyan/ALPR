import os
import csv
import easyocr
from pathlib import Path

reader = easyocr.Reader(['en'])

csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
image_folder = Path("/Users/picsartacademy/Desktop/cropped_plates/images")

expected_plates = []
with open(csv_file, 'r', newline='') as csvfile:
    reader_csv = csv.reader(csvfile)

    for row in reader_csv:
            plate_value = row[0].split()
            expected_plates.append(plate_value[1])

detected_plates = []
for image_path in sorted(image_folder.glob("*.png")):
    result = reader.readtext(str(image_path))
    for (_, text, prob) in result:
        detected_plates.append(text.strip())

# match = detected_plates == expected_plates

# print("Match:", match)

# print(len(detected_plates))
# print(len(expected_plates))

print(detected_plates)