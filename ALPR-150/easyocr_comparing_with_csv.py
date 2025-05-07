import os
import csv
import easyocr
from pathlib import Path

reader = easyocr.Reader(['en', 'ru'])

csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
image_folder = Path("/Users/picsartacademy/Desktop/cropped_plates/images")

expected_plates = []
with open(csv_file, 'r', newline='') as csvfile:
    reader_csv = csv.reader(csvfile)
    for row in reader_csv:
            plate_value = row[0].split()
            expected_plates.append(plate_value[1].replace(" ", ""))

detected_plates = []
for image_path in sorted(image_folder.glob("*.png")):
    result = reader.readtext(str(image_path))
    for (_, text, prob) in result:
        if len(text.strip()) > 4:
            detected_plates.append({text.strip(): f'{prob:.2f}'})

for elem in detected_plates:
    plate = list(elem.keys())[0]
    plate = plate.replace(" ", "")
    if plate in expected_plates:
        print({plate:f"Plate is in expected plate with probability of {list(elem.values())[0]}"})
    else:
        print({plate:"Plate is not in expected plate"})

print(len(expected_plates))
print(len(detected_plates))

# Length of expected and detected lists are different, because easyocr read everything in number plate. I was tried to minimize the incorrect strings.
# But with armenian number plates it works very well in almost all cases. The minimum probability of recognition is a 0.45
