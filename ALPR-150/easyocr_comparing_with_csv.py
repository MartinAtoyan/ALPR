import os
import csv
import time
import easyocr
from pathlib import Path
from algorithm import levenshtein_distance

reader = easyocr.Reader(['en'])

# csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
csv_file = "/home/martin/PycharmProjects/ALPR/ALPR-142/number_plate.csv"
# image_folder = Path("/Users/picsartacademy/Desktop/cropped_plates/images")
image_folder = Path("/home/martin/Desktop/cropped_plates/images")

image_folder_am = Path("/home/martin/Desktop/cropped_plates_1000/images/armenian")
image_folder_ge = Path("/home/martin/Desktop/cropped_plates_1000/images/georgian")
image_folder_eu = Path("/home/martin/Desktop/cropped_plates_1000/images/eu")
image_folder_ru = Path("/home/martin/Desktop/cropped_plates_1000/images/russian")


start_time = time.time()

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

results = []
undetected_images = []

for image_path in sorted(image_folder.glob("*.png")):
    image_name = image_path.name
    result = reader.readtext(str(image_path))

    if not result:
        undetected_images.append(image_name)
        continue

    detected_text = ""
    highest_prob = 0

    for bbox, text, prob in result:
        if prob > highest_prob:
            highest_prob = prob
            detected_text = text.replace("/", "").replace(",", "").replace(".", "").replace("-", "").replace(" ", "").replace("|", "").lower()

    if image_name in expected_plates:
        expected_plate = expected_plates[image_name]
        results.append({
            "image_name": image_name,
            "expected": expected_plate,
            "detected": detected_text,
            "match": expected_plate == detected_text,
            "confidence": highest_prob,
            "CER": 1 - (levenshtein_distance(expected_plate, detected_text) / len(expected_plate))
        })
    else:
        print(f"Warning: No expected value found for {image_name}")

with open("plate_results.txt", 'w', newline='') as fl:
    fl.write("Image Name | Expected Plate | Detected Plate | Match | Confidence | CER \n")
    fl.write("-" * 80 + "\n")
    CER_average = 0
    for item in results:
        match_str = "✓" if item["match"] else "✗"
        CER_average += item["CER"]
        fl.write(
            f"{item['image_name']} | {item['expected']} | {item['detected']} | {match_str} | {item['confidence']:.2f} | {item['CER']:.2f} \n")

    fl.write("\n\nSummary:\n")
    total = len(results)
    matches = sum(1 for item in results if item["match"])
    fl.write(f"Total images processed: {total}\n")
    fl.write(f"Successful matches: {matches} ({matches / total * 100:.2f}%)\n")
    fl.write(f"Failed matches: {total - matches} ({(total - matches) / total * 100:.2f}%)\n")
    fl.write(f"CER: {CER_average / total * 100:.2f}%\n")
    fl.write(f"{(time.time() - start_time):.2f} second")

    if undetected_images:
        fl.write(f"\nImages with no detected text ({len(undetected_images)}):\n")
        for img in undetected_images:
            fl.write(f"- {img}\n")

print(f"Processing complete. Results saved to plate_results.txt")