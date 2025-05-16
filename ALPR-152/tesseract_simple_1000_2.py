import os
import csv
import time
import pytesseract
from PIL import Image
from pathlib import Path
from algorithm import levenshtein_distance

# csv_file = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
csv_file = "/home/martin/PycharmProjects/ALPR/ALPR-150/number_plate_1000.csv"

image_folder_am = Path("/home/martin/Desktop/cropped_plates_1000/images/armenian")
image_folder_ge = Path("/home/martin/Desktop/cropped_plates_1000/images/georgian")
image_folder_eu = Path("/home/martin/Desktop/cropped_plates_1000/images/eu/images")
image_folder_ru = Path("/home/martin/Desktop/cropped_plates_1000/images/russian")

def clear_text(string: str):
    return (string.replace("/", "").replace(",", "").
            replace(".", "").replace("-", "").
            replace(" ", "").replace("|", "").replace("\n", "").lower().strip("\f"))


start_time = time.time()
expected_plates = {}

def image_name_in_expected_plates(image_name_func, detected_text, expect_plates, results_func):
    if image_name_func in expect_plates:
        expected_plate_func = expect_plates[image_name_func]
        results_func.append({
            "image_name": image_name_func,
            "expected": expected_plate_func,
            "detected": detected_text,
            "match": expected_plate_func == detected_text,
            "CER": 1 - (levenshtein_distance(expected_plate_func, detected_text) / max(1, len(expected_plate_func)))
        })

with open(csv_file, 'r', newline='') as csvfile:
    reader_csv = csv.reader(csvfile)
    for row in reader_csv:
        if row:
            parts = row[0].split()
            if len(parts) >= 2:
                image_name = parts[0]
                plate_value = parts[1].replace(" ", "").lower()
                expected_plates[image_name] = plate_value

results_am = []
results_ge = []
results_eu = []
results_ru = []

undetected_images_am = []
undetected_images_ge = []
undetected_images_eu = []
undetected_images_ru = []

for image_path_am in sorted(image_folder_am.glob("*.png")):
    image_name_am = image_path_am.name
    image_am = Image.open(image_path_am)
    result_am = pytesseract.image_to_string(image_am)

    detected_text_am = clear_text(result_am)

    if not detected_text_am:
        undetected_images_am.append(image_name_am)
        continue

    image_name_in_expected_plates(image_name_am, detected_text_am, expected_plates, results_am)

for image_path_ge in sorted(image_folder_ge.glob("*.png")):
    image_name_ge = image_path_ge.name
    image_ge = Image.open(image_path_ge)
    result_ge = pytesseract.image_to_string(image_ge)

    detected_text_ge = clear_text(result_ge)

    if not detected_text_ge:
        undetected_images_ge.append(image_name_ge)
        continue

    image_name_in_expected_plates(image_name_ge, detected_text_ge, expected_plates, results_ge)

for image_path_eu in sorted(image_folder_eu.glob("*.png")):
    image_name_eu = image_path_eu.name
    image_eu = Image.open(image_path_eu)
    result_eu = pytesseract.image_to_string(image_eu)

    detected_text_eu = clear_text(result_eu)

    if not detected_text_eu:
        undetected_images_eu.append(image_name_eu)
        continue

    image_name_in_expected_plates(image_name_eu, detected_text_eu, expected_plates, results_eu)

for image_path_ru in sorted(image_folder_ru.glob("*.png")):
    image_name_ru = image_path_ru.name
    image_ru = Image.open(image_path_ru)
    result_ru = pytesseract.image_to_string(image_ru)

    detected_text_ru = clear_text(result_ru)

    if not detected_text_ru:
        undetected_images_ru.append(image_name_ru)
        continue

    image_name_in_expected_plates(image_name_ru, detected_text_ru, expected_plates, results_ru)


with open("plate_results_tesseract_1000.txt", 'w', newline='') as fl:
    fl.write("Image Name | Expected Plate | Detected Plate | Match | CER \n")
    fl.write("-" * 80 + "\n")

    for item in results_am:
        match_str = "✓" if item["match"] else "✗"
        fl.write(
            f"{item['image_name']} | {item['expected']} | {item['detected']} | {match_str} | {item['CER']:.2f} \n")

    for item in results_ge:
        match_str = "✓" if item["match"] else "✗"
        fl.write(
            f"{item['image_name']} | {item['expected']} | {item['detected']} | {match_str} | {item['CER']:.2f} \n")

    for item in results_eu:
        match_str = "✓" if item["match"] else "✗"
        fl.write(
            f"{item['image_name']} | {item['expected']} | {item['detected']} | {match_str} | {item['CER']:.2f} \n")

    for item in results_ru:
        match_str = "✓" if item["match"] else "✗"
        fl.write(
            f"{item['image_name']} | {item['expected']} | {item['detected']} | {match_str} | {item['CER']:.2f} \n")

    fl.write("\n\nSummary:\n")

    total_am = len(results_am)
    matches_am = sum(1 for item in results_am if item["match"])

    total_ge = len(results_ge)
    matches_ge = sum(1 for item in results_ge if item["match"])

    total_eu = len(results_eu)
    matches_eu = sum(1 for item in results_eu if item["match"])

    total_ru = len(results_ru)
    matches_ru = sum(1 for item in results_ru if item["match"])

    fl.write(f"Total Armenian images processed: {total_am}\n")
    if total_am > 0:
        fl.write(f"Successful Armenian matches: {matches_am} ({matches_am / total_am * 100:.2f}%)\n")
        fl.write(
            f"Failed Armenian matches: {total_am - matches_am} ({(total_am - matches_am) / total_am * 100:.2f}%)\n")
        avg_cer_am = sum(item["CER"] for item in results_am) / total_am if total_am > 0 else 0
        fl.write(f"CER Armenian: {avg_cer_am * 100:.2f}%\n\n")
    else:
        fl.write("No Armenian images were processed.\n\n")

    fl.write(f"Total Georgian images processed: {total_ge}\n")
    if total_ge > 0:
        fl.write(f"Successful Georgian matches: {matches_ge} ({matches_ge / total_ge * 100:.2f}%)\n")
        fl.write(
            f"Failed Georgian matches: {total_ge - matches_ge} ({(total_ge - matches_ge) / total_ge * 100:.2f}%)\n")
        avg_cer_ge = sum(item["CER"] for item in results_ge) / total_ge if total_ge > 0 else 0
        fl.write(f"CER Georgian: {avg_cer_ge * 100:.2f}%\n\n")
    else:
        fl.write("No Georgian images were processed.\n\n")

    fl.write(f"Total European images processed: {total_eu}\n")
    if total_eu > 0:
        fl.write(f"Successful European matches: {matches_eu} ({matches_eu / total_eu * 100:.2f}%)\n")
        fl.write(
            f"Failed European matches: {total_eu - matches_eu} ({(total_eu - matches_eu) / total_eu * 100:.2f}%)\n")
        avg_cer_eu = sum(item["CER"] for item in results_eu) / total_eu if total_eu > 0 else 0
        fl.write(f"CER European: {avg_cer_eu * 100:.2f}%\n\n")
    else:
        fl.write("No European images were processed.\n\n")

    fl.write(f"Total Russian images processed: {total_ru}\n")
    if total_ru > 0:
        fl.write(f"Successful Russian matches: {matches_ru} ({matches_ru / total_ru * 100:.2f}%)\n")
        fl.write(f"Failed Russian matches: {total_ru - matches_ru} ({(total_ru - matches_ru) / total_ru * 100:.2f}%)\n")
        avg_cer_ru = sum(item["CER"] for item in results_ru) / total_ru if total_ru > 0 else 0
        fl.write(f"CER Russian: {avg_cer_ru * 100:.2f}%\n\n")
    else:
        fl.write("No Russian images were processed.\n\n")

    fl.write(f"\nProcessing time: {(time.time() - start_time):.2f} seconds\n\n")

    if undetected_images_am:
        fl.write(f"\nArmenian images with no detected text ({len(undetected_images_am)}):\n")
        for img in undetected_images_am:
            fl.write(f"- {img}\n")

    if undetected_images_ge:
        fl.write(f"\nGeorgian images with no detected text ({len(undetected_images_ge)}):\n")
        for img in undetected_images_ge:
            fl.write(f"- {img}\n")

    if undetected_images_eu:
        fl.write(f"\nEuropean images with no detected text ({len(undetected_images_eu)}):\n")
        for img in undetected_images_eu:
            fl.write(f"- {img}\n")

    if undetected_images_ru:
        fl.write(f"\nRussian images with no detected text ({len(undetected_images_ru)}):\n")
        for img in undetected_images_ru:
            fl.write(f"- {img}\n")

print(f"Processing complete. Results saved to plate_results_1000.txt")
