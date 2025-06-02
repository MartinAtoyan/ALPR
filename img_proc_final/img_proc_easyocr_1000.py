import os
import re
import csv
import cv2
import math
import time
import easyocr
import numpy as np
from pathlib import Path
from extra_func import (get_rotated_dimensions,
                        select_license_plate_line,
                        clear_text, levenshtein_distance)


class LicensePlateTemplates:
    def __init__(self):
        self.templates = {
            "armenian_standard": (r"^\d{2}[A-Z]{2}\d{3}$", "11AA111"),
            "armenian_old": (r"^\d{3}[A-Z]{2}\d{2}$", "111AA11"),
            "russian_type1": (r"^[A-Z]\d{3}[A-Z]{2}\d{2}$", "A111AA11"),
            "russian_type2": (r"^[A-Z]\d{3}[A-Z]{2}\d{3}$", "A111AA111"),
            "russian_type3": (r"^[A-Z]{2}\d{3}[A-Z]\d{2}$", "AA111A11"),
            "russian_type4": (r"^[A-Z]{2}\d{3}[A-Z]\d{3}$", "AA111A111"),
            "european_standard": (r"^[A-Z]{2}\d{2}[A-Z]{2}$", "AB12CD"),
            "european_extended": (r"^[A-Z]{1,3}\d{1,4}[A-Z]{1,3}$", "AB123CD"),
            "georgian_standard": (r"^[A-Z]{2}\d{3}[A-Z]{2}$", "AB123CD"),
            "georgian_old": (r"^\d{3}[A-Z]{3}$", "123ABC"),
            "generic_6char": (r"^[A-Z0-9]{6}$", "A1B2C3"),
            "generic_7char": (r"^[A-Z0-9]{7}$", "A1B2C3D"),
            "generic_8char": (r"^[A-Z0-9]{8}$", "A1B2C3D4"),
        }

        self.country_templates = {
            "armenian": ["armenian_standard", "armenian_old", "generic_6char", "generic_7char"],
            "russian": ["russian_type1", "russian_type2", "russian_type3", "russian_type4", "generic_8char"],
            "eu": ["european_standard", "european_extended", "generic_6char", "generic_7char"],
            "georgian": ["georgian_standard", "georgian_old", "generic_6char", "generic_7char"]
        }

    def matches_template(self, text, template_names=None):
        if not text:
            return False, None, 0.0

        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        cleaned_text = self._remove_country_codes(cleaned_text)

        if not cleaned_text:
            return False, None, 0.0

        if template_names is None:
            template_names = list(self.templates.keys())

        best_match = None
        best_confidence = 0.0
        best_template = None

        for template_name in template_names:
            if template_name in self.templates:
                pattern, example = self.templates[template_name]
                if re.match(pattern, cleaned_text):
                    confidence = self._calculate_match_confidence(cleaned_text, pattern)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_template = template_name
                        best_match = True

        return best_match or False, best_template, best_confidence

    def _remove_country_codes(self, text):
        country_codes = ['AM', 'RU', 'RUS', 'ARM', 'UA', 'UKR', 'GE', 'GEO', 'BY', 'BLR', 'EU']
        for code in country_codes:
            if text.startswith(code):
                text = text[len(code):]
            if text.endswith(code):
                text = text[:-len(code)]
        return text.strip()

    def _calculate_match_confidence(self, text, pattern):
        if re.match(pattern, text):
            base_confidence = 0.8
            length_bonus = min(len(text) * 0.02, 0.2)
            return min(base_confidence + length_bonus, 1.0)
        return 0.0

    def get_template_info(self):
        return {name: example for name, (_, example) in self.templates.items()}

    def get_country_templates(self, country):
        return self.country_templates.get(country, list(self.templates.keys()))


def enhanced_clear_text(string: str):
    return (string.replace("/", "").replace(",", "").
            replace(".", "").replace("-", "").
            replace(" ", "").replace("|", "").
            replace("_", "").replace(":", "").lower())


def process_license_plate_image(image_path, reader, plate_templates, country_type="armenian", display=False):
    filename = Path(image_path).name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Could not load: {filename}")
        return None

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
        original_display = cv2.resize(image, display_size)
    else:
        rotated_display = rotated.copy()
        original_display = cv2.resize(image, (new_w, new_h))

    rotated_gray = cv2.cvtColor(rotated_display, cv2.COLOR_BGR2GRAY)

    upscaled = cv2.resize(rotated_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    result = reader.readtext(upscaled, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    print(f"    OCR Raw Results for {filename}:")
    for i, (bbox, text, prob) in enumerate(result):
        print(f"      {i + 1}. Text: '{text}' | Confidence: {prob:.3f}")

    target_templates = plate_templates.get_country_templates(country_type)

    detected_text = ""
    highest_prob = 0
    template_match = False
    matched_template = None
    template_confidence = 0.0

    valid_candidates = []

    for bbox, text, prob in result:
        cleaned_text = enhanced_clear_text(text)
        cleaned_text = plate_templates._remove_country_codes(cleaned_text)
        is_match, template_name, temp_conf = plate_templates.matches_template(
            cleaned_text, target_templates)

        if is_match:
            combined_confidence = (prob + temp_conf) / 2
            valid_candidates.append({
                'text': cleaned_text,
                'ocr_confidence': prob,
                'template_confidence': temp_conf,
                'combined_confidence': combined_confidence,
                'template_name': template_name
            })

    if valid_candidates:
        best_candidate = max(valid_candidates, key=lambda x: x['combined_confidence'])
        detected_text = best_candidate['text']
        highest_prob = best_candidate['ocr_confidence']
        template_match = True
        matched_template = best_candidate['template_name']
        template_confidence = best_candidate['template_confidence']
    else:
        for bbox, text, prob in result:
            if prob > highest_prob:
                highest_prob = prob
                cleaned_fallback = enhanced_clear_text(text)
                detected_text = plate_templates._remove_country_codes(cleaned_fallback)

    if display and rotated_gray is not None and original_display is not None:
        cleaned_bgr = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)

        h1, w1 = cleaned_bgr.shape[:2]
        h2, w2 = original_display.shape[:2]

        if h1 != h2:
            target_height = min(h1, h2)
            cleaned_bgr = cv2.resize(cleaned_bgr, (int(w1 * target_height / h1), target_height))
            original_display = cv2.resize(original_display, (int(w2 * target_height / h2), target_height))

        concatenated = np.concatenate((cleaned_bgr, original_display), axis=1)

        status = "VALID PLATE" if template_match else "NO MATCH"
        text1 = f"File: {filename}"
        text2 = f"Text: '{detected_text}' | OCR Conf: {highest_prob:.3f}"
        text3 = f"Status: {status}"
        if template_match:
            text3 += f" | Template: {matched_template} | T.Conf: {template_confidence:.3f}"

        print(text1)
        print(text2)
        print(text3)

        cv2.imshow("License Plate Detection", concatenated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            return None

    return {
        "filename": filename,
        "detected_text": detected_text,
        "confidence": highest_prob,
        "template_match": template_match,
        "matched_template": matched_template,
        "template_confidence": template_confidence,
        "skew_angle": angle
    }


def image_name_in_expected_plates(image_name_func, detected_text, highest_prob, expect_plates, results_func):
    if image_name_func in expect_plates:
        expected_plate_func = expect_plates[image_name_func]
        cer = 1 - (levenshtein_distance(expected_plate_func, detected_text) / max(1, len(expected_plate_func)))
        results_func.append({
            "image_name": image_name_func,
            "expected": expected_plate_func,
            "detected": detected_text,
            "match": expected_plate_func == detected_text,
            "confidence": highest_prob,
            "CER": cer
        })


def main():
    csv_file = "/home/martin/PycharmProjects/ALPR/ALPR-150/number_plate_1000.csv"

    image_folders = {
        "armenian": Path("/home/martin/Desktop/cropped_plates_1000/images/armenian"),
        "georgian": Path("/home/martin/Desktop/cropped_plates_1000/images/georgian"),
        "eu": Path("/home/martin/Desktop/cropped_plates_1000/images/eu/images"),
        "russian": Path("/home/martin/Desktop/cropped_plates_1000/images/russian")
    }

    reader_en = easyocr.Reader(['en'])
    reader_multilang = easyocr.Reader(['en', 'hy'])

    plate_templates = LicensePlateTemplates()

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

    all_results = {
        "armenian": [],
        "georgian": [],
        "eu": [],
        "russian": []
    }

    undetected_images = {
        "armenian": [],
        "georgian": [],
        "eu": [],
        "russian": []
    }

    template_matched_images = {
        "armenian": [],
        "georgian": [],
        "eu": [],
        "russian": []
    }

    print("Available license plate templates:")
    for name, example in plate_templates.get_template_info().items():
        print(f"  {name}: {example}")
    print()

    for country, image_folder in image_folders.items():
        if not image_folder.exists():
            print(f"Warning: Folder {image_folder} does not exist, skipping {country}")
            continue

        print(f"\nProcessing {country.upper()} images...")

        current_reader = reader_multilang if country == "armenian" else reader_en

        for image_path in sorted(image_folder.glob("*.png")):
            print(f"Processing: {image_path.name}")

            result = process_license_plate_image(
                image_path, current_reader, plate_templates, country, display=False
            )

            if result is None:
                undetected_images[country].append(image_path.name)
                continue

            detected_text = result["detected_text"]
            confidence = result["confidence"]
            template_match = result["template_match"]

            image_name_in_expected_plates(
                result["filename"], detected_text, confidence,
                expected_plates, all_results[country]
            )

            if template_match:
                template_matched_images[country].append({
                    'filename': result["filename"],
                    'text': detected_text,
                    'template': result["matched_template"],
                    'ocr_confidence': confidence,
                    'template_confidence': result["template_confidence"],
                    'skew_angle': result["skew_angle"]
                })
                print(f"  ✓ TEMPLATE MATCH: '{detected_text}' "
                      f"(Template: {result['matched_template']}, "
                      f"Confidence: {confidence:.3f}, "
                      f"Skew: {result['skew_angle']:.1f}°)")
            else:
                print(f"  ✗ No template match: '{detected_text}' (Confidence: {confidence:.3f})")

    output_file = "enhanced_plate_results_1000.txt"
    with open(output_file, 'w', newline='') as fl:
        fl.write("Enhanced Multi-Country License Plate Recognition Results\n")
        fl.write("=" * 80 + "\n")
        fl.write("Image Name | Expected | Detected | Match | Confidence | CER | Country\n")
        fl.write("-" * 80 + "\n")

        for country in ["armenian", "georgian", "eu", "russian"]:
            fl.write(f"\n{country.upper()} RESULTS:\n")
            fl.write("-" * 40 + "\n")

            for item in all_results[country]:
                match_str = "✓" if item["match"] else "✗"
                fl.write(
                    f"{item['image_name']} | {item['expected']} | {item['detected']} | "
                    f"{match_str} | {item['confidence']:.2f} | {item['CER']:.3f} | {country}\n"
                )

        fl.write("\n\nDETAILED SUMMARY:\n")
        fl.write("=" * 50 + "\n")

        total_processed = 0
        total_matches = 0
        total_template_matches = 0
        total_cer_sum = 0.0

        for country in ["armenian", "georgian", "eu", "russian"]:
            country_results = all_results[country]
            country_total = len(country_results)
            country_matches = sum(1 for item in country_results if item["match"])
            country_template_matches = len(template_matched_images[country])

            total_processed += country_total
            total_matches += country_matches
            total_template_matches += country_template_matches

            fl.write(f"\n{country.upper()} Statistics:\n")
            fl.write(f"  Total images processed: {country_total}\n")

            if country_total > 0:
                accuracy = (country_matches / country_total) * 100
                template_rate = (country_template_matches / country_total) * 100
                avg_cer = sum(item["CER"] for item in country_results) / country_total
                total_cer_sum += sum(item["CER"] for item in country_results)

                fl.write(f"  Successful matches: {country_matches} ({accuracy:.2f}%)\n")
                fl.write(f"  Template matches: {country_template_matches} ({template_rate:.2f}%)\n")
                fl.write(f"  Average CER: {avg_cer:.3f} ({avg_cer * 100:.2f}%)\n")
                fl.write(f"  Failed matches: {country_total - country_matches} "
                         f"({((country_total - country_matches) / country_total) * 100:.2f}%)\n")

                if undetected_images[country]:
                    fl.write(f"  Undetected images: {len(undetected_images[country])}\n")
            else:
                fl.write("  No images processed.\n")

        fl.write(f"\nOVERALL SUMMARY:\n")
        fl.write(f"Total images processed: {total_processed}\n")
        if total_processed > 0:
            overall_accuracy = (total_matches / total_processed) * 100
            overall_template_rate = (total_template_matches / total_processed) * 100
            overall_cer = total_cer_sum / total_processed
            fl.write(f"Overall accuracy: {overall_accuracy:.2f}%\n")
            fl.write(f"Overall template match rate: {overall_template_rate:.2f}%\n")
            fl.write(f"Overall CER: {overall_cer:.3f} ({overall_cer * 100:.2f}%)\n")

        fl.write(f"\nProcessing time: {(time.time() - start_time):.2f} seconds\n")

        fl.write(f"\n🎯 VALID LICENSE PLATES BY TEMPLATE:\n")
        fl.write("=" * 50 + "\n")

        for country in ["armenian", "georgian", "eu", "russian"]:
            if template_matched_images[country]:
                fl.write(f"\n{country.upper()} Template Matches:\n")
                for match in template_matched_images[country]:
                    expected_plate = None
                    cer_value = "N/A"
                    for result_item in all_results[country]:
                        if result_item["image_name"] == match["filename"]:
                            expected_plate = result_item["expected"]
                            cer_value = f"{result_item['CER']:.3f}"
                            break

                    fl.write(f"  📋 {match['filename']:<25} → {match['text']:<10} "
                             f"(Expected: {expected_plate}, CER: {cer_value}, "
                             f"Template: {match['template']}, OCR: {match['ocr_confidence']:.3f}, "
                             f"Skew: {match['skew_angle']:.1f}°)\n")

        for country in ["armenian", "georgian", "eu", "russian"]:
            if undetected_images[country]:
                fl.write(f"\n{country.upper()} images with no detected text "
                         f"({len(undetected_images[country])}):\n")
                for img in undetected_images[country]:
                    fl.write(f"  - {img}\n")

    cv2.destroyAllWindows()

    print(f"\nProcessing complete! Results saved to {output_file}")
    print(f"Total processing time: {(time.time() - start_time):.2f} seconds")

    return all_results, template_matched_images, undetected


if __name__ == "__main__":
    results, template_matches, undetected = main()
