import math
import os
import cv2
import numpy as np
import easyocr
import re
from pathlib import Path

from extra_func import get_rotated_dimensions, select_license_plate_line, image_name_in_expected_plates, clear_text


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
            "generic_6char": (r"^[A-Z0-9]{6}$", "A1B2C3"),
            "generic_7char": (r"^[A-Z0-9]{7}$", "A1B2C3D"),
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

        for template_name in template_names:
            if template_name in self.templates:
                pattern, example = self.templates[template_name]
                if re.match(pattern, cleaned_text):
                    confidence = self._calculate_match_confidence(cleaned_text, pattern)
                    return True, template_name, confidence

        return False, None, 0.0

    def _remove_country_codes(self, text):
        country_codes = ['AM', 'RU', 'RUS', 'ARM', 'UA', 'UKR', 'GE', 'GEO', 'BY', 'BLR']
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


def process_license_plate_image(image_path, reader, plate_templates, target_templates=None):
    filename = Path(image_path).name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Could not load: {filename}")
        return None, None, None, None, None, None

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

    # _, thresh = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    result_am = reader.readtext(upscaled, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')



    print(f"    OCR Raw Results for {filename}:")
    for i, (bbox_am, text_am, prob_am) in enumerate(result_am):
        print(f"      {i + 1}. Text: '{text_am}' | Confidence: {prob_am:.3f}")

    detected_text_am = ""
    highest_prob_am = 0
    template_match = False
    matched_template = None
    template_confidence = 0.0

    valid_candidates = []

    for bbox_am, text_am, prob_am in result_am:
        cleaned_text = clear_text(text_am)
        cleaned_text = plate_templates._remove_country_codes(cleaned_text)
        is_match, template_name, temp_conf = plate_templates.matches_template(cleaned_text, target_templates)

        if is_match:
            combined_confidence = (prob_am + temp_conf) / 2
            valid_candidates.append({
                'text': cleaned_text,
                'ocr_confidence': prob_am,
                'template_confidence': temp_conf,
                'combined_confidence': combined_confidence,
                'template_name': template_name
            })

    if valid_candidates:
        best_candidate = max(valid_candidates, key=lambda x: x['combined_confidence'])
        detected_text_am = best_candidate['text']
        highest_prob_am = best_candidate['ocr_confidence']
        template_match = True
        matched_template = best_candidate['template_name']
        template_confidence = best_candidate['template_confidence']
    else:
        for bbox_am, text_am, prob_am in result_am:
            if prob_am > highest_prob_am:
                highest_prob_am = prob_am
                cleaned_fallback = clear_text(text_am)
                detected_text_am = plate_templates._remove_country_codes(cleaned_fallback)

    return (filename, detected_text_am, highest_prob_am, rotated_gray, original_display,
            template_match, matched_template, template_confidence)


def main():
    folder_path = "/Users/picsartacademy/Desktop/croppedimgs"

    reader = easyocr.Reader(['en'])
    plate_templates = LicensePlateTemplates()

    target_templates = ["armenian_standard", "russian_type1", "russian_type2"]

    print("Available license plate templates:")
    for name, example in plate_templates.get_template_info().items():
        marker = " ← ACTIVE" if target_templates is None or name in target_templates else ""
        print(f"  {name}: {example}{marker}")
    print()

    results_am = []
    undetected_images_am = []
    template_matched_images = []

    folder = Path(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Processing {len(image_files)} images...")

    for image_path in image_files:
        print(f"Processing: {image_path.name}")

        result = process_license_plate_image(image_path, reader, plate_templates, target_templates)

        if result[0] is None:
            continue

        (filename, detected_text, prob, cleaned, original,
         template_match, matched_template, template_confidence) = result

        image_name_in_expected_plates(filename, detected_text, prob, results_am)

        if template_match:
            template_matched_images.append({
                'filename': filename,
                'text': detected_text,
                'template': matched_template,
                'ocr_confidence': prob,
                'template_confidence': template_confidence
            })
            print(f"  ✓ TEMPLATE MATCH: '{detected_text}' (Template: {matched_template}, Confidence: {prob:.3f})")
        else:
            undetected_images_am.append(filename)
            print(f"  ✗ No template match: '{detected_text}' (Confidence: {prob:.3f})")

        if cleaned is not None and original is not None:
            cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

            h1, w1 = cleaned_bgr.shape[:2]
            h2, w2 = original.shape[:2]

            if h1 != h2:
                target_height = min(h1, h2)
                cleaned_bgr = cv2.resize(cleaned_bgr, (int(w1 * target_height / h1), target_height))
                original = cv2.resize(original, (int(w2 * target_height / h2), target_height))

            concatenated = np.concatenate((cleaned_bgr, original), axis=1)

            status = "VALID PLATE" if template_match else "NO MATCH"

            text1 = f"File: {filename}"
            text2 = f"Text: '{detected_text}' | OCR Conf: {prob:.3f}"
            text3 = f"Status: {status}"
            if template_match:
                text3 += f" | Template: {matched_template} | T.Conf: {template_confidence:.3f}"

            print(text1)
            print(text2)
            print(text3)

            cv2.imshow("License Plate Detection", concatenated)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                continue

    cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total images processed: {len(results_am)}")
    print(f"Template matched images: {len(template_matched_images)}")
    print(f"Unmatched images: {len(undetected_images_am)}")

    if template_matched_images:
        print(f"\n🎯 VALID LICENSE PLATES DETECTED:")
        print("-" * 60)
        for match in template_matched_images:
            print(f"📋 {match['filename']:<25} → {match['text']:<10} "
                  f"(Template: {match['template']}, OCR: {match['ocr_confidence']:.3f})")
    else:
        print(f"\n❌ No valid license plates found matching the specified templates.")

    if undetected_images_am:
        print(f"\n🔍 Images without template matches:")
        print("-" * 40)
        for img in undetected_images_am:
            print(f"   {img}")

    return results_am, template_matched_images, undetected_images_am


if __name__ == "__main__":
    results_am, template_matched, undetected = main()

# change for 1000 dataset with labels