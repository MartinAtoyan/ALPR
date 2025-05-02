import os
import csv

csv_file_path = "/Users/picsartacademy/Desktop/ALPR/ALPR-142/number_plate.csv"
images_path = "/Users/picsartacademy/Desktop/cropped_plates/images"

images_lst = os.listdir(images_path)
missing = []

with open(csv_file_path, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=' ')
    for row in reader:
        image_name = row[0]
        if image_name not in images_lst:
            missing.append(image_name)
            print(f"{image_name} not in images")

print(f"Total missing images: {len(missing)}")