import os
import csv
import json
import dotenv

json_files_path_am = "/home/martin/Desktop/cropped_plates_1000/labels/armenian"
json_files_path_ge = "/home/martin/Desktop/cropped_plates_1000/labels/georgian"
json_files_path_eu = "/home/martin/Desktop/cropped_plates_1000/images/eu/annotations"
json_files_path_ru = "/home/martin/Desktop/cropped_plates_1000/labels/russian"

json_files_am = sorted(os.listdir(json_files_path_am))
json_files_ge = sorted(os.listdir(json_files_path_ge))
json_files_eu = sorted(os.listdir(json_files_path_eu))
json_files_ru = sorted(os.listdir(json_files_path_ru))

with open("number_plate_1000.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=' ')
    
    for file in json_files_am:
        file_path_am = os.path.join(json_files_path_am, file)
        with open(file_path_am) as json_file:
            fl = json.load(json_file)
            
            image_name = fl["name"] + ".png"
            plate_number = fl["description"]
            
            writer.writerow([image_name, plate_number])

    for file in json_files_ge:
        file_path_ge = os.path.join(json_files_path_ge, file)
        with open(file_path_ge) as json_file:
            fl = json.load(json_file)

            image_name = fl["name"] + ".png"
            plate_number = fl["description"]

            writer.writerow([image_name, plate_number])

    for file in json_files_eu:
        file_path_eu = os.path.join(json_files_path_eu, file)
        with open(file_path_eu) as json_file:
            fl = json.load(json_file)

            image_name = fl["name"] + ".png"
            plate_number = fl["description"]

            writer.writerow([image_name, plate_number])

    for file in json_files_ru:
        file_path_ru = os.path.join(json_files_path_ru, file)
        with open(file_path_ru) as json_file:
            fl = json.load(json_file)

            image_name = fl["name"] + ".png"
            plate_number = fl["description"]

            writer.writerow([image_name, plate_number])