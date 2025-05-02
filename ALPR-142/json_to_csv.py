import os
import json
import csv
import dotenv

json_files_path = os.environ.get("json_files_path")
json_files = sorted(os.listdir(json_files_path))

count = 0

with open("number_plate.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=' ')
    
    for file in json_files:
        file_path = os.path.join(json_files_path, file)
        with open(file_path) as json_file:
            fl = json.load(json_file)
            
            image_name = fl["name"] + ".png"
            plate_number = fl["description"]
            
            writer.writerow([image_name, plate_number])
            count += 1

# print(count) 499
