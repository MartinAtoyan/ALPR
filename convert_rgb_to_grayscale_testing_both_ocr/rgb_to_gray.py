import cv2
from pathlib import Path

gray_dataset_499_path = Path("/home/martin/Desktop/gray_499")
gray_dataset_1000_path = Path("/home/martin/Desktop/gray_1000")

dataset_499_path = Path("/home/martin/Desktop/cropped_plates/images")
dataset_1000_paths = {
    "armenian": Path("/home/martin/Desktop/cropped_plates_1000/images/armenian"),
    "georgian": Path("/home/martin/Desktop/cropped_plates_1000/images/georgian"),
    "eu": Path("/home/martin/Desktop/cropped_plates_1000/images/eu/images"),
    "russian": Path("/home/martin/Desktop/cropped_plates_1000/images/russian"),
}

gray_dataset_499_path.mkdir(parents=True, exist_ok=True)

for image_path in sorted(dataset_499_path.glob("*.png")):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_path = gray_dataset_499_path / image_path.name
    cv2.imwrite(str(output_path), gray_image)

for name, input_path in dataset_1000_paths.items():
    output_path = gray_dataset_1000_path / name
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(input_path.glob("*.png")):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_path = output_path / image_path.name
        cv2.imwrite(str(save_path), gray_image)
