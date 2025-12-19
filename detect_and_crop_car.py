import numpy as np
from PIL import Image
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

MIN_AREA_RATIO = 0.06
VEHICLE_CLASSES = {2, 5, 7}  # car, bus, truck

def detect_and_crop_cars(pil_img):
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]

    results = yolo_model(img_np, conf=0.4, iou=0.5, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    cars = []

    for box in boxes:
        cls = int(box.cls[0])

        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_area = (x2 - x1) * (y2 - y1)
        img_area = w * h

        if box_area / img_area < MIN_AREA_RATIO:
            continue

        crop = pil_img.crop((x1, y1, x2, y2))
        cars.append(crop)

    return cars
