from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
import io
import json
import os
import numpy as np

from detect_and_crop_car import  detect_and_crop_cars

app = FastAPI(title="Распознавание автомобилей (PyTorch)")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "car_brand_resnet50.pth"
CLASSES_PATH = "car_classes.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(classes))
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    model_loaded = True
    print("✅ PyTorch модель загружена")
else:
    model = None
    classes = []
    model_loaded = False
    print("❌ Модель не найдена")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("car_index.html", {
        "request": request,
        "classes": classes,
        "model_loaded": model_loaded
    })

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return templates.TemplateResponse("car_result.html", {
            "request": request,
            "error": "Модель не загружена"
        })

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # ===== DETECT ALL CARS =====
    MIN_CONFIDENCE = 50.0

    cars = detect_and_crop_cars(img)

    if not cars:
        return templates.TemplateResponse("car_result.html", {
            "request": request,
            "error": "Автомобили подходящего размера не найдены"
        })

    results = []

    for idx, car_img in enumerate(cars):
        img_tensor = transform(car_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        confidence, pred_class = torch.max(probs, 0)
        confidence = confidence.item() * 100

        if confidence < MIN_CONFIDENCE:
            continue  # ❌ неуверенно

        # save crop
        os.makedirs("static/uploads", exist_ok=True)
        filename = f"{np.random.randint(1e9)}_{idx}.jpg"
        path = f"static/uploads/{filename}"
        car_img.save(path)

        results.append({
            "index": idx + 1,
            "class_name": classes[pred_class.item()],
            "confidence": round(confidence, 2),
            "image_url": f"/{path}",
            "all_predictions": [
                {
                    "name": classes[i],
                    "prob": round(float(probs[i]) * 100, 2)
                }
                for i in range(len(classes))
            ]
        })

    if not results:
        return templates.TemplateResponse("car_result.html", {
            "request": request,
            "error": "Обнаружены автомобили, но уверенность распознавания ниже порога"
        })

    return templates.TemplateResponse("car_result.html", {
        "request": request,
        "results": results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
