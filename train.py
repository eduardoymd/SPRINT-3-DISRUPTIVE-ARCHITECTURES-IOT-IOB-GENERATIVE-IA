import os
from ultralytics import YOLO

def treinar():
    if not os.path.exists("models"):
        os.makedirs("models")

    model = YOLO("yolov8n.pt")

    model.train(
        data="Motorcycle-Parking-2/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0
    )
    model.save("models/best.pt")
    print("Treinamento concluído! Modelo salvo em models/best.pt")

if __name__ == "__main__":
    treinar()
