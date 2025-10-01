import os
from ultralytics import YOLO

def treinar_segmentacao():
    if not os.path.exists("models"):
        os.makedirs("models")

    model = YOLO("yolov8s-seg.pt")

    model.train(
        data="Motorcycle-Parking-2/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        save_period=5
    )

    model.save("models/best_seg.pt")
    print("Treinamento de segmentação concluído! Modelo salvo em models/best_seg.pt")

if __name__ == "__main__":
    treinar_segmentacao()
