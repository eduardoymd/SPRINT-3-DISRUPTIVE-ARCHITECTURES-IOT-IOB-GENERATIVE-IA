from ultralytics import YOLO
import os

model = YOLO("models/best.pt")

def detectar_video(caminho):
    if os.path.isfile(caminho):
        model(caminho, show=True, save=True)
    elif os.path.isdir(caminho):
        for arquivo in os.listdir(caminho):
            if arquivo.endswith(".mp4"):
                model(os.path.join(caminho, arquivo), show=True, save=True)

if __name__ == "__main__":
    detectar_video("videos_test/test2.avi")
