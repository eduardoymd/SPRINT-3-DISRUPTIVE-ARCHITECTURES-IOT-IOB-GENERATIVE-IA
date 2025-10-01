from ultralytics import YOLO
import os

model = YOLO("models/best_seg.pt")

def segmentar_video(caminho):
    if os.path.isfile(caminho):
        model(caminho, show=True, save=True)
    elif os.path.isdir(caminho):
        for arquivo in os.listdir(caminho):
            if arquivo.endswith(".mp4"):
                model(os.path.join(caminho, arquivo), show=True, save=True)

if __name__ == "__main__":
    segmentar_video("videos_test/test2.avi")
