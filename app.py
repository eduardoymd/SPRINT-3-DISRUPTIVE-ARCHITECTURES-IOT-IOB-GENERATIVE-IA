from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import shutil
import os
from ultralytics import YOLO

app = FastAPI()
model = YOLO("models/best.pt")

output_dir = "videos_output"
os.makedirs(output_dir, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Upload de Vídeo - YOLOv8</title>
    </head>
    <body>
        <h1>Enviar vídeo para detecção/segmentação</h1>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".mp4,.avi" required>
            <button type="submit">Enviar vídeo</button>
        </form>
    </body>
    </html>
    """


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(output_dir, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model(video_path, show=True, save=True)

    return {"message": f"Vídeo {file.filename} processado com sucesso!", "path": video_path}
