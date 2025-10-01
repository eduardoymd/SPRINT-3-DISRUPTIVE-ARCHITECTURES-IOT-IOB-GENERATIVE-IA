# Motorcycle Parking Detection and Segmentation System

## About
Este projeto utiliza **YOLOv8** para **detecção e segmentação de motocicletas** em vídeos.  
Ele permite identificar motos em tempo real, registrar posições e analisar padrões de estacionamento. O sistema foi treinado com um **dataset customizado do Roboflow** e é facilmente extensível a novos vídeos.

## Features
- Detecção de motocicletas em vídeos  
- Segmentação para identificar a área exata da moto  
- Suporte a múltiplos vídeos em lote  
- Modelos treinados salvos para reutilização sem retraining  
- Testado com GPU (RTX 4070 ou superior) para treino rápido

## Requirements
- Python 3.10/11 (versões mais atualizadas ainda sem suporte)  
- PyTorch  
- Ultralytics YOLOv8  
- Roboflow  
- OpenCV  

Instale todas as dependências via `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Treinamento do modelo

Para treinar seu modelo YOLOv8 com o dataset do Roboflow:

1. Abra o `train.py`
2. Configure os caminhos e parâmetros:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Modelo pré-treinado

model.train(
    data="Motorcycle-Parking-2/data.yaml",  # Caminho para seu dataset
    epochs=100,                             # Número de épocas
    batch=16                                # Tamanho do batch
)
```
**Importante: O número atribuído em **batch** está ligado ao desempenho e capacidade da sua GPU, abaixe (8) ou aumente (32) conforme a sua máquina.**
### 1.1. Execute o script
```python
python train.py
```
- Pesos treinados serão salvos em: ```runs/segment/train/weights/best.pt```
- Não é necessário treinar novamente para rodar detecção ou segmentação.
  
### 2. Detecção de motos em vídeos
- Coloque os vídeos na pasta ```videos_test/```:
```
videos_test/
├─ test1.mp4
├─ test2.mp4
```
### 2.1. Configure o Caminho no ```detect.py```
```python
from ultralytics import YOLO
import os

model = YOLO("models/best.pt")

def detectar_video(caminho):
    if os.path.isfile(caminho):
        # Detecta em um único vídeo
        model(caminho, show=True, save=True)
    elif os.path.isdir(caminho):
        # Detecta em todos os vídeos da pasta
        for arquivo in os.listdir(caminho):
            if arquivo.endswith(".mp4"):
                model(os.path.join(caminho, arquivo), show=True, save=True)
```
**Para um único vídeo**
```python
detectar_video("videos_test/test2.avi")
```
**Para todos os vídeos em uma pasta**
```python
detectar_video("videos_test/")
```
- Os resultados serão salvos em ```runs/detect/predict/```

### 3. Segmentação de motos em vídeos
A lógica é igual à detecção, mas usando ```segment.py```
```python
from ultralytics import YOLO
import os

model = YOLO("models/best.pt")

def segmentar_video(caminho):
    if os.path.isfile(caminho):
        model(caminho, show=True, save=True, task="segment")
    elif os.path.isdir(caminho):
        for arquivo in os.listdir(caminho):
            if arquivo.endswith(".mp4"):
                model(os.path.join(caminho, arquivo), show=True, save=True, task="segment")
```
**Para um único vídeo**
```python
segmentar_video("videos_test/test2.avi")
```
**Para todos os vídeos em uma pasta**
```python
segmentar_video("videos_test/")
```
- Os resultados serão salvos em ```runs/segment/predict/```

### 4. Estrutura Recomendada
```
project_root/
│
├─ videos_test/               # Vídeos para teste
│   ├─ test1.mp4
│   └─ test2.mp4
│
├─ runs/                      # Resultados de detecção e segmentação (automático)
│   ├─ detect/
│   └─ segment/
│
├─ train.py                   # Código de treinamento
├─ detect.py                  # Código de detecção
├─ segment.py                 # Código de segmentação
├─ Motorcycle-Parking-2/      # Dataset e data.yaml
├─ models/                    # Pesos treinados
└─ README.md
```
### Notes
- Sempre use GPU para treino e inferência para evitar lentidão ou erros de memória.
- Não versionar vídeos ou pesos grandes no GitHub (runs/ e .venv/ devem estar no .gitignore).
- Modelos treinados podem ser reutilizados em qualquer momento, sem precisar treinar novamente.

## Resultados
### Modelo de Detecção
[Corrida de motos Mottu](https://www.youtube.com/shorts/UnXssdqJNKc)<br>
[Centro de produção Mottu](https://youtube.com/shorts/JjGxi5HUn7k)
### Modelo de Segmentação
[Corrida de motos Mottu](https://youtube.com/shorts/dKT6E-Xx_LU?feature=share)<br>
[Centro de produção Mottu](https://youtube.com/shorts/VJPWDe5eUhw?feature=share)



