from roboflow import Roboflow

def baixar_dataset():
    rf = Roboflow(api_key="u7iAkaqi2QaZ9tWepfUs")
    project = rf.workspace("fariz-project").project("motorcycle-parking")
    version = project.version(2)
    dataset = version.download("yolov8")

    print(f"Dataset salvo em: {dataset.location}")

if __name__ == "__main__":
    baixar_dataset()
