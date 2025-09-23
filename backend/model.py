import torch
from PIL import Image
from torchvision import transforms

# Simulación de un modelo entrenado
# En un caso real, cargarías un modelo con torch.load("xception_ffpp.pth")
class DummyModel:
    def __init__(self):
        pass

    def predict(self, tensor):
        # Simulación: si el promedio de píxeles > 0.5 decimos "FALSA"
        score = tensor.mean().item()
        if score > 0.5:
            return "Falsa"
        else:
            return "Verdadera"

model = DummyModel()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    return model.predict(tensor)

