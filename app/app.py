from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from torchvision import transforms
import os

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Modelo preentrenado (placeholder)
MODEL_PATH = "./model/xception_ffpp.pth"
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def load_model():
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=False, aux_logits=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    verdict = "Real" if probs[0] > probs[1] else "Falsa"
    return f"{verdict} (Real {probs[0]*100:.2f}% / Falsa {probs[1]*100:.2f}%)"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return redirect(request.url)
    file = request.files["photo"]
    if file.filename == "":
        return redirect(request.url)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = predict_image(file_path)
    return render_template("index.html", result=result, image_url=file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
