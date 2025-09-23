from flask import Flask, render_template, request
from model import predict_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None
    if request.method == "POST":
        if "file" not in request.files:
            result = "No se subió ningún archivo"
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "Archivo no válido"
            else:
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(img_path)
                result = predict_image(img_path)

    return render_template("index.html", result=result, img_path=img_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

