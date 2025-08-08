from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model and label map
model = load_model("fashion_cnn_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

IMAGE_SIZE = (64, 64)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("static", filename)
            file.save(filepath)

            img = load_img(filepath, target_size=IMAGE_SIZE)
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            label_index = np.argmax(pred)
            prediction = label_map[label_index]
            image_path = filepath

    return render_template("index.html", prediction=prediction, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
