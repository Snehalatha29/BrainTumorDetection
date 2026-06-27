from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Upload Folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Trained Model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Class Names
classes = ["glioma", "meningioma", "notumor", "pituitary"]


# Route to Display Uploaded Images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    # Check whether file is uploaded
    if "image" not in request.files:
        return "No file uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No image selected."

    # Save Uploaded Image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load Image
    img = image.load_img(filepath, target_size=(224, 224))

    # Convert Image to Array
    img_array = image.img_to_array(img)

    # Add Batch Dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = round(np.max(prediction) * 100)

    # Return Result
    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)