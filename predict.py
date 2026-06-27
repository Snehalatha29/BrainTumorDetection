import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Class names
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Load MRI image
img = image.load_img(
    "dataset/Testing/glioma/Te-gl_1.jpg",
    target_size=(224, 224)
)

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Get predicted class
predicted_class = classes[np.argmax(prediction)]

# Confidence
confidence = np.max(prediction) * 100

print("Predicted Class :", predicted_class)
print("Confidence :", round(confidence, 2), "%")