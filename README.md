# 🧠 Brain Tumor Detection System

## 📌 Project Overview

The Brain Tumor Detection System is a deep learning-based web application that classifies brain MRI images into four categories using a Convolutional Neural Network (CNN). The application allows users to upload an MRI scan and predicts the type of brain tumor with a confidence percentage.

---

## 🎯 Objective

To develop an AI-powered system that assists in the classification of brain tumors from MRI images using deep learning techniques.

---

## 🧠 Tumor Classes

- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Flask
- OpenCV
- NumPy
- HTML
- CSS

---

## 📂 Project Structure

```
BrainTumorDetection/
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── app.py
├── train.py
├── predict.py
├── read_image.py
├── brain_tumor_model.keras
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- Upload MRI brain scan images
- Predict tumor type using CNN
- Display prediction confidence
- User-friendly Flask web interface
- Supports four tumor classes

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Snehalatha29/BrainTumorDetection.git
```

### Go to project folder

```bash
cd BrainTumorDetection
```

### Install required libraries

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📸 Application Workflow

1. Upload an MRI image.
2. The CNN model processes the image.
3. The model predicts the tumor type.
4. The confidence percentage is displayed.
5. The uploaded MRI image is shown on the webpage.

---

## 📊 Sample Output

- Tumor Type : Glioma
- Confidence : 100%

---

## 📈 Future Enhancements

- Tumor localization using segmentation models
- Improved accuracy using Transfer Learning
- Support for DICOM images
- Patient history management
- Cloud deployment

---

## 👩‍💻 Developed By

**Snehalatha Mekala**

**Artificial Intelligence & Machine Learning**

---

## 📄 License

This project is developed for educational and academic purposes.