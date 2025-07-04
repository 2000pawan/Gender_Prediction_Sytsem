# 👤 Gender Prediction System using Machine Learning

This is a GUI-based application built with **Python**, **Tkinter**, **OpenCV**, and a **Machine Learning model** to predict gender from images, videos, or live camera feeds using facial features.

---

## 🔍 Features

- 📷 Predict gender from uploaded **images**
- 🎞️ Predict gender from **video files**
- 📹 Predict gender from **live webcam feed**
- 🧠 Trained ML model integrated for real-time predictions
- 🔐 Simple **login system**
- 🖥️ Interactive GUI using **Tkinter**
- 📦 Packaged and modular code structure

---

## 🛠️ Technologies Used

- Python
- Tkinter (GUI)
- OpenCV (Computer Vision)
- PIL (Image Display)
- Scikit-learn (Machine Learning)
- Joblib (Model Serialization)
- Haar Cascade (Face Detection)

---

## 📁 Project Structure

    Gender-Prediction-System/

       ├── model.pkl # Trained gender classification model

       ├── haarcascade_frontalface_default.xml # Face detection model

       ├── app.py # Main application code

       ├── README.md # Project documentation 

       ├── screenshots/
           ├── login_page.png
           ├── main_menu.png
           |── image_prediction.png


---

## 🚀 Getting Started

### 1. Clone the Repository

    https://github.com/2000pawan/Gender_Prediction_Sytsem.git
    cd Gender-Prediction-System

### 2. Install Dependencies
Make sure you have Python 3 installed. Then, install required packages:

       pip install opencv-python Pillow numpy scikit-learn joblib

### 3. Run the Application

       python app.py

### 4. Login Credentials

Username: admin

Password: admin

## 🤖 How the Model Works

The ML model is trained to classify facial images as either "Male" or "Female".

It uses grayscale facial pixel values, resized to 90x90, as input features.

Prediction is made using a pre-trained model saved as model.pkl.

## Screenshots

## Screenshots

### 🔐 Login Page
![Login Page](login.png)

### 🏠 Main Menu
![Main Menu](main.png)

### 📷 Image Prediction Example
![Prediction](image_prediction.png)


## 🖼 Sample Output

A bounding box and label (e.g., Gender: Male) are drawn around detected faces in images or video frames.

## 📌 Notes

To train your own model, use a labeled dataset of facial images and preprocess them into grayscale 90x90 arrays. If any Problem occur then use Gender Prediction.ipynb file for step by step process.

Replace the current model.pkl with your own if needed.

## 👨‍💻 Developed By

PAWAN YADAV

(AI Engineer) | 2025

📧 Contact: yaduvanshi2000pawan@gmail.com

## 📜 License

This project is licensed under the MIT License.


