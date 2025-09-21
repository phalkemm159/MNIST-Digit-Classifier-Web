# 🧠 MNIST Digit Classifier Web App

A simple and interactive web application that allows users to draw digits and get real-time predictions using a trained neural network on the MNIST dataset.

---

## 🚀 Features

- 🎨 Draw digits directly on a web canvas
- 🔮 Real-time predictions using a trained Keras model
- 📦 Flask backend for serving predictions
- 🧼 Clear button to reset the canvas
- 🎨 Modern and user-friendly UI

---

## 📁 Project Structure

- MNIST-Digit-Classifier-Web/
- │
- ├── app.py # Flask backend
- ├── mnist_model.h5 # Trained Keras model
- │
- ├── templates/
- │ └── index.html # Main web interface
- │
- ├── static/
- │ └── style.css # (Optional) Custom CSS styles
- │
- ├── requirements.txt # Python dependencies
- └── README.md # Project documentation

---

## 🧠 Model Training (Quick Overview)

The model is a simple feed-forward neural network trained on the MNIST dataset with Keras:

```
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)
model.save("mnist_model.h5")
```

## ✅ Getting Started
1. Clone the Repository
```
git clone https://github.com/yourusername/mnist-digit-classifier-web.git
cd mnist-digit-classifier-web
```
2. Install Dependencies
Use a virtual environment (recommended) and install:
```
pip install -r requirements.txt
3. Run the Flask Server
python app.py
Server will run at http://127.0.0.1:5000
```

4. Open the App
In your browser, open:
```
http://127.0.0.1:5000
```
Draw a digit on the canvas and click Predict!

## 🌐 Deployment
This app is deployed at : Railway 
Link
https://mnist-digit-classifier-project.up.railway.app/


## 📦 Dependencies
Flask

TensorFlow

NumPy

Pillow

Install with:

pip install -r requirements.txt

## ✨ Demo
Give it a Try : https://mnist-digit-classifier-project.up.railway.app/

## 🧑‍💻 Author
Mahesh Phalke
LinkedIn • GitHub

## 📄 License
MIT License – feel free to use and modify.
