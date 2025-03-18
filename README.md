# Handwritten Digits Recognition App

This project provides a **handwritten digits recognition application** that allows users to draw a number on a canvas and recognize it using **two different models**:
1. **Convolutional Neural Network (CNN)**
2. **Random Forest Classifier**

Users can load models with the `.keras` and `.joblib` extension, draw numbers, and get real-time predictions.

---

## Features

✅ Load trained **CNN** or **Random Forest** models (`.keras` and `.joblib` format)
✅ Draw digits on an interactive canvas 
✅ Recognize handwritten digits with high accuracy
✅ Supports **MNIST dataset-trained models**  
✅ Simple **GUI using Tkinter**  
✅ Available code for both models that can be modified and studied 

---

## Project Structure

```
📁 DigitRecognition/
│── 📁 keras_models/            # Folder for generated models (.keras, .joblib)
│── 📁 models/                  # Source code for models 
│   │── ConvulsionalModel.py     # CNN model training & loading
│   │── RandomForestModel.py     # Random Forest training & loading
│── 📁 src/                     # Source code
│   │── Application.py          # Main application with GUI
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
│── setup.py                    # Setup file
```

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/rrubaszek/DigitRecognition.git
cd DigitRecognition
```

### **2. Create a Virtual Environment (Optional)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## How to Use

### **0. If used the first time, generate preferred model**
```bash
python models/ConvolutionalModel.py
```
or
```bash
python models/RandomForestModel.py
```
- Models are generated to the keras_models/ folder

### **1. Run the Application**
```bash
python src/Application.py
```

### **2. Load a Model**
- Click **"Load Model"** and select a `.keras` or `.joblib` file from the `keras_models/` folder.

### **3. Draw a Digit**
- Use your mouse to **draw a number** on the canvas.

### **4. Recognize the Digit**
- Click the **"Recognize"** button to get the predicted digit.

---

## Training the Models

If you want to **train your own models**, run:

### **Train the CNN Model**
```bash
python models/ConvolutionalModel.py
```
This will save the model in keras_models folder. Also, feel free to experiment with the parameters in the code.


### **Train the Random Forest Model**
```bash
python models/RandomForestModel.py
```
This will save the model in keras_models folder. Also, feel free to experiment with the parameters in the code.

---

## Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** (for CNN model)
- **Scikit-learn** (for Random Forest)
- **Tkinter** (for GUI)
- **PIL** (for image processing)

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to **modify and use** it for your own purposes.

---

## 📬 Contact

For any questions or suggestions, feel free to reach out:  
🌎 GitHub: (https://github.com/rrubaszek)  