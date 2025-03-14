import tensorflow as tf
import tkinter as tk
import numpy as np
import os
import sys
from tkinter import filedialog, messagebox, Frame, Label, Canvas, Button
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DigitRecognitionApp:
    def __init__(self, root):
        self.selectModelFile()

        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.configure(bg="#f0f0f0")
        
        mainFrame = Frame(root, bg="#f0f0f0", padx=20, pady=20)
        mainFrame.pack(expand=True, fill="both")
        
        titleLabel = Label(mainFrame, text="Draw a digit (0-9)", font=("Arial", 16), bg="#f0f0f0")
        titleLabel.pack(pady=(0, 10))
        
        self.canvasFrame = Frame(mainFrame, bg="black", padx=2, pady=2)
        self.canvasFrame.pack(pady=10)

        self.canvas = Canvas(self.canvasFrame, width=280, height=280, bg="white", cursor="pencil")
        self.canvas.pack()

        buttonFrame = Frame(mainFrame, bg="#f0f0f0")
        buttonFrame.pack(pady=10)
        
        self.clearButton = Button(buttonFrame, text="Clear", command=self.clearCanvas, width=10, font=("Arial", 12), bg="#e0e0e0")
        self.clearButton.grid(row=0, column=0, padx=5)
        
        self.recognize_button = Button(buttonFrame, text="Recognize", command=self.recognizeDigit, width=10, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.recognize_button.grid(row=0, column=1, padx=5)
    
        self.resultFrame = Frame(mainFrame, bg="#f0f0f0")
        self.resultFrame.pack(pady=10, fill="x")
        
        self.resultLabel = Label(self.resultFrame, text="Draw a digit and click 'Recognize'",
                                 font=("Arial", 14), bg="#f0f0f0")
        self.resultLabel.pack()
        
        self.image = Image.new("L", (280, 280), color=255)  # "L" mode for grayscale
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.save_image)

        self.prev_x = None
        self.prev_y = None
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 8 
        
        if self.prev_x and self.prev_y:
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, 
                                    width=r*2, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.prev_x, self.prev_y, x, y], 
                          fill="black", width=r*2)
        else:
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill="black")
        
        self.prev_x = x
        self.prev_y = y
    
    def save_image(self, event):
        self.prev_x = None
        self.prev_y = None
        self.image.save("user_input.png")
    
    def recognizeDigit(self):
        img = self.image.resize((28, 28)).convert("L")
        img = np.array(img) 
        img = 255 - img  # Invert the image colors (white background, black digit)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255  # Normalize pixel values to [0, 1]
        
        predictions = self.model.predict(img)
        predictionText = "\n".join([f"Digit {i}: {predictions[0][i]:.4f}" for i in range(10)])
        predictedLabel = np.argmax(predictions)
        messagebox.showinfo("Prediction", f"Prediction Probabilities:\n{predictionText}\n\nPredicted Digit: {predictedLabel}")

    def clearCanvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None
        self.resultLabel.config(text="Draw a digit and click 'Recognize'")

    def loadModel(self, modelPath):
        try:
            model = tf.keras.models.load_model(modelPath)
            print(f"Model loaded successfully from {modelPath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None        

    def selectModelFile(self):
        modelPath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Keras Model Files", "*.keras")])
        if modelPath:
            self.model = self.loadModel(modelPath)

def runApp():
    root = tk.Tk()
    root.geometry("350x450")
    app = DigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    runApp()