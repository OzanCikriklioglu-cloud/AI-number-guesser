import tkinter as tk
import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

global model

def initialize_model():
    global model
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
 
    X = X.astype(np.float32) / 255.0
    
 
    X = X.reshape(-1, 28, 28, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

  
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

class DigitRecognizerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("AI number guesser")
        
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black", cursor="hand2")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        self.image = Image.new("L", (28, 28), 0)  
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_ui()
        self.setup_bindings()

    def setup_ui(self):
        self.clear_btn = tk.Button(self.root, text="CLEAN", command=self.clear_all, bg="#ff4444", fg="white")
        self.predict_btn = tk.Button(self.root, text="GUESS", command=self.predict_digit, bg="#44ff44", fg="black")
        self.result_label = tk.Label(self.root, text="GUESS: ", font=("Roboto", 16, "bold"), bg="black", fg="white")
        
        self.clear_btn.grid(row=1, column=0, pady=5, sticky="ew")
        self.predict_btn.grid(row=1, column=1, pady=5, sticky="ew")
        self.result_label.grid(row=2, column=0, columnspan=2)

    def setup_bindings(self):
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.refresh_image)

    def paint(self, event):
        x = event.x // 10
        y = event.y // 10
        
        self.canvas.create_oval(
            event.x-8, event.y-8,
            event.x+8, event.y+8,
            fill="white", outline="white"
        )
        
        self.draw.rectangle([x-1, y-1, x+1, y+1], fill=255)

    def refresh_image(self, event):
        img = ImageOps.invert(self.image)
        bbox = img.getbbox()
        
        if bbox:
            img = img.crop(bbox)
            img.thumbnail((22, 22))
            img = ImageOps.pad(img, (28, 28), color=0)
        else:
            img = Image.new("L", (28, 28), 0)
        
        self.image = ImageOps.invert(img)
        self.draw = ImageDraw.Draw(self.image)

    def clear_all(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="GUESS: ")

    def predict_digit(self):
        try:
            img_array = np.array(self.image, dtype=np.float32) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1) 
            
            prediction = model.predict(img_array, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            color = "#00ff00" if confidence > 0.8 else "#ff9900"
            self.result_label.config(
                text=f"GUESS: {predicted_digit} (%{confidence * 100:.1f})",
                fg=color
            )
        except Exception as e:
            self.result_label.config(text=f"ERROR: {str(e)}", fg="red")

initialize_model()

root = tk.Tk()
app = DigitRecognizerPro(root)
root.mainloop()