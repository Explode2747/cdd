import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('crop_disease_detector.h5')

# Define the class labels
class_labels = ['smut', 'Yellow Leaf', 'Brown Spot', 'Dried Leaves', 'Healthy Leaves', 
                'Banded Chlorosis', 'Grassy shoot', 'Pokkah Boeng', 
                'Sett Rot', 'Viral Disease', 'Brown Spot']  # Replace 'Other Disease' with actual class names

# Function to load and preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match the input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Function to make predictions
def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]  # Get confidence of the prediction

    return class_labels[predicted_class], confidence * 100

# Function to open the file picker and make predictions
def open_file():
    file_path = filedialog.askopenfilename()  # Open file picker dialog
    if file_path:
        # Load and display the image
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize the image to fit in the UI
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        
        # Make prediction and update labels
        disease, confidence = predict_disease(file_path)
        result_label.config(text=f"Disease: {disease}")
        confidence_label.config(text=f"Confidence: {confidence:.2f}%")

# Create the main window
root = tk.Tk()
root.title("Crop Disease Detector")

# File picker button
btn = tk.Button(root, text="Select Image", command=open_file)
btn.pack(pady=20)

# Panel for displaying the image
panel = tk.Label(root)
panel.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="Disease: None", font=("Arial", 14))
result_label.pack(pady=10)

# Label to display the confidence score
confidence_label = tk.Label(root, text="Confidence: N/A", font=("Arial", 14))
confidence_label.pack(pady=5)

# Start the GUI loop
root.mainloop()
