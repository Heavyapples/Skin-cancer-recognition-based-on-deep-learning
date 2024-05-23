import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image
from sklearn.metrics import classification_report
import pickle

lb_save_path = "C:/Users/13729/Desktop/HAM10000/label_binarizer.pkl"
with open(lb_save_path, "rb") as f:
    lb = pickle.load(f)

# 加载模型
model_save_path = "C:/Users/13729/Desktop/HAM10000/skin_cancer_model.h5"
model = load_model(model_save_path)

def import_image():
    global img
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = Image.open(file_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk

def classify_image():
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    result = lb.classes_[np.argmax(prediction)]
    result_label.config(text=f"分类结果: {result}")

from sklearn.metrics import classification_report, accuracy_score

def batch_test():
    test_dir = filedialog.askdirectory()
    csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    test_metadata = pd.read_csv(csv_path)
    test_images = []
    test_labels_list = []

    for index, row in test_metadata.iterrows():
        path = row['path']
        try:
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
            test_images.append(image)
            test_labels_list.append(row['dx'])
        except OSError:
            print(f"Error loading image: {path}. Skipping...")

    test_images = np.array(test_images) / 255.0
    test_labels = lb.transform(test_labels_list)
    predictions = model.predict(test_images)
    report = classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_, output_dict=True)
    accuracy = accuracy_score(test_labels.argmax(axis=1), predictions.argmax(axis=1))
    result_label.config(text=f"批量测试准确率: {accuracy:.2%}")
    print(report)

root = tk.Tk()
root.title("皮肤癌检测")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

canvas = tk.Canvas(frame, width=224, height=224, bg="white")
canvas.grid(row=0, column=0, rowspan=4)

import_button = tk.Button(frame, text="导入图像", command=import_image)
import_button.grid(row=0, column=1, padx=5, pady=5)

classify_button = tk.Button(frame, text="检测", command=classify_image)
classify_button.grid(row=1, column=1, padx=5, pady=5)

result_label = tk.Label(frame, text="分类结果:")
result_label.grid(row=2, column=1, padx=5, pady=5)

batch_test_button = tk.Button(frame, text="批量测试", command=batch_test)
batch_test_button.grid(row=3, column=1, padx=5, pady=5)

root.mainloop()
