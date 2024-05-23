import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 读取metadata
metadata_path = "C:/Users/13729/Desktop/HAM10000/HAM10000_metadata.csv"
metadata = pd.read_csv(metadata_path)

# 合并两个图像文件夹
image_dir1 = "C:/Users/13729/Desktop/HAM10000/HAM10000_images_part_1/"
image_dir2 = "C:/Users/13729/Desktop/HAM10000/HAM10000_images_part_2/"
metadata["path"] = metadata["image_id"].apply(lambda x: os.path.join(image_dir1, x + ".jpg") if os.path.isfile(os.path.join(image_dir1, x + ".jpg")) else os.path.join(image_dir2, x + ".jpg"))

# 加载图像数据
images = []
for path in metadata["path"]:
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    images.append(image)

images = np.array(images) / 255.0

# 对标签进行编码
lb = LabelBinarizer()
labels = lb.fit_transform(metadata["dx"])

# 分割训练集和测试集
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# 数据增强
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# 构建模型
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
head_model = base_model.output
head_model = GlobalAveragePooling2D()(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(len(lb.classes_), activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# 冻结base_model的层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练模型
epochs = 1
batch_size = 32
history = model.fit(aug.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, validation_data=(testX, testY), validation_steps=len(testX) // batch_size, epochs=epochs)

# 评估模型
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# 保存模型
#model_save_path = "C:/Users/13729/Desktop/HAM10000/skin_cancer_model.h5"
#model.save(model_save_path)
#print("模型已保存至:", model_save_path)


lb_save_path = "C:/Users/13729/Desktop/HAM10000/label_binarizer.pkl"
with open(lb_save_path, "wb") as f:
    pickle.dump(lb, f)
print("标签编码器已保存至:", lb_save_path)
