import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# 设置数据集路径
dataset_path = 'C:\\Users\\13729\\Desktop\\HAM10000'
metadata_path = os.path.join(dataset_path, 'HAM10000_metadata.csv')

# 读取metadata
metadata = pd.read_csv(metadata_path)

# 划分训练集和验证集
train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['dx'])

# 获取图片文件的完整路径
def get_image_full_path(image_id):
    part1_path = os.path.join(dataset_path, 'HAM10000_images_part_1', image_id + '.jpg')
    part2_path = os.path.join(dataset_path, 'HAM10000_images_part_2', image_id + '.jpg')

    if os.path.exists(part1_path):
        return part1_path
    elif os.path.exists(part2_path):
        return part2_path
    else:
        return None

# 定义图像预处理函数
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_equ = cv2.equalizeHist(img_gray)
    return img_equ

# 创建文件夹以保存预处理后的图像
os.makedirs(os.path.join(dataset_path, 'train_processed'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'val_processed'), exist_ok=True)

# 对训练集和验证集的图像进行预处理并保存
def save_preprocessed_images(df, dataset_type):
    for index, row in df.iterrows():
        img_path = get_image_full_path(row['image_id'])
        img = preprocess_image(img_path)
        save_path = os.path.join(dataset_path, dataset_type + '_processed', row['image_id'] + '.jpg')
        cv2.imwrite(save_path, img)

save_preprocessed_images(train_df, 'train')
save_preprocessed_images(val_df, 'val')
