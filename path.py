import os
import pandas as pd

images_folder = "C:\\Users\\13729\\Desktop\\HAM10000\\ISIC2018_Task3_Test_Images"
input_csv_path = "C:\\Users\\13729\\Desktop\\HAM10000\\ISIC2018_Task3_Test_GroundTruth.csv"
output_csv_path = "C:\\Users\\13729\\Desktop\\HAM10000\\ISIC2018_Task3_Test_GroundTruth_with_path.csv"

def add_image_paths_to_csv(images_folder, input_csv_path, output_csv_path):
    data = pd.read_csv(input_csv_path)
    data['path'] = data['image_id'].apply(lambda x: os.path.join(images_folder, f"{x}.jpg"))

    data.to_csv(output_csv_path, index=False)
    print(f"New CSV file with image paths has been created: {output_csv_path}")

add_image_paths_to_csv(images_folder, input_csv_path, output_csv_path)
