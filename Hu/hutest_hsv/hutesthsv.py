import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy.stats import zscore

def extract_feature(image):
    # Trích xuất đặc trưng Hu Moments từ ảnh
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments.flatten()

def extract_hu_features_from_folder(path, label, output_gray_folder='gray_images'):
    list_of_files = os.listdir(path)
    features = []
    labels = []
    
    # Tạo thư mục lưu ảnh xám nếu chưa tồn tại
    os.makedirs(output_gray_folder, exist_ok=True)

    for i in list_of_files:
        img = plt.imread(os.path.join(path, i))  # Đọc ảnh từ đường dẫn
        if img.ndim == 3:  # Nếu ảnh là ảnh màu, chuyển sang không gian màu HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_gray = img_hsv[:, :, 2]  # Chọn kênh V (Value) từ ảnh HSV
        else:
            img_gray = img  

        # Lưu ảnh xám
        gray_image_path = os.path.join(output_gray_folder, f'gray_{i}')
        cv2.imwrite(gray_image_path, img_gray)  

        # Trích xuất đặc trưng Hu Moments từ ảnh xám
        hu_features = extract_feature(img_gray)  
        features.append(hu_features)
        labels.append(label)

    return features, labels

def save_to_csv(features, labels, file_name):
    # Lưu các đặc trưng và nhãn vào tệp CSV
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(file_name, index=False)

def z_score_standardization(input_csv, output_csv):
    # Tải tệp CSV vào DataFrame
    df = pd.read_csv(input_csv)

    # Áp dụng chuẩn hóa Z-score cho các cột đặc trưng (giả sử cột cuối cùng là nhãn)
    feature_columns = df.columns[:-1]  # Loại bỏ cột nhãn
    df[feature_columns] = df[feature_columns].apply(zscore)

    # Lưu dữ liệu đã chuẩn hóa vào tệp CSV mới
    df.to_csv(output_csv, index=False)

# Đường dẫn ảnh đầu vào và lưu lại ảnh xám
output_gray_folder = r'E:/Downloads/DATA/Hu/gray_images'  # Thư mục lưu ảnh xám
la_chi, nhanlachi = extract_hu_features_from_folder(       r"E:/Downloads/DATA/LEAF/THAIDUCTOAN/la_chi", 1, output_gray_folder=output_gray_folder)                  #lachi
la_cham, nhanlacham = extract_hu_features_from_folder(     r"E:/Downloads/DATA/LEAF/TRANTHANHKHOA/la_cham", 2, output_gray_folder=output_gray_folder)     #lacham
la_phong, nhanlaphong = extract_hu_features_from_folder(   r"E:/Downloads/DATA/LEAF/MAIDUCKHIEM/la_phong", 3, output_gray_folder=output_gray_folder)                    #laphong
la_tao, nhanlatao = extract_hu_features_from_folder(       r"E:/Downloads/DATA/LEAF/NGOHUUMINH/la tao", 4, output_gray_folder=output_gray_folder)                                       #la tao
rau_muong, nhanraumuong = extract_hu_features_from_folder( r"E:/Downloads/DATA/LEAF/NGUYENTHANHLAN/rau muong", 5, output_gray_folder=output_gray_folder)                                #rau muong

# Lưu vào file csv
features = la_chi + la_cham + la_phong + la_tao + rau_muong
labels = nhanlachi + nhanlacham + nhanlaphong + nhanlatao + nhanraumuong
save_to_csv(features, labels, r'E:/Downloads/DATA/Hu/hutest_hsv/HUnhom11.csv')

# Chuẩn hóa dữ liệu Hu Moments
input_csv_path =    r'E:/Downloads/DATA/Hu/hutest_hsv/HUnhom11.csv'
output_csv_path =   r'E:/Downloads/DATA/Hu/hutest_hsv/HUnhom11Std.csv'
z_score_standardization(input_csv_path, output_csv_path)
