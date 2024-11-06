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

def extract_hu_features_from_folder(path, label, output_binary_folder='binary_images'):
    list_of_files = os.listdir(path)
    features = []
    labels = []
    
    # Tạo thư mục lưu ảnh binary nếu chưa tồn tại
    os.makedirs(output_binary_folder, exist_ok=True)

    for i in list_of_files:
        img = plt.imread(os.path.join(path, i))  # Đọc ảnh từ đường dẫn
        # if img.ndim == 3:  # Nếu ảnh là ảnh màu, chuyển sang không gian màu RGB
        #     img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #     img_gray = img_hsv[:, :, 2]  # Chọn kênh V (Value) từ ảnh HSV
        # else:
        #     img_gray = img
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # resize ảnh với kích thước giảm 10 lần
        # gray_img = cv2.resize(gray_img, (gray_img.shape[1]//10, gray_img.shape[0]//10))
        _, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_img = cv2.bitwise_not(gray_img)
        img_gray=gray_img
        #return gray_img  

        # Lưu ảnh nhị phân
        gray_image_path = os.path.join(output_binary_folder, f'gray_{i}')
        cv2.imwrite(gray_image_path, img_gray)  

        # Trích xuất đặc trưng Hu Moments từ ảnh nhị phân
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

def save_data():
    os.chdir("Hu")
    # Đường dẫn ảnh đầu vào và lưu lại ảnh binary
    output_binary_folder = "binary_image"  # Thư mục lưu ảnh binary
    la_tu_kinh, nhanlatukinh = extract_hu_features_from_folder("../LEAF/THAIDUCTOAN/tu_kinh", 1, output_binary_folder=output_binary_folder)                  #lachi
    la_cham, nhanlacham = extract_hu_features_from_folder("../LEAF/TRANTHANHKHOA/la_cham", 2, output_binary_folder=output_binary_folder)     #lacham
    la_phong, nhanlaphong = extract_hu_features_from_folder("../LEAF/MAIDUCKHIEM/la_phong", 3, output_binary_folder=output_binary_folder)                    #laphong
    la_rau_ma, nhanlarauma = extract_hu_features_from_folder("../LEAF/NGOHUUMINH/rau_ma", 4, output_binary_folder=output_binary_folder)                                       #la tao
    rau_muong, nhanraumuong = extract_hu_features_from_folder("../LEAF/NGUYENTHANHLAN/rau muong", 5, output_binary_folder=output_binary_folder)                                #rau muong

    os.chdir("hutest_hsv")
    # Lưu vào file csv
    features = la_tu_kinh + la_cham + la_phong + la_rau_ma + rau_muong
    labels = nhanlatukinh + nhanlacham + nhanlaphong + nhanlarauma + nhanraumuong
    save_to_csv(features, labels,'HUnhom11.csv')

    # Chuẩn hóa dữ liệu Hu Moments
    input_csv_path = 'HUnhom11.csv'
    output_csv_path =   'HUnhom11Std.csv'
    z_score_standardization(input_csv_path, output_csv_path)

save_data()