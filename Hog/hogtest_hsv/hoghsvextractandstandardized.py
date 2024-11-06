import cv2
import numpy as np
from skimage.feature import hog
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy.stats import zscore  # Nhập khẩu thư viện zscore để chuẩn hóa

def extract_hog_features(image):
    # Trích xuất đặc trưng HOG từ ảnh xám
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    return features

def extract_hog_features_from_folder(path, label, size=(128, 64), output_gray_folder='gray_images'):  # Thêm tham số cho thư mục lưu ảnh xám
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
            img_gray = img  # Nếu ảnh đã là ảnh xám

        # Lưu ảnh xám
        gray_image_path = os.path.join(output_gray_folder, f'gray_{i}')
        cv2.imwrite(gray_image_path, img_gray) 
        # Thay đổi kích thước ảnh
        img_gray = cv2.resize(img_gray, size)  #
        
        hog_features = extract_hog_features(img_gray)  # Trích xuất đặc trưng HOG
        features.append(hog_features)
        labels.append(label)
    return features, labels

def save_to_csv(features, labels, file_name):
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(file_name, index=False)

def standardize_features(input_csv, output_csv):
    # Tải tệp CSV vào DataFrame
    df = pd.read_csv(input_csv)

    # Áp dụng chuẩn hóa Z-score cho các cột đặc trưng (giả sử cột cuối cùng là nhãn)
    feature_columns = df.columns[:-1]  # Loại bỏ cột nhãn
    df[feature_columns] = df[feature_columns].apply(zscore)

    # Lưu dữ liệu đã chuẩn hóa vào tệp CSV mới
    df.to_csv(output_csv, index=False)

def save_data():
    # Đường dẫn ảnh đầu vào và lưu lại ảnh xám
    os.chdir("Hog")
    output_gray_folder = 'gray_images'  # Thư mục lưu ảnh xám
    la_tu_kinh, nhanlatukinh = extract_hog_features_from_folder("../LEAF/THAIDUCTOAN/tu_kinh", 1, output_gray_folder=output_gray_folder)                  #lachi
    la_cham, nhanlacham = extract_hog_features_from_folder("../LEAF/TRANTHANHKHOA/la_cham", 2, output_gray_folder=output_gray_folder)     #lacham
    la_phong, nhanlaphong = extract_hog_features_from_folder("../LEAF/MAIDUCKHIEM/la_phong", 3, output_gray_folder=output_gray_folder)                    #laphong
    la_rau_ma, nhanlarauma = extract_hog_features_from_folder("../LEAF/NGOHUUMINH/rau_ma", 4, output_gray_folder=output_gray_folder)                                       #la tao
    rau_muong, nhanraumuong = extract_hog_features_from_folder("../LEAF/NGUYENTHANHLAN/rau muong", 5, output_gray_folder=output_gray_folder)                                #rau muong

    os.chdir("hogtest_hsv")
    # Lưu vào file csv
    features = la_tu_kinh + la_cham + la_phong + la_rau_ma + rau_muong
    labels = nhanlatukinh + nhanlacham + nhanlaphong + nhanlarauma + nhanraumuong
    save_to_csv(features, labels, 'HOGnhom11.csv')

    # Chuẩn hóa dữ liệu HOG
    input_csv_path =  'HOGnhom11.csv'
    output_csv_path = 'HOGnhom11Std.csv'
    standardize_features(input_csv_path, output_csv_path)

save_data()