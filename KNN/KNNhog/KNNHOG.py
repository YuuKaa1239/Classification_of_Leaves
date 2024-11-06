# Thiết lập mã hóa để hỗ trợ in ra ký tự Unicode
import sys
import io
import os
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold

# Thiết lập mã hóa
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Bước 1: Tải dataset từ file CSV
dataset = pd.read_csv(r'E:/Downloads/DATA/Hog/hogtest_hsv/HOGnhom11Std.csv')

# Hiển thị số mẫu dữ liệu tương ứng với từng nhãn
print(dataset.groupby('label').size())

# Bước 2: Chia dữ liệu và nhãn
X = dataset.drop('label', axis=1)
y = dataset['label'].astype(str)  # Chuyển đổi y thành chuỗi

# Xác định tất cả các nhãn có trong dataset và chuyển đổi thành chuỗi
all_labels = sorted(y.unique())  # Đảm bảo các nhãn đều là chuỗi
num_classes = len(all_labels)

# Khởi tạo k-fold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Danh sách các giá trị K để thử nghiệm
k_values = [1, 3, 5, 7]

# Biến lưu kết quả
results = []

# Bước 3 vòng lặp qua các giá trị K
for k in k_values:
    print(f"\nThử nghiệm với K = {k}")

    # Khởi tạo mô hình KNN với K hàng xóm
    classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)

    # Khởi tạo biến lưu trữ kết quả trung bình cho các chỉ số hiệu suất
    accuracy_tb = precision_tb = recall_tb = f1_tb = 0
    cm = np.zeros((num_classes, num_classes))  # Ma trận nhầm lẫn tổng có kích thước cố định

    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        # Chia dữ liệu cho từng fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Huấn luyện mô hình KNN
        classifier.fit(X_train, y_train)

        # Dự đoán trên tập kiểm thử
        y_pred = classifier.predict(X_test)

        # Hiển thị báo cáo phân loại cho từng fold
        print(f"Results for fold {fold} with K = {k}:")
        print(classification_report(y_test, y_pred, target_names=all_labels))

        # Ma trận nhầm lẫn cho từng fold
        cm_fold = confusion_matrix(y_test, y_pred, labels=all_labels)
        cm += cm_fold  # Cộng dồn ma trận nhầm lẫn, đảm bảo kích thước cố định

        # Tính toán các chỉ số hiệu suất
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Hiển thị các chỉ số cho từng fold
        print(f"Accuracy for fold {fold}: {accuracy:.4f}")
        print(f"Precision for fold {fold}: {precision:.4f}")
        print(f"Recall for fold {fold}: {recall:.4f}")
        print(f"F1 score for fold {fold}: {f1:.4f}")
        print("--------------------------------------------------")

        # Cộng dồn các chỉ số để tính trung bình sau tất cả các fold
        accuracy_tb += accuracy
        precision_tb += precision
        recall_tb += recall
        f1_tb += f1

    # Tính và hiển thị các chỉ số hiệu suất trung bình cho tất cả các fold
    accuracy_tb /= fold
    precision_tb /= fold
    recall_tb /= fold
    f1_tb /= fold

    # Lưu kết quả cho giá trị K hiện tại
    results.append({
        'K': k,
        'fold': fold,
        'accuracy': accuracy_tb,
        'precision': precision_tb,
        'recall': recall_tb,
        'f1': f1_tb,
        'confusion_matrix': cm  # Lưu ma trận nhầm lẫn tổng
    })

    print(f"K = {k} - Accuracy average: {accuracy_tb:.4f}")
    print(f"K = {k} - Precision average: {precision_tb:.4f}")
    print(f"K = {k} - Recall average: {recall_tb:.4f}")
    print(f"K = {k} - F1 score average: {f1_tb:.4f}")

    # Vẽ ma trận nhầm lẫn đẹp mắt với Seaborn
    class_labels = [str(label) for label in all_labels]  # Nhãn lớp
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.title(f'Confusion Matrix - K: {k}, Fold {fold}\nAccuracy: {accuracy_tb:.4f}, Precision: {precision_tb:.4f}, Recall: {recall_tb:.4f}, F1 Score: {f1_tb:.4f}')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.savefig(f'confusion_matrix_k_{k}_fold_{fold + 1}.png')
    plt.close()  # Đóng biểu đồ để tránh hiện thị đè lên nhau

# Hiển thị kết quả tổng quan cho tất cả các giá trị K đã thử nghiệm
print("\nTổng quan kết quả:")
for result in results:
    print(f"K = {result['K']} - Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1 Score: {result['f1']:.4f}")

# Lưu kết quả vào file CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results_KNN_HOG.csv', index=False)
