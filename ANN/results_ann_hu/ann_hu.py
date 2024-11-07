import pandas as pd 
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import numpy as np
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Thiết lập mã hóa để hỗ trợ in ra ký tự Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Bước 1: Tải dữ liệu từ file CSV
data = pd.read_csv("Hu\hutest_hsv\HUnhom11Std.csv")

os.chdir("ANN/results_ann_hu")
# Giả sử các cột từ 0 đến n-1 là đặc trưng và cột cuối cùng là nhãn
X = data.iloc[:, :-1].values  # Các đặc trưng
y = data.iloc[:, -1].values  # Nhãn

# Kiểm tra giá trị duy nhất trong y
unique_labels = np.unique(y)
print("Các nhãn duy nhất trong dữ liệu:", unique_labels)

# Điều chỉnh nhãn để chúng nằm trong khoảng từ 0 đến num_classes - 1
y = y - unique_labels[0]  # Điều chỉnh nếu nhãn bắt đầu từ 1 hoặc không bắt đầu từ 0

# Chuyển đổi nhãn thành dạng one-hot encoding
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y.astype(int)]

# Bước 2: K-fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Danh sách số neuron để thử nghiệm
neurons_list = [5, 10, 15, 20]

# Lưu kết quả cho tất cả các số neuron
results = []

# Bước 3: Chia dữ liệu một lần duy nhất cho tất cả các số neuron
for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    print(f"\nFold {fold + 1}")

    # Chia dữ liệu theo chỉ số train và test
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_one_hot[train_idx], y_one_hot[test_idx]

    for n_neurons in neurons_list:
        print(f"\nThử nghiệm với số nơ-ron lớp ẩn: {n_neurons}")

        # Bước 4: Xây dựng mô hình ANN
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))  # Sử dụng lớp Input cho đầu vào
        model.add(Dense(n_neurons, activation='sigmoid'))  # Số nơ-ron thay đổi
        model.add(Dense(num_classes, activation='softmax'))  # Lớp đầu ra

        # Biên dịch mô hình với tốc độ học là 0.05
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.05), metrics=['accuracy'])

        # Huấn luyện mô hình
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # Dự đoán trên tập kiểm tra
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Tính toán confusion matrix và các chỉ số hiệu suất
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

        # Lưu kết quả cho từng fold
        results.append({
            'neurons': n_neurons,
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        })

        # Hiển thị thông số đánh giá
        print(f'Fold {fold + 1} - Số nơ-ron: {n_neurons} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
        print('-----------------------------------------')

        # Vẽ confusion matrix cho từng fold và lưu vào file
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_test_classes), yticklabels=np.unique(y_test_classes))
        plt.title(f'Confusion Matrix - Neurons: {n_neurons}, Fold {fold + 1}\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Lưu hình vào file (đặt tên theo số nơ-ron và số fold)
        plt.savefig(f'confusion_matrix_neurons_{n_neurons}_fold_{fold + 1}.png')
        plt.close()  # Đóng hình để giải phóng bộ nhớ

# Lưu kết quả vào file CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results_ANN_HU.csv', index=False)
