import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# 1. Tải lại các tệp pickle (Mô hình, Scaler, Label Encoder)
model_path = 'xgboost_model.pkl'
scaler_path = 'scaler.pkl'
encoder_path = 'label_encoder.pkl'
feature_importance_path = 'Feature_Importance.csv'

# Load mô hình
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Load scaler
with open(scaler_path, 'rb') as file:
    loaded_scaler = pickle.load(file)

# Load label encoder
with open(encoder_path, 'rb') as file:
    loaded_encoder = pickle.load(file)

# Load đặc trưng quan trọng
feature_importance_df = pd.read_csv(feature_importance_path)

# 2. Đọc dữ liệu mới
new_test_data = pd.read_csv('new_test.csv')
X_new_test = new_test_data.drop('Label', axis=1)
y_new_test = new_test_data['Label']

# 3. Tiền xử lý dữ liệu mới (làm sạch, chuẩn hóa, v.v.)
X_new_test = X_new_test.fillna(X_new_test.mean())

# 4. Chuẩn hóa dữ liệu mới sử dụng scaler đã học
X_new_test = loaded_scaler.transform(X_new_test)

# 5. Lọc các đặc trưng quan trọng đã chọn
important_features = feature_importance_df[feature_importance_df['Importance'] > 0.003]['Feature']
important_features_indices = [new_test_data.columns.get_loc(f) for f in important_features]
X_new_test_filtered = X_new_test[:, important_features_indices]

# 6. Dự đoán nhãn cho tập kiểm thử mới
y_new_pred_encoded = loaded_model.predict(X_new_test_filtered)
y_new_pred = loaded_encoder.inverse_transform(y_new_pred_encoded)

# 7. Đánh giá mô hình với tập kiểm thử mới
new_accuracy = accuracy_score(y_new_test, y_new_pred)
print(f"\nAccuracy của mô hình trên tập kiểm thử mới: {new_accuracy:.2f}")
print("\nClassification Report cho dữ liệu mới:")
print(classification_report(y_new_test, y_new_pred))

# 8. Tính Confusion Matrix cho dữ liệu mới
new_cm = confusion_matrix(y_new_test, y_new_pred)
new_labels = loaded_encoder.classes_

# 9. Vẽ Heatmap cho Confusion Matrix
plt.figure(figsize=(18, 15))
new_cm_percentage = new_cm.astype('float') / new_cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(new_cm_percentage, annot=True, fmt=".1f", cmap="Blues", cbar=True,
            xticklabels=new_labels, yticklabels=new_labels, linewidths=0.5, linecolor='black')
plt.title("Confusion Matrix cho Dữ Liệu Mới", fontsize=18, pad=15)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=18)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# 10. Tính TP, FP, TN, FN cho từng lớp
results = []
for i, label in enumerate(new_labels):
    TP = new_cm[i, i]
    FN = new_cm[i, :].sum() - TP
    FP = new_cm[:, i].sum() - TP
    TN = new_cm.sum() - (TP + FN + FP)

    precision = precision_score(y_new_test, y_new_pred, average=None, labels=[label])[0]
    recall = recall_score(y_new_test, y_new_pred, average=None, labels=[label])[0]
    f1 = f1_score(y_new_test, y_new_pred, average=None, labels=[label])[0]

    results.append([label, TP, FP, TN, FN, round(precision, 4), round(recall, 4), round(f1, 4)])

# Chuyển kết quả sang DataFrame
metrics_df = pd.DataFrame(results, columns=['Class', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1-Score'])

# In kết quả dưới dạng bảng có tiêu đề
print("\nBảng TP, FP, TN, FN, Precision, Recall, F1-Score cho từng lớp:")
print(tabulate(metrics_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# 11. Trực quan hóa Precision, Recall, F1-Score
metrics_df.set_index('Class', inplace=True)
metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(15, 8), colormap='coolwarm')
plt.title('Precision, Recall, F1-Score for Each Class')
plt.ylabel('Score')
plt.xlabel('Class')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
