import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import warnings
import kagglehub
import shutil
import os

from module import CustomRandomForestClassifier
from util import cross_val_score

warnings.filterwarnings('ignore')

# 1. Tải dữ liệu
path = kagglehub.dataset_download("armanakbari/connectionist-bench-sonar-mines-vs-rocks")
dst = "data/sonar"
filename = 'sonar.all-data.csv'
print("File path:", os.path.join(os.path.abspath(dst), filename))
shutil.copytree(path, dst, dirs_exist_ok=True)
data = pd.read_csv(os.path.join(dst, filename), header=None)

# 2. Tách Biến độc lập (X) và Biến phụ thuộc (y)
X = data.iloc[1:, 0:-1] 
y = data.iloc[1:, -1]

# 3. Tiền xử lý (Chuyển đổi nhãn y 'R', 'M' thành 0, 1)
y = y.map({'R': 0, 'M': 1})

# 4. Thiết lập quy trình đánh giá chéo (Cross-Validation)
# Đây là bước tương đương với hàm cross_validation_split của bạn
n_folds = 5

# Chúng ta tạo một đối tượng KFold để kiểm soát việc chia
# shuffle=True: Trộn dữ liệu trước khi chia (giống code scratch)
# random_state=2: Đặt seed cho việc trộn (giống seed(2) ở code scratch)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2)

print("Bắt đầu đánh giá mô hình (sklearn 5-fold cross-validation)...\n")

# 5. Lặp qua các số lượng cây để so sánh
for n_trees in [1, 5, 10]:
    
    # Khởi tạo mô hình bên trong vòng lặp
    # random_state=42: Đảm bảo cây được xây dựng giống nhau mỗi lần
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    custom_rf_classifier = CustomRandomForestClassifier(n_estimators=n_trees, random_state=42)
    
    # Hàm cross_val_score sẽ tự động làm toàn bộ quy trình:
    # 1. Chia X, y thành 5 fold (dùng kf)
    # 2. Lặp 5 lần
    # 3. Mỗi lần, huấn luyện trên 4 fold và đánh giá trên 1 fold
    # 4. Trả về một danh sách 5 điểm số (accuracy)
    ref_scores = cross_val_score(rf_classifier, X, y, cv=kf, scoring='accuracy')
    ref_percent_scores = ref_scores * 100.0
    custom_scores = cross_val_score(custom_rf_classifier, X, y, cv=kf, scoring='accuracy')
    custom_percent_scores = custom_scores * 100.0

    print('Trees: %d' % n_trees)
    print('Reference Scores: %s' % np.round(ref_percent_scores, 3))
    print('Reference Mean Accuracy: %.3f%%' % (ref_percent_scores.mean()))
    print('Custom Scores: %s' % np.round(custom_percent_scores, 3))
    print('Custom Mean Accuracy: %.3f%%' % (custom_percent_scores.mean()))
    print('---')