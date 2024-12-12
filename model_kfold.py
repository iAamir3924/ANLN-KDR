

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# 读取数据
data = pd.read_csv('clinical_data.csv')

map_stage = {}
stages = sorted(data['stage'].unique().tolist())
for i in range(len(stages)):
    map_stage[stages[i]] = i

# 将阶段映射应用到数据中
data['stage'] = data['stage'].map(map_stage)

# 分离特征和标签
X = data.drop(['id', 'level'], axis=1)
y = data['level']

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义交叉验证策略
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 支持向量机分类器
svm_clf = SVC()

# 进行交叉验证并计算评分
svm_scores = cross_val_score(svm_clf, X, y, cv=kf, scoring='accuracy')
svm_recall_scores = cross_val_score(svm_clf, X, y, cv=kf, scoring='recall_weighted')
svm_f1_scores = cross_val_score(svm_clf, X, y, cv=kf, scoring='f1_weighted')

# 打印支持向量机的交叉验证结果
print("支持向量机分类器 (Cross-Validation):")
print(f"Accuracy: {svm_scores.mean()} ± {svm_scores.std()}")
print(f"Recall: {svm_recall_scores.mean()} ± {svm_recall_scores.std()}")
print(f"F1 Score: {svm_f1_scores.mean()} ± {svm_f1_scores.std()}")

# 随机森林分类器
rf_clf = RandomForestClassifier()

# 进行交叉验证并计算评分
rf_scores = cross_val_score(rf_clf, X, y, cv=kf, scoring='accuracy')
rf_recall_scores = cross_val_score(rf_clf, X, y, cv=kf, scoring='recall_weighted')
rf_f1_scores = cross_val_score(rf_clf, X, y, cv=kf, scoring='f1_weighted')

# 打印随机森林的交叉验证结果
print("\n随机森林分类器 (Cross-Validation):")
print(f"Accuracy: {rf_scores.mean()} ± {rf_scores.std()}")
print(f"Recall: {rf_recall_scores.mean()} ± {rf_recall_scores.std()}")
print(f"F1 Score: {rf_f1_scores.mean()} ± {rf_f1_scores.std()}")





