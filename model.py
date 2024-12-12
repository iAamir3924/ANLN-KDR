
"""

（1）读取CSV文件并加载数据。
（2）分离特征(X)和标签(y)。
（3）将数据集拆分为训练集和测试集。
（4）对特征进行标准化处理。
（5）初始化并训练支持向量机分类器（SVM）。
（6）初始化并训练随机森林分类器（Random Forest）。
（7）计算并打印支持向量机分类器的准确率、召回率、F1-score以及分类报告。
（8）计算并打印随机森林分类器的准确率、召回率、F1-score以及分类报告。

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

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

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 支持向量机分类器
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

# 随机森林分类器
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# 计算支持向量机的性能指标
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
classification_report_svm = classification_report(y_test, y_pred_svm)

# 计算随机森林的性能指标
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
classification_report_rf = classification_report(y_test, y_pred_rf)

# 输出结果
print("采用支持向量机分类模型:")
print(f"Accuracy: {accuracy_svm}")
print(f"Recall: {recall_svm}")
print(f"F1 Score: {f1_svm}")
print("Classification Report:\n", classification_report_svm)

print("\n采用随机森林分类模型:")
print(f"Accuracy: {accuracy_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_rf}")
print("Classification Report:\n", classification_report_rf)


