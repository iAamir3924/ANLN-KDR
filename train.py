
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score
import warnings

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 可选的字体名称列表

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MFIC(nn.Module):
    def __init__(self, num_layers, input_size, dropout_rate):
        """初始化多源特征交互学习控制器."""
        super(MFIC, self).__init__()
        self.num_layers = num_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        for i in range(self.num_layers):
            gate = torch.sigmoid(self.gate[i](x))
            non_linear = self.leaky_relu(self.non_linear[i](x))
            linear = self.linear[i](x)
            x = gate * non_linear + (1 - gate) * linear
            x = self.dropout(x)
        return x


class Moldel_mrna(nn.Module):
    def __init__(self):
        super().__init__()

        self.mfic = MFIC(2, 3 * 5, 0.1)

        self.synergy_pm = nn.Sequential(
            nn.Linear(3 * 5, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x, ANLN, KDR):
        y = self.synergy_pm(self.mfic(torch.cat([x, ANLN, KDR], dim=1)))
        return y


class Moldel_proteom(nn.Module):
    def __init__(self):
        super().__init__()

        self.mfic = MFIC(2, 16, 0.1)

        # 协同效应预测模块
        self.synergy_pm = nn.Sequential(
            nn.Linear(16, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x, ANLN, KDR):
        y = self.synergy_pm(self.mfic(torch.cat([x, ANLN, KDR], dim=1)))
        return y


class Moldel_mix_up(nn.Module):
    def __init__(self, in_c):
        super(Moldel_mix_up, self).__init__()

        self.mfic = MFIC(2, in_c, 0.1)

        self.synergy_pm = nn.Sequential(
            nn.Linear(in_c, 56),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(56, 56),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(56, 4)
        )

    def forward(self, x, ANLN, KDR):
        y = self.synergy_pm(self.mfic(torch.cat([x, ANLN, KDR], dim=1)))
        return y



# 转录_MLP
def train_mrna(is_train=False):

    data = pd.read_csv('clinical_data.csv')

    map_stage = {}
    stages = sorted(data['stage'].unique().tolist())
    for i in range(len(stages)):
        map_stage[stages[i]] = i

    data['stage'] = data['stage'].map(map_stage)

    X = data.drop(['id', 'level'], axis=1)
    y = data['level'].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    criterian = nn.CrossEntropyLoss()
    model = Moldel_mrna().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if is_train:
        model.train()
        for ep in range(1500):
            x = X_train[:, 0:4]
            ANLN = X_train[:, 4:9]
            KDR = X_train[:, 9:]
            y_prec = model(x, ANLN, KDR)
            loss = criterian(y_prec, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep % 500 == 0:
                print("epoch {} => loss = {}".format(ep, loss))
    else:
        model.load_state_dict(torch.load('checkpoint/mrna_best.pth', map_location='cpu'))
        model.eval()

    # 使用模型进行预测
    x_test = X_test[:, 0:4]
    ANLN_test = X_test[:, 4:9]
    KDR_test = X_test[:, 9:]
    y_pred = model(x_test, ANLN_test, KDR_test)

    # 将预测值转为numpy数组并取最大值索引作为分类结果
    print(y_pred.shape, y_test.shape)
    y_pred_class = y_pred.detach().cpu().numpy().argmax(axis=1)
    y_test_class = y_test.detach().cpu().numpy()

    metrix('MLP', y_test_class, y_pred_class)

    if is_train:
        torch.save(model.state_dict(), 'checkpoint/mrna_now.pth')


# 翻译_MLP
def train_proteom(is_train=False):
    # 读取数据
    data = pd.read_csv('clinical_data_Proteom.csv')

    map_stage = {}
    stages = sorted(data['stage'].unique().tolist())
    for i in range(len(stages)):
        map_stage[stages[i]] = i

    # 将阶段映射应用到数据中
    data['stage'] = data['stage'].map(map_stage)

    # 分离特征和标签
    X = data.drop(['id', 'level'], axis=1)
    y = data['level'].to_numpy()

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    criterian = nn.CrossEntropyLoss()
    model = Moldel_proteom().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if is_train:
        model.train()
        for ep in range(5000):
            x = X_train[:, 0:3]
            ANLN = X_train[:, 9:]
            KDR = X_train[:, 3:9]
            y_prec = model(x, ANLN, KDR)
            loss = criterian(y_prec, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep % 500 == 0:
                print("epoch {} => loss = {}".format(ep, loss))
    else:
        model.load_state_dict(torch.load('checkpoint/proteom_best.pth'))
        model.eval()

    # 使用模型进行预测
    x_test = X_test[:, 0:4]
    ANLN_test = X_test[:, 4:9]
    KDR_test = X_test[:, 9:]
    y_pred = model(x_test, ANLN_test, KDR_test)

    # 将预测值转为numpy数组并取最大值索引作为分类结果
    print(y_pred.shape, y_test.shape)
    y_pred_class = y_pred.detach().cpu().numpy().argmax(axis=1)
    y_test_class = y_test.detach().cpu().numpy()

    metrix('MLP', y_test_class, y_pred_class)


    if is_train:
        torch.save(model.state_dict(), 'checkpoint/proteom_now.pth')


# 协同_MLP
def train_mixup(is_train=False):
    # 读取数据
    data = pd.read_csv('clinical_data - mixup.csv')

    map_stage = {}
    stages = sorted(data['stage'].unique().tolist())
    for i in range(len(stages)):
        map_stage[stages[i]] = i

    # 将阶段映射应用到数据中
    data['stage'] = data['stage'].map(map_stage)

    # 分离特征和标签
    X = data.drop(['id', 'level'], axis=1)
    y = data['level'].to_numpy()

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=45)  # 42

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    criterian = nn.CrossEntropyLoss()
    model = Moldel_mix_up(in_c=28).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if is_train:
        model.train()
        for ep in range(10000):
            x = X_train[:, 0:3]
            ANLN = X_train[:, 3:16]  # 16
            KDR = X_train[:, 16:]
            y_prec = model(x, ANLN, KDR)
            loss = criterian(y_prec, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep % 500 == 0:
                print("epoch {} => loss = {}".format(ep, loss))
    else:
        model.load_state_dict(torch.load('checkpoint/mixup_best.pth'))
        #model.eval()

    # 使用模型进行预测
    x_test = X_test[:, 0:3]
    ANLN_test = X_test[:, 3:16]
    KDR_test = X_test[:, 16:]
    y_pred = model(x_test, ANLN_test, KDR_test)

    # 将预测值转为numpy数组并取最大值索引作为分类结果
    print(y_pred.shape, y_test.shape)
    y_pred_class = y_pred.detach().cpu().numpy().argmax(axis=1)
    y_test_class = y_test.detach().cpu().numpy()

    metrix('MLP', y_test_class, y_pred_class)


    if is_train:
        torch.save(model.state_dict(), 'checkpoint/mixup_now.pth')


def metrix(name, y_test, y_pred):
    # 计算随机森林的性能指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # 输出结果
    print(f"采用{name}分类模型:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 10})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'{name}', fontsize=12)
    plt.savefig(f'{name}_mrna_confusion_matrix.jpg', format='jpg', dpi=600)


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# 转录_其他模型（三个）
def other_mrna():
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    metrix('SVC', y_test, y_pred_svm)

    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    metrix('Naive Bayes', y_test, y_pred_nb)

    logreg_clf = LogisticRegression()
    logreg_clf.fit(X_train, y_train)
    y_pred_logreg = logreg_clf.predict(X_test)
    metrix('Logistic Regression', y_test, y_pred_logreg)

# 翻译_其他模型（三个）
def other_proteom():
    # 读取数据
    data = pd.read_csv('clinical_data_Proteom.csv')

    map_stage = {}
    stages = sorted(data['stage'].unique().tolist())
    for i in range(len(stages)):
        map_stage[stages[i]] = i

    print(map_stage)

    # 将阶段映射应用到数据中
    data['stage'] = data['stage'].map(map_stage)

    # 分离特征和标签
    X = data.drop(['id', 'level'], axis=1)
    y = data['level'].to_numpy()

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    metrix('SVM', y_test, y_pred_svm)

    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    metrix('Naive Bayes', y_test, y_pred_nb)

    logreg_clf = LogisticRegression()
    logreg_clf.fit(X_train, y_train)
    y_pred_logreg = logreg_clf.predict(X_test)
    metrix('Logistic Regression', y_test, y_pred_logreg)

# 协同_其他模型
def other_mix_up():
    # 读取数据
    data = pd.read_csv('clinical_data - mixup.csv')

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=45)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    metrix('SVC', y_test, y_pred_svm)

    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    metrix('Naive Bayes', y_test, y_pred_nb)

    logreg_clf = LogisticRegression()
    logreg_clf.fit(X_train, y_train)
    y_pred_logreg = logreg_clf.predict(X_test)
    metrix('Logistic Regression', y_test, y_pred_logreg)


if __name__ == '__main__':

    #转录水平预测模型
    train_mrna(is_train=False)
    other_mrna()

    # 翻译水平预测模型
    #train_proteom(is_train=False)
    #other_proteom()

    # 协同预测模型
    #train_mixup(is_train=True)
    #other_mix_up()




