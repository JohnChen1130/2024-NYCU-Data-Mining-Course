import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
from sklearn import tree
import matplotlib.pyplot as plt


deaths = pd.read_csv('/Users/chenjianheng/Desktop/Python/DM-HW1/character-deaths.csv')
predictions = pd.read_csv('/Users/chenjianheng/Desktop/Python/DM-HW1/character-predictions.csv')

#合併資料集
data = pd.merge(deaths, predictions, left_on='Name', right_on='name')

#修改 'Death' 標籤生成邏輯
data['Death'] = data[['Death Year', 'Book of Death', 'Death Chapter']].apply(lambda row: 1 if any(row.notna()) else 0, axis=1)

#檢查 Death = 0 的樣本數量
print("Death = 0 的樣本數量：", len(data[data['Death'] == 0]))

# 如果沒有 Death = 0 的樣本，手動增加虛擬樣本
if len(data[data['Death'] == 0]) == 0:
    print("資料集中沒有 Death = 0 的樣本，添加虛擬樣本。")
    virtual_samples = pd.DataFrame({
       'Name': ['Virtual Character 1', 'Virtual Character 2'],
       'Allegiances': ['House Stark', 'House Lannister'],
       'Death Year': [None, None],
       'Book of Death': [None, None],
       'Death Chapter': [None, None],
       'Gender': [1, 0],
       'Nobility': [1, 0],
       'GoT': [1, 0],
       'CoK': [1, 0],
       'SoS': [0, 1],
       'FfC': [1, 0],
       'DwD': [0, 1],
       'Death': [0, 0]
    })
    data = pd.concat([data, virtual_samples], ignore_index=True)

#分割資料為 X 和 y
X = data.drop('Death', axis=1)
y = data['Death']

#處理類別型資料
X = pd.get_dummies(X)

#分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#訓練決策樹模型
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

#預測
y_pred = clf.predict(X_test)

#計算評估指標
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')


plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True)
plt.show()

