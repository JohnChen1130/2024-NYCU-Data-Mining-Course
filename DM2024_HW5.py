import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. 讀取數據
file_path = 'movieRating.csv'  # 請更改為您的文件路徑
data = pd.read_csv(file_path)

# 移除不必要的列
data = data.drop(columns=["TrainDataID"])

# 檢查 UserID 和 MovieID 的範圍
print(f"最大 UserID: {data['UserID'].max()}, 最大 MovieID: {data['MovieID'].max()}")

# 打亂數據
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 分割數據集 (80% 訓練集, 20% 測試集)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 2. 獲取 UserID 和 MovieID 的唯一值數量
num_users = data['UserID'].max()  # 注意這裡改用最大值
num_movies = data['MovieID'].max()

# 嵌入維度
embedding_dim = 50

# 3. 建立模型
# Input 層
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

# Embedding 層
user_embedding = Embedding(input_dim=num_users + 1, output_dim=embedding_dim)(user_input)
movie_embedding = Embedding(input_dim=num_movies + 1, output_dim=embedding_dim)(movie_input)

# Flatten 層
user_flatten = Flatten()(user_embedding)
movie_flatten = Flatten()(movie_embedding)

# Dot 產品計算
dot_product = Dot(axes=1)([user_flatten, movie_flatten])

# Dense 層作為輸出
output = Dense(1, activation='linear')(dot_product)

# 定義模型
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 4. 訓練模型
X_train = [train_data['UserID'].values, train_data['MovieID'].values]
y_train = train_data['Rating'].values

X_test = [test_data['UserID'].values, test_data['MovieID'].values]
y_test = test_data['Rating'].values

# 訓練
model.fit(X_train, y_train, batch_size=256, epochs=5, verbose=1)

# 5. 預測與評估
y_pred = model.predict(X_test).flatten()

# 計算 MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
