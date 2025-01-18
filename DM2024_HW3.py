import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('/Users/chenjianheng/Desktop/HW3/未命名檔案夾/新竹_2021.xls')
data.columns = [col.strip() if isinstance(col, str) else col for col in data.columns]

data['測站'] = data['測站'].astype(str).str.strip()
data['測項'] = data['測項'].astype(str).str.strip()

invalid_values = ['#', '*', 'x', 'A', '']

data.iloc[:, 3:] = data.iloc[:, 3:].applymap(lambda x: x.strip() if isinstance(x, str) else x).applymap(lambda x: ''.join(c for c in x if c.isprintable()) if isinstance(x, str) else x)

# 無效值換NaN
data.replace(invalid_values, np.nan, inplace=True)

# 日期格式
data['日期'] = pd.to_datetime(data['日期'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

# 過濾月份
train_data = data[data['日期'].dt.month.isin([10, 11])]
test_data = data[data['日期'].dt.month == 12]

train_data.replace('NR', 0, inplace=True)
test_data.replace('NR', 0, inplace=True)

train_melted = pd.melt(train_data, id_vars=['測站', '日期', '測項'], var_name='小時', value_name='數值')
train_pivoted = train_melted.pivot_table(index=['測站', '日期', '小時'], columns='測項', values='數值')
train_pivoted.reset_index(inplace=True)
train_data = train_pivoted

test_melted = pd.melt(test_data, id_vars=['測站', '日期', '測項'], var_name='小時', value_name='數值')
test_pivoted = test_melted.pivot_table(index=['測站', '日期', '小時'], columns='測項', values='數值')
test_pivoted.reset_index(inplace=True)
test_data = test_pivoted

# 填補缺失值
def fill_missing_with_avg(series):

    filled_series = series.fillna(method='ffill').fillna(method='bfill')
    for i in range(len(filled_series)):
        if pd.isnull(series[i]):
            prev_value = filled_series[i - 1] if i > 0 else np.nan
            next_value = filled_series[i + 1] if i < len(filled_series) - 1 else np.nan

            if pd.notnull(prev_value) and pd.notnull(next_value):
                series[i] = round(np.mean([prev_value, next_value]), 2)

    return series

train_data.iloc[:, 3:] = train_data.iloc[:, 3:].apply(fill_missing_with_avg)
test_data.iloc[:, 3:] = test_data.iloc[:, 3:].apply(fill_missing_with_avg)

# train_data.to_csv('train_data.csv')
# test_data.to_csv('test_data.csv')

# 定義特徵
def create_time_series_features(data, target_hour, feature_hours=6, feature_cols=['PM2.5']):
    X1, y1, X6, y6 = [], [], [], []
    data = data[feature_cols] if target_hour == 1 else data.iloc[:, 3:]

    for i in range(len(data) - 6):
        X1.append(data.iloc[i:i + feature_hours].values.flatten())
        y1.append(data.iloc[i + 6]['PM2.5'])
    for i in range(len(data) - 11):
        X6.append(data.iloc[i:i + feature_hours].values.flatten())
        y6.append(data.iloc[i + 11]['PM2.5'])

    return np.array(X1), np.array(y1), np.array(X6), np.array(y6)

# 創建訓練集和測試集
train_pm25_X1, train_pm25_y1, train_pm25_X6, train_pm25_y6 = create_time_series_features(train_data, target_hour=1)
test_pm25_X1, test_pm25_y1, test_pm25_X6, test_pm25_y6 = create_time_series_features(test_data, target_hour=1)

train_all_X1, train_all_y1, train_all_X6, train_all_y6 = create_time_series_features(train_data, target_hour=0)
test_all_X1, test_all_y1, test_all_X6, test_all_y6 = create_time_series_features(test_data, target_hour=0)

models = {
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(objective='reg:squarederror', eval_metric='mae')
}

results = {}


train_pm25_X1, train_pm25_y1, train_pm25_X6, train_pm25_y6 = create_time_series_features(train_data, target_hour=1)
test_pm25_X1, test_pm25_y1, test_pm25_X6, test_pm25_y6 = create_time_series_features(test_data, target_hour=1)

train_all_X1, train_all_y1, train_all_X6, train_all_y6 = create_time_series_features(train_data, target_hour=0)
test_all_X1, test_all_y1, test_all_X6, test_all_y6 = create_time_series_features(test_data, target_hour=0)

#訓練及評估模型
for model_name, model in models.items():

    model.fit(train_pm25_X1, train_pm25_y1)
    pred_y = model.predict(test_pm25_X1)
    mae = mean_absolute_error(test_pm25_y1, pred_y)
    results[f'{model_name}_PM2.5_預測未來第1個小時'] = mae

    model.fit(train_pm25_X6, train_pm25_y6)
    pred_y = model.predict(test_pm25_X6)
    mae = mean_absolute_error(test_pm25_y6, pred_y)
    results[f'{model_name}_PM2.5_預測未來第6個小時'] = mae

    model.fit(train_all_X1, train_all_y1)
    pred_y = model.predict(test_all_X1)
    mae = mean_absolute_error(test_all_y1, pred_y)
    results[f'{model_name}_全屬性_預測未來第1個小時'] = mae

    model.fit(train_all_X6, train_all_y6)
    pred_y = model.predict(test_all_X6)
    mae = mean_absolute_error(test_all_y6, pred_y)
    results[f'{model_name}_全屬性_預測未來第6個小時'] = mae

#顯示結果
for key, value in results.items():
    print(f"{key} MAE: {value:.4f}")