# --------------------------
# 任务1：数据预处理（终极修复版）
# 问题：文件是 逗号分隔 , 不是 \t 分隔！
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据 —— 【最关键修复】sep=',' 逗号分隔！
df = pd.read_csv('ICData.csv', sep=',')

# 2. 转换交易时间（现在列名正确了！）
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour

# 3. 构造搭乘站点数，删除异常数据
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()
delete_abnormal_num = (df['ride_stops'] == 0).sum()
df = df[df['ride_stops'] != 0].copy()

# 4. 缺失值处理
missing_info = df.isnull().sum()
df = df.dropna()


# ==========================
# 【任务1检验代码 - 开始】
# ==========================
print("✅ 所有列名（正确应该有10列）：")
print(df.columns.tolist())

print("\n✅ 前5行数据：")
print(df.head())

print("\n✅ 时间与小时列：")
print(df[['交易时间', 'hour']].head())

print("\n✅ 删除的异常行数：", delete_abnormal_num)
print("✅ 剩余异常行数（必须=0）：", (df['ride_stops']==0).sum())

print("\n✅ 缺失值（必须全=0）：")
print(df.isnull().sum())
# ==========================
# 【任务1检验代码 - 结束】
# ==========================