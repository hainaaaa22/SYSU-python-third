# --------------------------
# 任务1：数据预处理（完整无错版）
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据（逗号分隔）
df = pd.read_csv('ICData.csv', sep=',')

# 2. 转换交易时间
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour

# 3. 构造搭乘站点数，删除异常数据
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()
delete_abnormal_num = (df['ride_stops'] == 0).sum()
df = df[df['ride_stops'] != 0].copy()

# 4. 缺失值处理
missing_info = df.isnull().sum()
df = df.dropna()

# --------------------------
# 任务2：时间分布分析（numpy + matplotlib）
# --------------------------
# 筛选：只统计 上车刷卡（刷卡类型=0）
df_board = df[df['刷卡类型'] == 0].copy()

# 1. 使用 numpy 统计时段
hour_array = np.array(df_board['hour'])
early_count = np.sum(hour_array < 7)       # 早峰前 <7
night_count = np.sum(hour_array >= 22)     # 深夜 >=22
total_count = len(hour_array)

# 计算占比
early_ratio = early_count / total_count * 100
night_ratio = night_count / total_count * 100

# 2. 按小时统计24小时客流量
hour_counts = df_board['hour'].value_counts().sort_index()

# 3. matplotlib 绘制24小时柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(hour_counts.index, hour_counts.values, color='skyblue', alpha=0.7)

# 高亮早峰前 / 深夜
for h in range(24):
    if h < 7 or h >= 22:
        bars[h].set_color('orange')
        bars[h].set_alpha(0.9)

plt.xlabel('小时', fontsize=12)
plt.ylabel('刷卡次数', fontsize=12)
plt.title('公交IC卡24小时刷卡量分布', fontsize=14)
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(['正常时段', '早峰前/深夜时段'])

plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=300)
plt.close()


# ==========================
# 【任务2检验代码 - 开始】
# ==========================
print("===== 任务2 检验结果 =====")
print(f"全天总上车刷卡量：{total_count}")
print(f"早峰前（小时<7）刷卡量：{early_count}，占比：{early_ratio:.2f}%")
print(f"深夜时段（小时≥22）刷卡量：{night_count}，占比：{night_ratio:.2f}%")
print("\n每小时刷卡量统计：")
print(hour_counts)
print("\n✅ 图片已保存：hour_distribution.png")
# ==========================
# 【任务2检验代码 - 结束】
# ==========================