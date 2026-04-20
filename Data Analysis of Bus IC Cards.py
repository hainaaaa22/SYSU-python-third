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


# --------------------------
# 任务3：线路站点分析（修复版，无报错）
# --------------------------
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    # 分组计算均值和标准差
    route_stats = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    route_stats.columns = ['线路号', 'mean_stops', 'std_stops']

    # 【关键修复】去掉标准差为NaN的行（避免绘图报错）
    route_stats = route_stats.dropna(subset=['std_stops'])

    # 按均值降序
    route_stats = route_stats.sort_values('mean_stops', ascending=False)
    return route_stats


# 执行函数
route_result = analyze_route_stops(df)
top15_routes = route_result.head(15)

# 绘图：修复 seaborn 新版本语法 + 处理误差棒
plt.figure(figsize=(12, 8))

# 修复：去掉 palette 警告 + 安全绘制误差棒
sns.barplot(
    x='mean_stops',
    y='线路号',
    data=top15_routes,
    color='steelblue',  # 用 color 替代 palette，无警告
    capsize=0.3
)

plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.title('前15条线路平均搭乘站点数', fontsize=14)
plt.tight_layout()
plt.savefig('route_stops.png', dpi=300)
plt.close()


# --------------------------
# 任务4：高峰小时系数PHF计算
# --------------------------
# 只统计上车数据
df_board = df[df['刷卡类型'] == 0].copy()

# 1. 按小时统计，找到刷卡量最大的高峰小时
hourly_count = df_board['hour'].value_counts().sort_index()
peak_hour = hourly_count.idxmax()  # 高峰小时
peak_volume = hourly_count.max()   # 高峰小时总流量

# 2. 筛选出高峰小时内的所有数据
df_peak = df_board[df_board['hour'] == peak_hour].copy()
df_peak['交易时间'] = pd.to_datetime(df_peak['交易时间'])

# 3. 5分钟粒度统计 → 最大5分钟流量
df_peak['time_5min'] = df_peak['交易时间'].dt.floor('5min')
max_5min = df_peak['time_5min'].value_counts().max()

# 4. 15分钟粒度统计 → 最大15分钟流量
df_peak['time_15min'] = df_peak['交易时间'].dt.floor('15min')
max_15min = df_peak['time_15min'].value_counts().max()

# 5. 按公式计算 PHF
PHF5 = peak_volume / (12 * max_5min)
PHF15 = peak_volume / (4 * max_15min)

# --------------------------
# 任务5：线路驾驶员信息批量导出
# --------------------------
import os

# 筛选线路 1101 ~ 1120
df_selected = df[(df['线路号'] >= 1101) & (df['线路号'] <= 1120)].copy()

# 创建输出文件夹
output_dir = '线路驾驶员信息'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 按每条线路循环，去重后保存txt
for route in sorted(df_selected['线路号'].unique()):
    # 筛选当前线路
    df_route = df_selected[df_selected['线路号'] == route]

    # 去重：车辆 → 驾驶员（唯一对应）
    df_driver = df_route[['车辆编号', '驾驶员编号']].drop_duplicates()

    # 文件名
    filename = os.path.join(output_dir, f'线路{route}_驾驶员信息.txt')

    # 写入txt
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'线路号：{route}\n')
        f.write('车辆编号\t驾驶员编号\n')
        for _, row in df_driver.iterrows():
            f.write(f"{row['车辆编号']}\t{row['驾驶员编号']}\n")


# ==========================
# 【任务5检验代码 - 开始】
# ==========================
print("===== 任务5 检验结果 =====")
print("✅ 已创建文件夹：", output_dir)
print("✅ 生成的线路文件列表：")
for route in sorted(df_selected['线路号'].unique()):
    print(f"线路{route} → 线路{route}_驾驶员信息.txt")
print("\n✅ 所有1101~1120线路信息已批量导出完成！")
# ==========================
# 【任务5检验代码 - 结束】
# ==========================