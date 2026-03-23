"""
回测结果分析脚本
分析：
1. 最大涨幅发生的时间分布
2. 30 天后盈亏分布占比
3. 排名与盈亏/最大涨幅的相关性
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 读取回测结果（处理 BOM 编码）
results_file = Path(__file__).parent / 'backtest_results' / 'b1_backtest_readable_20260322_184933.csv'
df = pd.read_csv(results_file, encoding='utf-8-sig')

# 数据清洗函数
def clean_pct(x):
    # 清理百分比数据，去掉 + 号和%
    return float(str(x).replace('+', '').replace('%', ''))

def clean_day(x):
    # 清理天数数据，去掉"第"和"天"
    return int(str(x).replace('第', '').replace('天', ''))

# 数据清洗
df['return_pct'] = df['涨跌幅'].apply(clean_pct)
df['max_gain'] = df['最大涨幅'].apply(clean_pct)
df['max_gain_day'] = df['最大涨幅天数'].apply(clean_day)
df['similarity_val'] = df['相似度'].str.replace('%', '').astype(float)

print("=" * 80)
print("B1 碗口反弹策略 - 13 个月回测深度分析")
print("=" * 80)

# ============================================
# 1. 最大涨幅发生的时间分布分析
# ============================================
print("\n" + "=" * 80)
print("一、最大涨幅发生时间分布分析")
print("=" * 80)

# 时间区间分布
time_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
time_labels = ['0-5 天', '6-10 天', '11-15 天', '16-20 天', '21-25 天', '26-30 天', '31-35 天', '36-40 天']
df['time_period'] = pd.cut(df['max_gain_day'], bins=time_bins, labels=time_labels, right=True)

time_dist = df['time_period'].value_counts().sort_index()
print("\n最大涨幅发生时间区间分布:")
print("-" * 40)
for period, count in time_dist.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {period}: {count:2d} 只 ({pct:5.1f}%) {bar}")

# 计算平均天数
avg_day = df['max_gain_day'].mean()
median_day = df['max_gain_day'].median()
print(f"\n最大涨幅发生天数统计:")
print(f"  平均天数：{avg_day:.1f} 天")
print(f"  中位天数：{median_day:.1f} 天")
print(f"  最早发生：第 {df['max_gain_day'].min()} 天")
print(f"  最晚发生：第 {df['max_gain_day'].max()} 天")

# ============================================
# 2. 30 天后盈亏分布占比
# ============================================
print("\n" + "=" * 80)
print("二、30 天后盈亏分布占比")
print("=" * 80)

# 盈亏分类
df['profit_category'] = pd.cut(
    df['return_pct'],
    bins=[-100, -10, -5, 0, 5, 10, 100],
    labels=['大亏 (>10%)', '中亏 (5-10%)', '小亏 (0-5%)', '小盈 (0-5%)', '中盈 (5-10%)', '大盈 (>10%)']
)

profit_dist = df['profit_category'].value_counts().sort_index()
print("\n盈亏区间分布:")
print("-" * 40)
for category, count in profit_dist.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {category}: {count:2d} 只 ({pct:5.1f}%) {bar}")

# 总体盈亏
win_count = len(df[df['return_pct'] > 0])
lose_count = len(df[df['return_pct'] <= 0])
win_rate = win_count / len(df) * 100
print(f"\n总体盈亏统计:")
print(f"  盈利股票：{win_count} 只 ({win_rate:.1f}%)")
print(f"  亏损股票：{lose_count} 只 ({100-win_rate:.1f}%)")
print(f"  平均盈亏：{df['return_pct'].mean():+.2f}%")

# ============================================
# 3. 排名与盈亏的相关性分析
# ============================================
print("\n" + "=" * 80)
print("三、排名与盈亏/最大涨幅的相关性分析")
print("=" * 80)

# 按排名分组统计
df['rank_group'] = pd.cut(
    df['排名'],
    bins=[0, 3, 5, 7, 10],
    labels=['第 1-3 名', '第 4-5 名', '第 6-7 名', '第 8-10 名']
)

print("\n不同排名区间的盈亏表现:")
print("-" * 70)
print(f"{'排名区间':<12} {'样本数':>8} {'胜率':>10} {'平均盈亏':>12} {'最大涨幅均值':>14}")
print("-" * 70)

for group in df['rank_group'].cat.categories:
    group_df = df[df['rank_group'] == group]
    sample_count = len(group_df)
    group_win_rate = len(group_df[group_df['return_pct'] > 0]) / sample_count * 100
    avg_return = group_df['return_pct'].mean()
    avg_max_gain = group_df['max_gain'].mean()
    print(f"{group:<12} {sample_count:>8} {group_win_rate:>9.1f}% {avg_return:>+11.2f}% {avg_max_gain:>+13.2f}%")

# 计算相关系数
rank_return_corr = df['排名'].corr(df['return_pct'])
rank_maxgain_corr = df['排名'].corr(df['max_gain'])

print(f"\n相关系数分析 (Pearson):")
print(f"  排名 vs 最终盈亏：{rank_return_corr:.4f}")
print(f"  排名 vs 最大涨幅：{rank_maxgain_corr:.4f}")

if rank_return_corr < -0.1:
    print(f"\n  解读：排名与盈亏呈负相关，排名越靠前（数字越小），盈利概率越高")
elif rank_return_corr > 0.1:
    print(f"\n  解读：排名与盈亏呈正相关，排名越靠后（数字越大），盈利概率越高")
else:
    print(f"\n  解读：排名与盈亏相关性较弱，排名对最终盈亏影响不明显")

# ============================================
# 4. 相似度分数与盈亏的相关性
# ============================================
print("\n" + "=" * 80)
print("四、相似度分数与盈亏的相关性分析")
print("=" * 80)

# 按相似度分组
df['similarity_group'] = pd.cut(
    df['similarity_val'],
    bins=[80, 85, 88, 90, 92, 95, 100],
    labels=['80-85%', '85-88%', '88-90%', '90-92%', '92-95%', '95%+']
)

print("\n不同相似度区间的盈亏表现:")
print("-" * 70)
print(f"{'相似度':<12} {'样本数':>8} {'胜率':>10} {'平均盈亏':>12} {'最大涨幅均值':>14}")
print("-" * 70)

for group in df['similarity_group'].cat.categories:
    group_df = df[df['similarity_group'] == group]
    if len(group_df) > 0:
        sample_count = len(group_df)
        group_win_rate = len(group_df[group_df['return_pct'] > 0]) / sample_count * 100
        avg_return = group_df['return_pct'].mean()
        avg_max_gain = group_df['max_gain'].mean()
        print(f"{group:<12} {sample_count:>8} {group_win_rate:>9.1f}% {avg_return:>+11.2f}% {avg_max_gain:>+13.2f}%")

# 计算相关系数
similarity_return_corr = df['similarity_val'].corr(df['return_pct'])
similarity_maxgain_corr = df['similarity_val'].corr(df['max_gain'])

print(f"\n相关系数分析 (Pearson):")
print(f"  相似度 vs 最终盈亏：{similarity_return_corr:.4f}")
print(f"  相似度 vs 最大涨幅：{similarity_maxgain_corr:.4f}")

if similarity_return_corr > 0.1:
    print(f"\n  解读：相似度与盈亏呈正相关，相似度越高，盈利概率越高")
elif similarity_return_corr < -0.1:
    print(f"\n  解读：相似度与盈亏呈负相关，相似度越低，盈利概率越高")
else:
    print(f"\n  解读：相似度与盈亏相关性较弱")

# ============================================
# 5. 最大涨幅 vs 最终盈亏对比
# ============================================
print("\n" + "=" * 80)
print("五、最大涨幅 vs 最终盈亏对比分析")
print("=" * 80)

# 计算最大涨幅与最终盈亏的差异
df['gain_diff'] = df['max_gain'] - df['return_pct']

print(f"\n最大涨幅与最终盈亏差异统计:")
print(f"  平均最大涨幅：{df['max_gain'].mean():.2f}%")
print(f"  平均最终盈亏：{df['return_pct'].mean():+.2f}%")
print(f"  平均差异：{df['gain_diff'].mean():.2f}% (意味着如果不在 30 天卖出，平均少赚这么多)")

# 统计有多少股票曾经达到过更高涨幅
high_gain_count = len(df[df['gain_diff'] > 5])
print(f"  曾达到过比最终盈亏高 5% 以上的股票数：{high_gain_count} 只 ({high_gain_count/len(df)*100:.1f}%)")

# ============================================
# 6. 分类标记与盈亏关系
# ============================================
print("\n" + "=" * 80)
print("六、选股分类与盈亏关系")
print("=" * 80)

print("\n不同选股分类的表现:")
print("-" * 70)
print(f"{'分类':<20} {'样本数':>8} {'胜率':>10} {'平均盈亏':>12} {'最大涨幅均值':>14}")
print("-" * 70)

for category in df['类别'].unique():
    cat_df = df[df['类别'] == category]
    sample_count = len(cat_df)
    win_count = len(cat_df[cat_df['return_pct'] > 0])
    win_rate = win_count / sample_count * 100 if sample_count > 0 else 0
    avg_return = cat_df['return_pct'].mean()
    avg_max_gain = cat_df['max_gain'].mean()
    cat_name = category.replace('bowl_center', '回落碗中').replace('near_duokong', '靠近多空线').replace('near_short_trend', '靠近短期趋势线')
    print(f"{cat_name:<20} {sample_count:>8} {win_rate:>9.1f}% {avg_return:>+11.2f}% {avg_max_gain:>+13.2f}%")

# ============================================
# 7. 关键发现总结
# ============================================
print("\n" + "=" * 80)
print("七、关键发现总结")
print("=" * 80)

print("\n1. 时间分布特征:")
peak_period = time_dist.idxmax()
print(f"   最大涨幅最集中出现在：{peak_period} ({time_dist[peak_period]}只，{time_dist[peak_period]/len(df)*100:.1f}%)")

print("\n2. 盈亏分布特征:")
if win_rate > 50:
    print(f"   策略整体盈利面较好，胜率 {win_rate:.1f}%")
else:
    print(f"   策略整体盈利面一般，胜率 {win_rate:.1f}%")

print("\n3. 排名相关性:")
if abs(rank_return_corr) < 0.1:
    print(f"   排名与最终盈亏几乎无相关性 (r={rank_return_corr:.4f})")
    print(f"   说明 B1 图形匹配的排名不能完全预测 30 日后的涨跌")
elif rank_return_corr < 0:
    print(f"   排名与盈亏呈负相关 (r={rank_return_corr:.4f})")
    print(f"   排名越靠前，盈利概率越高")

print("\n4. 相似度相关性:")
if abs(similarity_return_corr) < 0.1:
    print(f"   相似度与最终盈亏几乎无相关性 (r={similarity_return_corr:.4f})")
elif similarity_return_corr > 0:
    print(f"   相似度与盈亏呈正相关 (r={similarity_return_corr:.4f})")
    print(f"   相似度越高，盈利概率越高")

print("\n5. 最大涨幅启示:")
print(f"   平均最大涨幅 ({df['max_gain'].mean():.2f}%) 远高于最终盈亏 ({df['return_pct'].mean():+.2f}%)")
print(f"   {high_gain_count/len(df)*100:.1f}% 的股票曾达到过比最终盈亏高 5% 以上的涨幅")
print(f"   说明持有期内大部分股票都有过不错的表现，但最后回吐了涨幅")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
