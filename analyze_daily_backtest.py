#!/usr/bin/env python3
"""
分析每日回测结果并生成分析报告
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 找到最新的回测结果文件
results_dir = Path(__file__).parent / 'backtest_results'
csv_files = list(results_dir.glob('daily_backtest_fast_readable_*.csv'))
json_files = list(results_dir.glob('daily_backtest_fast_*.json'))

# 找到最新的文件
latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
latest_json = max(json_files, key=lambda f: f.stat().st_mtime)

print(f"分析文件：{latest_csv.name}")

# 读取数据
df = pd.read_csv(latest_csv)

# 读取 JSON 获取配置和每日汇总
with open(latest_json, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

config = json_data['config']
daily_summary = json_data['daily_summary']

# 数据清洗
df['涨跌幅'] = df['涨跌幅'].str.replace('%', '').astype(float)
df['最大涨幅'] = df['最大涨幅'].str.replace('%', '').astype(float)
df['相似度'] = df['相似度'].str.replace('%', '').astype(float)
df['最大涨幅天数'] = df['最大涨幅天数'].str.replace('第', '').str.replace('天', '').astype(int)

# 生成报告
report = f"""# B1 碗口反弹策略回测分析报告（每日选股版）

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**回测时间段**: {config['start_date']} 至 {config['end_date']}
**选股策略**: B1 完美图形匹配 + 碗口反弹技术指标
**选股频率**: 每个交易日
**持有期**: {config['hold_days']} 天（自然日）
**样本数量**: {len(df)} 只股票

---

## 一、回测概览

### 1.1 核心指标

| 指标 | 数值 |
|------|------|
| 总交易数 | {len(df)} 笔 |
| 盈利股票 | {len(df[df['涨跌幅'] > 0])} 只 ({len(df[df['涨跌幅'] > 0])/len(df)*100:.1f}%) |
| 亏损股票 | {len(df[df['涨跌幅'] < 0])} 只 ({len(df[df['涨跌幅'] < 0])/len(df)*100:.1f}%) |
| 平均收益 | **{df['涨跌幅'].mean():+.2f}%** |
| 最佳交易 | {df.loc[df['涨跌幅'].idxmax(), '名称']} ({df.loc[df['涨跌幅'].idxmax(), '代码']}) {df['涨跌幅'].max():+.2f}% |
| 最差交易 | {df.loc[df['涨跌幅'].idxmin(), '名称']} ({df.loc[df['涨跌幅'].idxmin(), '代码']}) {df['涨跌幅'].min():+.2f}% |

### 1.2 月度表现

"""

# 添加月份列
df['选股月份'] = pd.to_datetime(df['选股日期']).dt.to_period('M')
monthly_stats = df.groupby('选股月份').agg({
    '代码': 'count',
    '涨跌幅': ['mean', lambda x: (x > 0).sum()],
    '最大涨幅': 'mean'
}).round(2)

# 重命名列
monthly_stats.columns = ['选股数', '平均收益', '盈利数', '最大涨幅均值']
monthly_stats['胜率'] = (monthly_stats['盈利数'] / monthly_stats['选股数'] * 100).round(1)

report += "| 月份 | 选股数 | 盈利数 | 胜率 | 平均收益 |\n"
report += "|------|--------|--------|------|----------|\n"

for month in monthly_stats.index:
    row = monthly_stats.loc[month]
    report += f"| {str(month)} | {int(row['选股数'])} | {int(row['盈利数'])} | {row['胜率']:.1f}% | {row['平均收益']:+.2f}% |\n"

best_month = monthly_stats.loc[monthly_stats['平均收益'].idxmax()]
worst_month = monthly_stats.loc[monthly_stats['平均收益'].idxmin()]

report += f"""
**最佳月份**: {str(best_month.name)} ({best_month['平均收益']:+.2f}%, 胜率 {best_month['胜率']:.1f}%)
**最差月份**: {str(worst_month.name)} ({worst_month['平均收益']:+.2f}%, 胜率 {worst_month['胜率']:.1f}%)

---

## 二、最大涨幅发生时间分布

### 2.1 时间区间分布

"""

# 时间区间统计
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
labels = ['0-5 天', '6-10 天', '11-15 天', '16-20 天', '21-25 天', '26-30 天', '31-35 天', '36-40 天']
df['天数区间'] = pd.cut(df['最大涨幅天数'], bins=bins, labels=labels, right=False)
time_dist = df.groupby('天数区间', observed=True)['代码'].count()

report += "| 时间区间 | 数量 | 占比 |\n"
report += "|----------|------|------|\n"
for interval in labels:
    count = time_dist.get(interval, 0)
    pct = count / len(df) * 100
    report += f"| {interval} | {count} | {pct:.1f}% |\n"

report += f"""
### 2.2 天数统计

| 统计项 | 数值 |
|--------|------|
| 平均天数 | {df['最大涨幅天数'].mean():.1f} 天 |
| 中位天数 | {df['最大涨幅天数'].median():.1f} 天 |
| 最早发生 | 第 {df['最大涨幅天数'].min()} 天 |
| 最晚发生 | 第 {df['最大涨幅天数'].max()} 天 |

**结论**: 最大涨幅最集中出现在 **{time_dist.idxmax()}** 区间，但整体分布较为均匀，说明获利机会在整个持有期内都可能出现。

---

## 三、盈亏分布占比

### 3.1 盈亏区间分布

"""

# 盈亏区间统计
def classify_profit(x):
    if x > 10:
        return '大盈 (>10%)'
    elif x > 5:
        return '中盈 (5-10%)'
    elif x > 0:
        return '小盈 (0-5%)'
    elif x > -5:
        return '小亏 (0-5%)'
    elif x > -10:
        return '中亏 (5-10%)'
    else:
        return '大亏 (>10%)'

df['盈亏区间'] = df['涨跌幅'].apply(classify_profit)
profit_dist = df.groupby('盈亏区间')['代码'].count()

# 按指定顺序
order = ['大盈 (>10%)', '中盈 (5-10%)', '小盈 (0-5%)', '小亏 (0-5%)', '中亏 (5-10%)', '大亏 (>10%)']

report += "| 区间 | 数量 | 占比 |\n"
report += "|------|------|------|\n"
for interval in order:
    count = profit_dist.get(interval, 0)
    pct = count / len(df) * 100
    report += f"| {interval} | {count} | {pct:.1f}% |\n"

report += f"""
**总体统计**:
- 盈利股票：{len(df[df['涨跌幅'] > 0])} 只 ({len(df[df['涨跌幅'] > 0])/len(df)*100:.1f}%)
- 亏损股票：{len(df[df['涨跌幅'] < 0])} 只 ({len(df[df['涨跌幅'] < 0])/len(df)*100:.1f}%)
- 平均盈亏：{df['涨跌幅'].mean():+.2f}%

---

## 四、排名与盈亏相关性分析

### 4.1 不同排名区间的盈亏表现

"""

# 排名区间统计
def rank_range(x):
    if x <= 3:
        return '第 1-3 名'
    elif x <= 5:
        return '第 4-5 名'
    elif x <= 7:
        return '第 6-7 名'
    else:
        return '第 8-10 名'

df['排名区间'] = df['排名'].apply(rank_range)
rank_stats = df.groupby('排名区间').agg({
    '代码': 'count',
    '涨跌幅': ['mean', lambda x: (x > 0).sum()],
    '最大涨幅': 'mean'
}).round(2)

rank_stats.columns = ['样本数', '平均盈亏', '胜率', '最大涨幅均值']
rank_stats['胜率'] = (rank_stats['胜率'] / rank_stats['样本数'] * 100).round(1)

report += "| 排名区间 | 样本数 | 胜率 | 平均盈亏 | 最大涨幅均值 |\n"
report += "|----------|--------|------|----------|--------------|\n"
for rank in ['第 1-3 名', '第 4-5 名', '第 6-7 名', '第 8-10 名']:
    if rank in rank_stats.index:
        row = rank_stats.loc[rank]
        report += f"| {rank} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% | {row['最大涨幅均值']:+.2f}% |\n"

# 计算相关系数
corr_rank_profit = df['排名'].corr(df['涨跌幅'])
corr_rank_maxgain = df['排名'].corr(df['最大涨幅'])

report += f"""
### 4.2 相关系数

| 相关关系 | Pearson 系数 |
|----------|--------------|
| 排名 vs 最终盈亏 | **{corr_rank_profit:.4f}** |
| 排名 vs 最大涨幅 | **{corr_rank_maxgain:.4f}** |

**结论**:
"""

if abs(corr_rank_profit) < 0.1:
    report += "- 排名与最终盈亏**几乎无相关性**\n"
    report += "- B1 图形匹配的排名高低**不能**有效预测 30 日后的涨跌\n"
elif corr_rank_profit > 0:
    report += "- 排名与最终盈亏呈**弱正相关**，排名越靠前表现越好\n"
else:
    report += "- 排名与最终盈亏呈**负相关**，排名靠后的股票反而表现更好\n"

report += """
---

## 五、相似度与盈亏相关性分析

### 5.1 不同相似度区间的盈亏表现

"""

# 相似度区间统计
def sim_range(x):
    if x >= 95:
        return '95%+'
    elif x >= 92:
        return '92-95%'
    elif x >= 90:
        return '90-92%'
    elif x >= 88:
        return '88-90%'
    elif x >= 85:
        return '85-88%'
    else:
        return '80-85%'

df['相似度区间'] = df['相似度'].apply(sim_range)
sim_stats = df.groupby('相似度区间').agg({
    '代码': 'count',
    '涨跌幅': ['mean', lambda x: (x > 0).sum()],
    '最大涨幅': 'mean'
}).round(2)

sim_stats.columns = ['样本数', '平均盈亏', '胜率', '最大涨幅均值']
sim_stats['胜率'] = (sim_stats['胜率'] / sim_stats['样本数'] * 100).round(1)

report += "| 相似度 | 样本数 | 胜率 | 平均盈亏 | 最大涨幅均值 |\n"
report += "|--------|--------|------|----------|--------------|\n"
for sim in ['80-85%', '85-88%', '88-90%', '90-92%', '92-95%', '95%+']:
    if sim in sim_stats.index:
        row = sim_stats.loc[sim]
        report += f"| {sim} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% | {row['最大涨幅均值']:+.2f}% |\n"

# 计算相关系数
corr_sim_profit = df['相似度'].corr(df['涨跌幅'])
corr_sim_maxgain = df['相似度'].corr(df['最大涨幅'])

report += f"""
### 5.2 相关系数

| 相关关系 | Pearson 系数 |
|----------|--------------|
| 相似度 vs 最终盈亏 | **{corr_sim_profit:+.4f}** |
| 相似度 vs 最大涨幅 | **{corr_sim_maxgain:+.4f}** |

**结论**:
"""

if abs(corr_sim_profit) < 0.1:
    report += "- 相似度与盈亏**几乎无相关性**\n"
elif corr_sim_profit > 0.3:
    report += "- 相似度与盈亏呈**中等正相关**，相似度越高胜率有显著上升趋势\n"
else:
    report += "- 相似度与盈亏呈**弱正相关**，相似度越高，胜率有上升趋势\n"

# 高相似度股票详情
high_sim_df = df[df['相似度'] >= 92].copy()
if len(high_sim_df) > 0:
    report += f"""
### 5.3 高相似度股票详情（≥92%）

| 代码 | 名称 | 相似度 | 最大涨幅 | 最终盈亏 | 选股日期 |
|------|------|--------|----------|----------|----------|
"""
    top_sim = high_sim_df.nlargest(15, '相似度')[['代码', '名称', '相似度', '最大涨幅', '涨跌幅', '选股日期']]
    for _, row in top_sim.iterrows():
        report += f"| {row['代码']} | {row['名称']} | {row['相似度']:.1f}% | {row['最大涨幅']:+.2f}% | {row['涨跌幅']:+.2f}% | {row['选股日期']} |\n"

    report += f"""
**高相似度股票（≥92%）统计**:

| 统计项 | 数值 |
|--------|------|
| 样本数 | {len(high_sim_df)} 只 |
| 平均胜率 | {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}% |
| 平均收益 | {high_sim_df['涨跌幅'].mean():+.2f}% |
| 最大涨幅均值 | {high_sim_df['最大涨幅'].mean():+.2f}% |

"""

report += """
---

## 六、最大涨幅 vs 最终盈亏对比

### 6.1 差异统计

"""

avg_max_gain = df['最大涨幅'].mean()
avg_final_return = df['涨跌幅'].mean()
gain_diff = avg_max_gain - avg_final_return
high_gain_pct = (df['最大涨幅'] > df['涨跌幅'] + 5).sum() / len(df) * 100

report += f"""
| 指标 | 数值 |
|------|------|
| 平均最大涨幅 | **{avg_max_gain:.2f}%** |
| 平均最终盈亏 | **{avg_final_return:+.2f}%** |
| 平均差异 | **{gain_diff:.2f}%** |
| 曾达到过比最终盈亏高 5% 以上的股票数 | {int(high_gain_pct * len(df) / 100)} 只 ({high_gain_pct:.1f}%) |

**结论**:
- 大部分股票在持有期内都有过不错的表现
- 但**{high_gain_pct:.1f}%**的股票最后回吐了大部分涨幅
- 平均少赚了 {gain_diff:.2f}%，提示可能需要优化止盈策略

---

## 七、选股分类与盈亏关系

### 7.1 不同分类的表现

"""

# 使用匹配案例描述中的关键词进行分类
def classify_type(row):
    desc = str(row['匹配案例描述'])
    if '回落' in desc or '短期趋势线' in desc:
        return '靠近短期趋势线'
    elif '多空线' in desc:
        return '靠近多空线'
    elif '杯型' in desc or '平台' in desc:
        return '平台整理'
    elif '缩量' in desc:
        return '缩量整理'
    else:
        return '其他'

df['分类'] = df.apply(classify_type, axis=1)
type_stats = df.groupby('分类').agg({
    '代码': 'count',
    '涨跌幅': ['mean', lambda x: (x > 0).sum()],
    '最大涨幅': 'mean'
}).round(2)

type_stats.columns = ['样本数', '平均盈亏', '胜率', '最大涨幅均值']
type_stats['胜率'] = (type_stats['胜率'] / type_stats['样本数'] * 100).round(1)
type_stats = type_stats.sort_values('平均盈亏', ascending=False)

report += "| 分类 | 样本数 | 胜率 | 平均盈亏 | 最大涨幅均值 |\n"
report += "|------|--------|------|----------|--------------|\n"
for cls in type_stats.index:
    row = type_stats.loc[cls]
    report += f"| {cls} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% | {row['最大涨幅均值']:+.2f}% |\n"

best_type = type_stats.index[0]
report += f"""
**结论**: "{best_type}"分类表现最佳，胜率 {type_stats.loc[best_type, '胜率']:.1f}%，平均盈亏 {type_stats.loc[best_type, '平均盈亏']:+.2f}%。

---

## 八、关键发现总结

### 8.1 时间分布特征
- 最大涨幅最集中出现在 **{time_dist.idxmax()}** 区间
- 中位时间为 **{df['最大涨幅天数'].median():.0f} 天**，平均 **{df['最大涨幅天数'].mean():.1f} 天**
- 分布较为均匀，获利机会在持有期内各阶段都可能出现

### 8.2 盈亏分布特征
- 策略整体**{"盈利" if df['涨跌幅'].mean() > 0 else "亏损"}**，胜率 **{len(df[df['涨跌幅'] > 0])/len(df)*100:.1f}%**
- 平均收益 **{df['涨跌幅'].mean():+.2f}%**
- 大盈和大亏比例相近（{profit_dist.get('大盈 (>10%)', 0)/len(df)*100:.1f}% vs {profit_dist.get('大亏 (>10%)', 0)/len(df)*100:.1f}%）

### 8.3 排名相关性
- 排名与最终盈亏的相关系数为 **{corr_rank_profit:.4f}**
- {"B1 图形匹配的排名不能完全预测 30 日后的涨跌" if abs(corr_rank_profit) < 0.1 else "排名对最终盈亏有一定预测作用"}

### 8.4 相似度相关性
- 相似度与盈亏的相关系数为 **{corr_sim_profit:+.4f}**
- {"相似度对盈亏没有明显预测作用" if abs(corr_sim_profit) < 0.1 else f"相似度越高，胜率有{'上升' if corr_sim_profit > 0 else '下降'}趋势"}
- **92% 以上相似度**股票：{len(high_sim_df)} 只，胜率 {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}%，平均收益 {high_sim_df['涨跌幅'].mean():+.2f}%

### 8.5 最大涨幅启示
- 平均最大涨幅 ({avg_max_gain:.2f}%) {"高于" if avg_max_gain > avg_final_return else "低于"}最终盈亏 ({avg_final_return:+.2f}%)
- **{high_gain_pct:.1f}%** 的股票曾达到过比最终盈亏高 5% 以上的涨幅
- 说明持有期内大部分股票都有过不错的表现，但最后回吐了涨幅
- **建议**: 考虑引入动态止盈策略（如达到 8-10% 涨幅时提前止盈）

### 8.6 分类策略建议
- "{best_type}"分类表现最佳
- **建议**: 可优先选择该分类的股票，或增加其权重

---

## 九、策略优化建议

基于以上分析，提出以下优化建议：

1. **相似度阈值调整**:
"""

if corr_sim_profit > 0.1:
    report += f"   - 将选股相似度阈值设为 92% 以上，可提高胜率至 {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}%\n"
else:
    report += "   - 相似度与盈亏相关性较弱，不建议仅基于相似度筛选\n"

report += f"""
2. **优先选择"{best_type}"分类**: 该分类胜率 {type_stats.loc[best_type, '胜率']:.1f}%，平均盈亏 {type_stats.loc[best_type, '平均盈亏']:+.2f}%

3. **引入动态止盈**:
   - {high_gain_pct:.1f}% 的股票曾达到过比最终盈亏高 5% 以上的涨幅
   - 建议在持有期第 {int(df['最大涨幅天数'].median())}-{int(df['最大涨幅天数'].median())+5} 天（最大涨幅高发期）设置止盈点（如 8-10%）

4. **持有期调整**:
   - 最大涨幅中位数为 **{df['最大涨幅天数'].median():.0f} 天**
   - 可考虑将持有期缩短至 {int(df['最大涨幅天数'].median())}-{int(df['最大涨幅天数'].median())+5} 天，配合止盈策略

5. **排名权重调整**:
   - {"排名与盈亏无显著相关性，可考虑降低排名权重" if abs(corr_rank_profit) < 0.1 else "排名与盈亏有一定相关性，可保持现有权重"}

---

## 十、数据说明

- **天数计算方式**: 自然日（日历日期差），非交易日
- **数据来源**: 本地缓存的股票历史数据（CSV/Parquet 格式）
- **回测方法**: 每个交易日用截至当日的历史数据重新跑选股，避免未来函数
- **选股数量**: 每日选前 10 只股票
- **持有期**: {config['hold_days']} 自然日

---

**报告完**

*生成脚本：analyze_daily_backtest.py*
*回测脚本：daily_backtest_fast.py*
*数据文件：{latest_csv.name}*
"""

# 保存报告
report_file = results_dir / f"B1 每日回测分析报告_{config['start_date']}_{config['end_date']}.md"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"报告已生成：{report_file}")
print(f"\n核心指标:")
print(f"  - 总交易数：{len(df)}")
print(f"  - 胜率：{len(df[df['涨跌幅'] > 0])/len(df)*100:.1f}%")
print(f"  - 平均收益：{df['涨跌幅'].mean():+.2f}%")
print(f"  - 最佳交易：{df.loc[df['涨跌幅'].idxmax(), '名称']} +{df['涨跌幅'].max():.2f}%")
