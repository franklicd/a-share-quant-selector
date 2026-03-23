#!/usr/bin/env python3
"""
分析每日回测结果并生成分析报告（持有 15 日版）
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse

def analyze_and_generate_report(csv_file=None, json_file=None):
    # 找到最新的回测结果文件
    results_dir = Path(__file__).parent / 'backtest_results'

    if csv_file is None:
        csv_files = list(results_dir.glob('daily_backtest_readable_*.csv'))
        if not csv_files:
            print("未找到回测结果文件")
            return
        csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    else:
        csv_file = Path(csv_file)

    if json_file is None:
        json_files = list(results_dir.glob('daily_backtest_*.json'))
        if not json_files:
            print("未找到 JSON 文件")
            return
        json_file = max(json_files, key=lambda f: f.stat().st_mtime)
    else:
        json_file = Path(json_file)

    print(f"分析文件：{csv_file.name}")

    # 读取数据
    df = pd.read_csv(csv_file)

    # 读取 JSON 获取配置和每日汇总
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    config = json_data['config']
    daily_summary = json_data.get('daily_summary', [])

    # 数据清洗
    df['涨跌幅'] = df['涨跌幅'].str.replace('%', '').astype(float)
    df['最大涨幅'] = df['最大涨幅'].str.replace('%', '').astype(float)
    df['相似度'] = df['相似度'].str.replace('%', '').astype(float)
    df['最大涨幅天数'] = df['最大涨幅天数'].str.replace('第', '').str.replace('天', '').astype(int)

    # 添加持股日期列（用于计算交易日）
    df['选股日期'] = pd.to_datetime(df['选股日期'])
    df['卖出日期'] = pd.to_datetime(df['卖出日期'])

    # 生成报告
    hold_days = config.get('hold_days', 15)

    report = f"""# B1 碗口反弹策略回测分析报告（持有{hold_days}日版）

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**回测时间段**: {config['start_date']} 至 {config['end_date']}
**选股策略**: B1 完美图形匹配 + 碗口反弹技术指标
**选股频率**: 每个交易日
**持有期**: {hold_days} 天（自然日）
**样本数量**: {len(df)} 只股票

---

## 一、回测概览

### 1.1 核心指标

| 指标 | 数值 |
|------|------|
| 总交易数 | {len(df)} 笔 |
| 盈利股票 | {len(df[df['涨跌幅'] > 0])} 只 ({len(df[df['涨跌幅'] > 0])/len(df)*100:.1f}%) |
| 亏损股票 | {len(df[df['涨跌幅'] <= 0])} 只 ({len(df[df['涨跌幅'] <= 0])/len(df)*100:.1f}%) |
| 平均收益 | **{df['涨跌幅'].mean():+.2f}%** |
| 最佳交易 | {df.loc[df['涨跌幅'].idxmax(), '名称']} ({df.loc[df['涨跌幅'].idxmax(), '代码']}) {df['涨跌幅'].max():+.2f}% |
| 最差交易 | {df.loc[df['涨跌幅'].idxmin(), '名称']} ({df.loc[df['涨跌幅'].idxmin(), '代码']}) {df['涨跌幅'].min():+.2f}% |

"""

    # 如果选股日期间隔是连续的（每日选股），则按月份统计
    df['选股月份'] = df['选股日期'].dt.to_period('M').astype(str)
    monthly_stats = df.groupby('选股月份').agg({
        '代码': 'count',
        '涨跌幅': ['mean', lambda x: (x > 0).sum()],
        '最大涨幅': 'mean'
    }).round(2)

    monthly_stats.columns = ['选股数', '平均收益', '盈利数', '最大涨幅均值']
    monthly_stats['胜率'] = (monthly_stats['盈利数'] / monthly_stats['选股数'] * 100).round(1)

    report += "### 1.2 月度/周度表现\n\n"
    report += "| 期间 | 选股数 | 盈利数 | 胜率 | 平均收益 |\n"
    report += "|------|--------|--------|------|----------|\n"

    for period in monthly_stats.index:
        row = monthly_stats.loc[period]
        report += f"| {period} | {int(row['选股数'])} | {int(row['盈利数'])} | {row['胜率']:.1f}% | {row['平均收益']:+.2f}% |\n"

    best_period = monthly_stats.loc[monthly_stats['平均收益'].idxmax()]
    worst_period = monthly_stats.loc[monthly_stats['平均收益'].idxmin()]

    report += f"""
**最佳期间**: {best_period.name} ({best_period['平均收益']:+.2f}%, 胜率 {best_period['胜率']:.1f}%)
**最差期间**: {worst_period.name} ({worst_period['平均收益']:+.2f}%, 胜率 {worst_period['胜率']:.1f}%)

---

## 二、最大涨幅发生时间分布

### 2.1 时间区间分布

"""

    # 时间区间统计 - 根据持有期调整区间
    max_day = df['最大涨幅天数'].max()
    if max_day <= 15:
        bins = [0, 3, 5, 7, 10, 15, 20]
        labels = ['0-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天']
    else:
        bins = [0, 3, 5, 7, 10, 15, 20, 25, 30]
        labels = ['0-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天', '21-25 天', '26-30 天']

    df['天数区间'] = pd.cut(df['最大涨幅天数'], bins=bins, labels=labels, right=False)
    time_dist = df.groupby('天数区间', observed=True)['代码'].count()

    report += "| 时间区间 | 数量 | 占比 |\n"
    report += "|----------|------|------|\n"
    for interval in labels:
        count = time_dist.get(interval, 0) if interval in time_dist.index else 0
        pct = count / len(df) * 100 if count > 0 else 0
        report += f"| {interval} | {count} | {pct:.1f}% |\n"

    report += f"""
### 2.2 天数统计

| 统计项 | 数值 |
|--------|------|
| 平均天数 | {df['最大涨幅天数'].mean():.1f} 天 |
| 中位天数 | {df['最大涨幅天数'].median():.1f} 天 |
| 最早发生 | 第 {df['最大涨幅天数'].min()} 天 |
| 最晚发生 | 第 {df['最大涨幅天数'].max()} 天 |

**结论**: 最大涨幅最集中出现在 **{time_dist.idxmax() if len(time_dist) > 0 else 'N/A'}** 区间。

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
    profit_dist = df.groupby('盈亏区间', observed=True)['代码'].count()

    order = ['大盈 (>10%)', '中盈 (5-10%)', '小盈 (0-5%)', '小亏 (0-5%)', '中亏 (5-10%)', '大亏 (>10%)']

    report += "| 区间 | 数量 | 占比 |\n"
    report += "|------|------|------|\n"
    for interval in order:
        count = profit_dist.get(interval, 0) if interval in profit_dist.index else 0
        pct = count / len(df) * 100 if count > 0 else 0
        report += f"| {interval} | {count} | {pct:.1f}% |\n"

    win_rate = len(df[df['涨跌幅'] > 0]) / len(df) * 100
    report += f"""
**总体统计**:
- 盈利股票：{len(df[df['涨跌幅'] > 0])} 只 ({win_rate:.1f}%)
- 亏损股票：{len(df[df['涨跌幅'] <= 0])} 只 ({100-win_rate:.1f}%)
- 平均盈亏：{df['涨跌幅'].mean():+.2f}%

---

## 四、排名与盈亏相关性分析

### 4.1 不同排名区间的盈亏表现

"""

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
        report += "- B1 图形匹配的排名高低**不能**有效预测持有期后的涨跌\n"
    elif corr_rank_profit > 0:
        report += f"- 排名与最终盈亏呈**正相关** (r={corr_rank_profit:.3f})，排名越靠前表现越好\n"
    else:
        report += f"- 排名与最终盈亏呈**负相关** (r={corr_rank_profit:.3f})，排名靠后的股票反而表现更好\n"

    report += """
---

## 五、相似度与盈亏相关性分析

### 5.1 不同相似度区间的盈亏表现

"""

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
        report += f"- 相似度与盈亏呈{'弱' if abs(corr_sim_profit) < 0.2 else '中等'}正相关，相似度越高，胜率有上升趋势\n"

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
| 胜率 | {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}% |
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
    high_gain_count = len(df[df['最大涨幅'] > df['涨跌幅'] + 5])
    high_gain_pct = high_gain_count / len(df) * 100

    report += f"""
| 指标 | 数值 |
|------|------|
| 平均最大涨幅 | **{avg_max_gain:.2f}%** |
| 平均最终盈亏 | **{avg_final_return:+.2f}%** |
| 平均差异 | **{gain_diff:.2f}%** |
| 曾达到过比最终盈亏高 5% 以上的股票数 | {high_gain_count} 只 ({high_gain_pct:.1f}%) |

**结论**:
- 大部分股票在持有期内都有过不错的表现
- 但**{high_gain_pct:.1f}%**的股票最后回吐了大部分涨幅
- 平均少赚了 {gain_diff:.2f}%，提示可能需要优化止盈策略

---

## 七、选股分类与盈亏关系

### 7.1 不同分类的表现

"""

    category_map = {
        'bowl_center': '回落碗中',
        'near_duokong': '靠近多空线',
        'near_short_trend': '靠近短期趋势线'
    }

    df['分类名称'] = df['类别'].map(category_map)
    df['分类名称'] = df['分类名称'].fillna(df['类别'])

    type_stats = df.groupby('分类名称').agg({
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

    if len(type_stats) > 0:
        best_type = type_stats.index[0]
        report += f"""
**结论**: "{best_type}"分类表现最佳，胜率 {type_stats.loc[best_type, '胜率']:.1f}%，平均盈亏 {type_stats.loc[best_type, '平均盈亏']:+.2f}%。

"""
    else:
        best_type = "N/A"
        report += "\n**结论**: 数据不足，无法分析分类表现。\n\n"

    report += """
---

## 八、关键发现总结

"""

    peak_time = time_dist.idxmax() if len(time_dist) > 0 else "N/A"

    report += f"""
### 8.1 时间分布特征
- 最大涨幅最集中出现在 **{peak_time}** 区间
- 中位时间为 **{df['最大涨幅天数'].median():.0f} 天**，平均 **{df['最大涨幅天数'].mean():.1f} 天**

### 8.2 盈亏分布特征
- 策略整体**{"盈利" if df['涨跌幅'].mean() > 0 else "亏损"}**，胜率 **{win_rate:.1f}%**
- 平均收益 **{df['涨跌幅'].mean():+.2f}%**
- 大盈比例：{profit_dist.get('大盈 (>10%)', 0)/len(df)*100:.1f}%，大亏比例：{profit_dist.get('大亏 (>10%)', 0)/len(df)*100:.1f}%

### 8.3 排名相关性
- 排名与最终盈亏的相关系数为 **{corr_rank_profit:.4f}**
- {"B1 图形匹配的排名不能完全预测持有期后的涨跌" if abs(corr_rank_profit) < 0.1 else "排名对最终盈亏有一定预测作用"}

### 8.4 相似度相关性
- 相似度与盈亏的相关系数为 **{corr_sim_profit:+.4f}**
"""

    if len(high_sim_df) > 0:
        report += f"- **92% 以上相似度**股票：{len(high_sim_df)} 只，胜率 {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}%，平均收益 {high_sim_df['涨跌幅'].mean():+.2f}%\n"
    else:
        report += "- 无 92% 以上相似度的股票样本\n"

    report += f"""
### 8.5 最大涨幅启示
- 平均最大涨幅 ({avg_max_gain:.2f}%) {"高于" if avg_max_gain > avg_final_return else "低于"}最终盈亏 ({avg_final_return:+.2f}%)
- **{high_gain_pct:.1f}%** 的股票曾达到过比最终盈亏高 5% 以上的涨幅
- 说明持有期内大部分股票都有过不错的表现，但最后回吐了涨幅
- **建议**: 考虑引入动态止盈策略（如达到 8-10% 涨幅时提前止盈）

### 8.6 分类策略建议
"""

    if len(type_stats) > 0:
        report += f"- '{best_type}'分类表现最佳，建议可优先选择该分类的股票\n"
    else:
        report += "- 数据不足，无法提供分类建议\n"

    report += """
---

## 九、策略优化建议

基于以上分析，提出以下优化建议：

"""

    if len(high_sim_df) > 0 and corr_sim_profit > 0.1:
        report += f"1. **提高相似度阈值**: 将选股相似度阈值设为 92% 以上，可提高胜率至 {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}%\n\n"
    elif abs(corr_sim_profit) < 0.1:
        report += "1. **相似度参考作用有限**: 相似度与盈亏相关性较弱，不建议仅基于相似度筛选\n\n"
    else:
        report += f"1. **相似度参考**: 相似度与盈亏呈正相关 (r={corr_sim_profit:.3f})，可作为辅助参考\n\n"

    if len(type_stats) > 0:
        report += f"2. **优先选择'{best_type}'分类**: 该分类胜率 {type_stats.loc[best_type, '胜率']:.1f}%，平均盈亏 {type_stats.loc[best_type, '平均盈亏']:+.2f}%\n\n"

    median_day = int(df['最大涨幅天数'].median())
    report += f"""3. **引入动态止盈**:
   - {high_gain_pct:.1f}% 的股票曾达到过比最终盈亏高 5% 以上的涨幅
   - 建议在持有期第 {median_day}-{min(median_day+5, hold_days)} 天设置止盈点（如 8-10%）

4. **持有期调整**:
   - 最大涨幅中位数为 **{df['最大涨幅天数'].median():.0f} 天**
   - 当前持有期 {hold_days} 天，可考虑调整为 {median_day}-{median_day+5} 天配合止盈策略

5. **排名权重**:
   - {"排名与盈亏无显著相关性，可考虑降低排名权重" if abs(corr_rank_profit) < 0.1 else "排名与盈亏有一定相关性，可保持现有权重"}

---

## 十、数据说明

- **天数计算方式**: 自然日（日历日期差），非交易日
- **数据来源**: 本地缓存的股票历史数据（CSV 格式）
- **回测方法**: 每个交易日用截至当日的历史数据重新跑选股，避免未来函数
- **选股数量**: 每日选前 10 只股票
- **持有期**: {hold_days} 自然日

---

**报告完**

*生成脚本：analyze_daily_backtest_hold15.py*
*回测脚本：daily_backtest.py*
*数据文件：{csv_file.name}*
"""

    # 保存报告
    start_date = config['start_date'].replace('-', '')
    end_date = config['end_date'].replace('-', '')
    report_file = results_dir / f"B1 每日回测分析报告_持有{hold_days}日_{config['start_date']}_{config['end_date']}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已生成：{report_file}")
    print(f"\n核心指标:")
    print(f"  - 总交易数：{len(df)}")
    print(f"  - 胜率：{win_rate:.1f}%")
    print(f"  - 平均收益：{df['涨跌幅'].mean():+.2f}%")
    print(f"  - 最佳交易：{df.loc[df['涨跌幅'].idxmax(), '名称']} +{df['涨跌幅'].max():+.2f}%")

    return report_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析每日回测结果并生成报告')
    parser.add_argument('--csv', type=str, help='CSV 文件路径（默认使用最新文件）')
    parser.add_argument('--json', type=str, help='JSON 文件路径（默认使用最新文件）')

    args = parser.parse_args()
    analyze_and_generate_report(args.csv, args.json)
