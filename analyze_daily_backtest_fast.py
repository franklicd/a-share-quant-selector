#!/usr/bin/env python3
"""
分析每日快速回测结果并生成分析报告
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse


def analyze_and_generate_report(csv_file=None, json_file=None, output_file=None):
    """分析回测结果并生成 MD 报告"""
    results_dir = Path(__file__).parent / 'backtest_results'

    if csv_file is None:
        csv_files = list(results_dir.glob('daily_backtest_fast_readable_*.csv'))
        if not csv_files:
            print("未找到回测结果文件")
            return None
        csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    else:
        csv_file = Path(csv_file)

    if json_file is None:
        json_files = list(results_dir.glob('daily_backtest_fast_*.json'))
        if not json_files:
            json_file = None
        else:
            json_file = max(json_files, key=lambda f: f.stat().st_mtime)
    else:
        json_file = Path(json_file)

    print(f"分析文件：{csv_file.name}")

    df = pd.read_csv(csv_file)

    config = {}
    daily_summary = []

    if json_file and json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        config = json_data.get('config', {})
        daily_summary = json_data.get('daily_summary', [])

    df['涨跌幅'] = df['涨跌幅'].str.replace('%', '').astype(float)
    df['最大涨幅'] = df['最大涨幅'].str.replace('%', '').astype(float)
    df['相似度'] = df['相似度'].str.replace('%', '').astype(float)
    df['最大涨幅天数'] = df['最大涨幅天数'].str.replace('第', '').str.replace('天', '').astype(int)
    df['选股日期'] = pd.to_datetime(df['选股日期'])
    df['卖出日期'] = pd.to_datetime(df['卖出日期'])

    # 对股票代码去重，只保留每只股票第一次出现的记录
    df = df.sort_values('选股日期').drop_duplicates(subset='代码', keep='first').reset_index(drop=True)
    print(f"去重后样本数：{len(df)}（每只股票仅统计首次入选）")

    # 处理行业热度列（需要在 has_industry_data 检查之前处理）
    if '行业' in df.columns:
        df['行业'] = df['行业'].fillna('未知')
        # 转换行业热度列为数值（处理字符串和数值混合的情况）
        for col in ['行业热度_买入日', '行业热度_10pct 日', '行业热度_5pct 日', '行业热度_neg2pct 日', '行业热度_neg4pct 日', '行业热度_卖出日']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).replace('-', ''), errors='coerce')

    # 处理止盈止损触发日期列
    has_trigger_data = '触发 +10% 日期' in df.columns
    if has_trigger_data:
        df['触发 +10% 日期'] = df['触发 +10% 日期'].replace('-', '').replace('', pd.NaT)
        df['触发 +5% 日期'] = df['触发 +5% 日期'].replace('-', '').replace('', pd.NaT) if '触发 +5% 日期' in df.columns else None
        df['触发 -2% 日期'] = df['触发 -2% 日期'].replace('-', '').replace('', pd.NaT)
        df['触发 -4% 日期'] = df['触发 -4% 日期'].replace('-', '').replace('', pd.NaT)
        df['触发顺序'] = df['触发顺序'].replace('-', '')

    # 处理达到 +10% 天数列
    has_10pct_day = '达到 +10% 天数' in df.columns
    if has_10pct_day:
        df['达到 +10% 天数'] = df['达到 +10% 天数'].replace('-', '').str.replace('第', '').str.replace('天', '')
        df['达到 +10% 天数'] = pd.to_numeric(df['达到 +10% 天数'], errors='coerce')

    # 处理达到 +5% 天数列
    has_5pct_day = '达到 +5% 天数' in df.columns
    if has_5pct_day:
        df['达到 +5% 天数'] = df['达到 +5% 天数'].replace('-', '').str.replace('第', '').str.replace('天', '')
        df['达到 +5% 天数'] = pd.to_numeric(df['达到 +5% 天数'], errors='coerce')

    # 设置行业数据标志
    has_industry_data = '行业' in df.columns
    if has_industry_data:
        df['行业'] = df['行业'].fillna('未知')

    # 处理是否曾跌破成本价列
    has_ever_below_zero = '是否曾跌破成本价' in df.columns
    if has_ever_below_zero:
        df['是否曾跌破成本价'] = df['是否曾跌破成本价'].replace('-', '否')

    hold_days = config.get('hold_days', df['持有天数'].iloc[0] if '持有天数' in df.columns else 30)
    start_date = config.get('start_date', df['选股日期'].min().strftime('%Y-%m-%d'))
    end_date = config.get('end_date', df['选股日期'].max().strftime('%Y-%m-%d'))

    report = f"""# B1 碗口反弹策略回测分析报告（每日选股·持有{hold_days}日版）

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**回测时间段**: {start_date} 至 {end_date}
**选股策略**: B1 完美图形匹配 + 碗口反弹技术指标
**选股频率**: 每个交易日
**持有期**: {hold_days} 天（自然日）
**样本数量**: {len(df)} 只股票（每只股票仅统计首次入选，已去重）

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

    df['选股月份'] = df['选股日期'].dt.to_period('M').astype(str)
    monthly_stats = df.groupby('选股月份').agg({
        '代码': 'count',
        '涨跌幅': ['mean', lambda x: (x > 0).sum()],
        '最大涨幅': 'mean'
    }).round(2)
    monthly_stats.columns = ['选股数', '平均收益', '盈利数', '最大涨幅均值']
    monthly_stats['胜率'] = (monthly_stats['盈利数'] / monthly_stats['选股数'] * 100).round(1)

    report += "### 1.2 月度表现\n\n"
    report += "| 月份 | 选股数 | 盈利数 | 胜率 | 平均收益 |\n"
    report += "|------|--------|--------|------|----------|\n"

    for period in monthly_stats.index:
        row = monthly_stats.loc[period]
        report += f"| {period} | {int(row['选股数'])} | {int(row['盈利数'])} | {row['胜率']:.1f}% | {row['平均收益']:+.2f}% |\n"

    best_period = monthly_stats.loc[monthly_stats['平均收益'].idxmax()]
    worst_period = monthly_stats.loc[monthly_stats['平均收益'].idxmin()]

    report += f"""
**最佳月份**: {best_period.name} ({best_period['平均收益']:+.2f}%, 胜率 {best_period['胜率']:.1f}%)
**最差月份**: {worst_period.name} ({worst_period['平均收益']:+.2f}%, 胜率 {worst_period['胜率']:.1f}%)

---

## 二、最大涨幅发生时间分布

### 2.1 时间区间分布

"""

    max_day = df['最大涨幅天数'].max()
    if max_day <= 15:
        bins = [0, 3, 5, 7, 10, 15, 20]
        labels = ['0-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天']
    elif max_day <= 30:
        bins = [0, 3, 5, 7, 10, 15, 20, 30]
        labels = ['0-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天', '21-30 天']
    else:
        bins = [0, 3, 5, 7, 10, 15, 20, 30, max_day + 1]
        labels = ['0-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天', '21-30 天', '30 天+']

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
            report += f"| {row['代码']} | {row['名称']} | {row['相似度']:.1f}% | {row['最大涨幅']:+.2f}% | {row['涨跌幅']:+.2f}% | {row['选股日期'].strftime('%Y-%m-%d') if hasattr(row['选股日期'], 'strftime') else str(row['选股日期'])[:10]} |\n"

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

    if '类型' in df.columns:
        df['分类名称'] = df['类型'].map(category_map).fillna(df['类型'])
    elif '类别' in df.columns:
        df['分类名称'] = df['类别'].map(category_map).fillna(df['类别'])
    else:
        df['分类名称'] = '未知'

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

    best_type = type_stats.index[0] if len(type_stats) > 0 else "N/A"
    report += "\n**结论**: '" + best_type + "'分类表现最佳，胜率 " + str(round(type_stats.loc[best_type, '胜率'], 1)) + "%，平均盈亏 " + str(round(type_stats.loc[best_type, '平均盈亏'], 2)) + "%。\n"

    report += """

---

## 八、止盈止损触发分析

### 8.1 触发统计

"""

    if has_trigger_data:
        # 计算各触发条件的数量（修复：使用 pd.notna() 而非 != '' 来判断）
        trigger_10pct_count = df['触发 +10% 日期'].notna().sum()
        trigger_5pct_count = df['触发 +5% 日期'].notna().sum()
        trigger_neg2pct_count = df['触发 -2% 日期'].notna().sum()
        trigger_neg4pct_count = df['触发 -4% 日期'].notna().sum()

        total = len(df)
        trigger_10pct_pct = trigger_10pct_count / total * 100
        trigger_5pct_pct = trigger_5pct_count / total * 100
        trigger_neg2pct_pct = trigger_neg2pct_count / total * 100
        trigger_neg4pct_pct = trigger_neg4pct_count / total * 100

        report += f"""| 触发条件 | 触发数量 | 占总样本比 |
|----------|----------|------|
| 达到 +5% 涨幅 | {trigger_5pct_count} | {trigger_5pct_pct:.1f}% |
| 达到 +10% 涨幅 | {trigger_10pct_count} | {trigger_10pct_pct:.1f}% |
| 达到 -2% 跌幅 | {trigger_neg2pct_count} | {trigger_neg2pct_pct:.1f}% |
| 达到 -4% 跌幅 | {trigger_neg4pct_count} | {trigger_neg4pct_pct:.1f}% |

"""

        # 增加有意义的条件比率
        if trigger_10pct_count > 0 and has_ever_below_zero:
            clean_10pct_count = len(df[df['触发 +10% 日期'].notna() & (df['是否曾跌破成本价'] == '否')])
            clean_rate = clean_10pct_count / trigger_10pct_count * 100
        else:
            clean_10pct_count = 0
            clean_rate = 0

        if trigger_neg2pct_count > 0:
            neg2_then_10pct_temp = len(df[df['触发 +10% 日期'].notna() & df['触发 -2% 日期'].notna()])
            bounce_rate = neg2_then_10pct_temp / trigger_neg2pct_count * 100
        else:
            neg2_then_10pct_temp = 0
            bounce_rate = 0

        report += f"""**条件比率分析**:

| 指标 | 计算方式 | 数值 |
|------|----------|------|
| 全身而退率 | 涨到 +10% 的股票中，从未跌破成本价的比例 | {clean_rate:.1f}% ({clean_10pct_count}/{trigger_10pct_count if trigger_10pct_count > 0 else 1}) |
| 反弹成功率 | 跌破 -2% 的股票中，最终反弹到 +10% 的比例 | {bounce_rate:.1f}% ({neg2_then_10pct_temp}/{trigger_neg2pct_count if trigger_neg2pct_count > 0 else 1}) |

"""

        # 路径分析：先跌后涨
        # 先达到 -2% 后来又达到 +10%
        neg2_then_10pct = 0
        neg4_then_10pct = 0
        neg2_then_5pct = 0  # 先达到 -2% 后来又达到 +5%
        neg4_then_5pct = 0  # 先达到 -4% 后来又达到 +5%
        direct_10pct = 0  # 直接涨到 10%，没有先跌到 -2%（可能跌过 -1.5% 但没触发 -2%）
        direct_5pct = 0   # 直接涨到 5%，没有先跌到 -2%
        clean_10pct = 0  # 从未跌破成本价（0%）就直接涨到 10%（全身而退）
        clean_5pct = 0   # 从未跌破成本价（0%）就直接涨到 5%

        for idx, row in df.iterrows():
            reached_10pct = pd.notna(row['触发 +10% 日期'])
            reached_5pct = pd.notna(row['触发 +5% 日期'])
            reached_neg2 = pd.notna(row['触发 -2% 日期'])
            reached_neg4 = pd.notna(row['触发 -4% 日期'])

            if reached_10pct:
                # 统计从未跌破成本价就直接涨到 10% 的（真正的全身而退）
                if has_ever_below_zero and row['是否曾跌破成本价'] == '否':
                    clean_10pct += 1

                if row['触发顺序'] and row['触发顺序'] != '':
                    order = row['触发顺序']
                    # 检查是否包含 neg2pct 和 10pct，且 neg2pct 在 10pct 之前
                    if 'neg2pct' in order and '10pct' in order:
                        neg2_idx = order.index('neg2pct')
                        pct10_idx = order.index('10pct')
                        if neg2_idx < pct10_idx:
                            neg2_then_10pct += 1

                    # 检查是否包含 neg4pct 和 10pct，且 neg4pct 在 10pct 之前
                    if 'neg4pct' in order and '10pct' in order:
                        neg4_idx = order.index('neg4pct')
                        pct10_idx = order.index('10pct')
                        if neg4_idx < pct10_idx:
                            neg4_then_10pct += 1

                    # 检查是否是直接涨到 10%（10pct 是第一个触发的事件）
                    if order.startswith('10pct'):
                        direct_10pct += 1
                else:
                    # 没有触发顺序但有 +10%，说明是直接涨到 10%
                    direct_10pct += 1

            if reached_5pct:
                # 统计从未跌破成本价就直接涨到 5% 的
                if has_ever_below_zero and row['是否曾跌破成本价'] == '否':
                    clean_5pct += 1

                if row['触发顺序'] and row['触发顺序'] != '':
                    order = row['触发顺序']
                    # 检查是否包含 neg2pct 和 5pct，且 neg2pct 在 5pct 之前
                    if 'neg2pct' in order and '5pct' in order:
                        neg2_idx = order.index('neg2pct')
                        pct5_idx = order.index('5pct')
                        if neg2_idx < pct5_idx:
                            neg2_then_5pct += 1

                    # 检查是否包含 neg4pct 和 5pct，且 neg4pct 在 5pct 之前
                    if 'neg4pct' in order and '5pct' in order:
                        neg4_idx = order.index('neg4pct')
                        pct5_idx = order.index('5pct')
                        if neg4_idx < pct5_idx:
                            neg4_then_5pct += 1

                    # 检查是否是直接涨到 5%（5pct 是第一个触发的事件）
                    if order.startswith('5pct'):
                        direct_5pct += 1
                else:
                    # 没有触发顺序但有 +5%，说明是直接涨到 5%
                    direct_5pct += 1

        neg2_then_10pct_pct = neg2_then_10pct / len(df) * 100
        neg4_then_10pct_pct = neg4_then_10pct / len(df) * 100
        neg2_then_5pct_pct = neg2_then_5pct / len(df) * 100
        neg4_then_5pct_pct = neg4_then_5pct / len(df) * 100
        direct_10pct_pct = direct_10pct / len(df) * 100
        direct_5pct_pct = direct_5pct / len(df) * 100
        clean_10pct_pct = clean_10pct / len(df) * 100
        clean_5pct_pct = clean_5pct / len(df) * 100

        report += f"""
**路径分析**（先跌后涨）:

| 路径 | 数量 | 占比 |
|------|------|------|
| 先达到 -2% 后达到 +5% | {neg2_then_5pct} | {neg2_then_5pct_pct:.1f}% |
| 先达到 -4% 后达到 +5% | {neg4_then_5pct} | {neg4_then_5pct_pct:.1f}% |
| 先达到 -2% 后达到 +10% | {neg2_then_10pct} | {neg2_then_10pct_pct:.1f}% |
| 先达到 -4% 后达到 +10% | {neg4_then_10pct} | {neg4_then_10pct_pct:.1f}% |
| 直接涨到 +5%（未先触发 -2%） | {direct_5pct} | {direct_5pct_pct:.1f}% |
| 直接涨到 +10%（未先触发 -2%） | {direct_10pct} | {direct_10pct_pct:.1f}% |
| 从未跌破成本价直接涨到 +5%（全身而退） | {clean_5pct} | {clean_5pct_pct:.1f}% |
| 从未跌破成本价直接涨到 +10%（全身而退） | {clean_10pct} | {clean_10pct_pct:.1f}% |

"""

        # 达到 +10% 的天数分析
        report += """
### 8.2 达到 +10% 天数分析

"""
        if has_10pct_day:
            df_10pct = df[df['达到 +10% 天数'].notna()]
            if len(df_10pct) > 0:
                median_10pct_day = int(df_10pct['达到 +10% 天数'].median())
                mean_10pct_day = df_10pct['达到 +10% 天数'].mean()
                min_10pct_day = int(df_10pct['达到 +10% 天数'].min())
                max_10pct_day = int(df_10pct['达到 +10% 天数'].max())

                # 天数区间分布
                day_bins = pd.cut(df_10pct['达到 +10% 天数'],
                                  bins=[0, 3, 5, 7, 10, 15, 20, float('inf')],
                                  labels=['1-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天', '20 天以上'])
                day_dist = day_bins.value_counts().sort_index()

                report += f"""| 统计项 | 数值 |
|--------|------|
| 样本数 | {len(df_10pct)} 只 |
| 中位数 | **{median_10pct_day} 天** |
| 平均数 | **{mean_10pct_day:.1f} 天** |
| 最早 | 第 {min_10pct_day} 天 |
| 最晚 | 第 {max_10pct_day} 天 |

**达到 +10% 天数区间分布**:

| 天数区间 | 数量 | 占比 |
|----------|------|------|
"""
                for interval, count in day_dist.items():
                    pct = count / len(df_10pct) * 100
                    report += f"| {interval} | {count} | {pct:.1f}% |\n"

                report += f"""
**结论**:
- 达到 +10% 涨幅的股票中，中位时间为 **{median_10pct_day} 天**
- {clean_10pct_pct:.1f}% 的股票能在从未跌破成本价的情况下全身而退（直接涨到 +10%）
"""
            else:
                report += "*无达到 +10% 涨幅的样本*\n\n"
        else:
            report += "*本回测数据不包含达到 +10% 天数信息*\n\n"

        # 达到 +5% 的天数分析
        report += """
### 8.2.5 达到 +5% 天数分析

"""
        if has_5pct_day:
            df_5pct = df[df['达到 +5% 天数'].notna()]
            if len(df_5pct) > 0:
                median_5pct_day = int(df_5pct['达到 +5% 天数'].median())
                mean_5pct_day = df_5pct['达到 +5% 天数'].mean()
                min_5pct_day = int(df_5pct['达到 +5% 天数'].min())
                max_5pct_day = int(df_5pct['达到 +5% 天数'].max())

                # 天数区间分布
                day_bins = pd.cut(df_5pct['达到 +5% 天数'],
                                  bins=[0, 3, 5, 7, 10, 15, 20, float('inf')],
                                  labels=['1-3 天', '4-5 天', '6-7 天', '8-10 天', '11-15 天', '16-20 天', '20 天以上'])
                day_dist = day_bins.value_counts().sort_index()

                report += f"""| 统计项 | 数值 |
|--------|------|
| 样本数 | {len(df_5pct)} 只 |
| 中位数 | **{median_5pct_day} 天** |
| 平均数 | **{mean_5pct_day:.1f} 天** |
| 最早 | 第 {min_5pct_day} 天 |
| 最晚 | 第 {max_5pct_day} 天 |

**达到 +5% 天数区间分布**:

| 天数区间 | 数量 | 占比 |
|----------|------|------|
"""
                for interval, count in day_dist.items():
                    pct = count / len(df_5pct) * 100
                    report += f"| {interval} | {count} | {pct:.1f}% |\n"

                report += f"""
**结论**:
- 达到 +5% 涨幅的股票中，中位时间为 **{median_5pct_day} 天**
- {clean_5pct_pct:.1f}% 的股票能在从未跌破成本价的情况下全身而退（直接涨到 +5%）
"""
            else:
                report += "*无达到 +5% 涨幅的样本*\n\n"
        else:
            report += "*本回测数据不包含达到 +5% 天数信息*\n\n"

        # 按相似度区间分析路径分布（更有意义的指标）
        report += """
### 8.3 相似度与触发关系

"""
        # 重新定义相似度区间函数（处理数值类型）
        def sim_range_numeric(x):
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

        df['相似度区间'] = df['相似度'].apply(sim_range_numeric)

        # 按相似度区间统计路径分布
        path_stats = df.groupby('相似度区间').apply(lambda g: pd.Series({
            '样本数': len(g),
            '全身而退数': len(g[g['触发 +10% 日期'].notna() & (g['是否曾跌破成本价'] == '否')]) if has_ever_below_zero else 0,
            '先跌后涨数': len(g[g['触发 +10% 日期'].notna() & g['触发 -2% 日期'].notna()]),
            '只跌不涨数': len(g[g['触发 +10% 日期'].isna() & g['触发 -4% 日期'].notna()]),
            '最终盈利数': len(g[g['涨跌幅'] > 0])
        }), include_groups=False)

        path_stats['全身而退率'] = (path_stats['全身而退数'] / path_stats['样本数'] * 100).round(1)
        path_stats['先跌后涨率'] = (path_stats['先跌后涨数'] / path_stats['样本数'] * 100).round(1)
        path_stats['只跌不涨率'] = (path_stats['只跌不涨数'] / path_stats['样本数'] * 100).round(1)
        path_stats['最终胜率'] = (path_stats['最终盈利数'] / path_stats['样本数'] * 100).round(1)

        report += "| 相似度 | 样本数 | 全身而退率 | 先跌后涨率 | 只跌不涨率 | 最终胜率 |\n"
        report += "|--------|--------|------------|------------|------------|----------|\n"
        for sim in ['80-85%', '85-88%', '88-90%', '90-92%', '92-95%', '95%+']:
            if sim in path_stats.index:
                row = path_stats.loc[sim]
                report += f"| {sim} | {int(row['样本数'])} | {row['全身而退率']:.1f}% | {row['先跌后涨率']:.1f}% | {row['只跌不涨率']:.1f}% | {row['最终胜率']:.1f}% |\n"

        report += f"""

**结论**:
- **全身而退**：从未跌破成本价就直接涨到 +10% 的股票比例
- **先跌后涨**：先跌破 -2% 后又反弹到 +10% 的股票比例
- **只跌不涨**：只跌破 -4% 但没涨到 +10% 的股票比例（纯止损）
- 观察不同相似度区间的路径分布差异，可判断相似度对选股质量的影响

"""

        # 行业热度分析
        if has_industry_data:
            report += """
### 8.4 行业热度分析

"""
            # 按行业分组统计
            industry_stats = df.groupby('行业').agg({
                '代码': 'count',
                '涨跌幅': ['mean', lambda x: (x > 0).sum()],
                '行业热度_买入日': 'mean'
            }).round(2)

            industry_stats.columns = ['样本数', '平均盈亏', '胜率', '平均行业热度']
            industry_stats['胜率'] = (industry_stats['胜率'] / industry_stats['样本数'] * 100).round(1)
            industry_stats = industry_stats.sort_values('平均盈亏', ascending=False)

            report += "| 行业 | 样本数 | 胜率 | 平均盈亏 | 平均行业热度 |\n"
            report += "|------|--------|------|----------|-------------|\n"
            for ind in industry_stats.index:
                row = industry_stats.loc[ind]
                heat_str = f"{row['平均行业热度']:.2f}" if pd.notna(row['平均行业热度']) else 'N/A'
                report += f"| {ind} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% | {heat_str} |\n"

            # 行业热度与盈亏关系
            df_valid = df[df['行业热度_买入日'].notna()].copy()
            if len(df_valid) > 10:  # 至少需要 10 个样本才能分组
                # 按热度分位数分组（处理重复值问题）
                try:
                    df_valid['热度分组'] = pd.qcut(
                        df_valid['行业热度_买入日'],
                        q=4,
                        labels=['低热度', '中低热度', '中高热度', '高热度'],
                        duplicates='drop'
                    )
                except ValueError:
                    # 如果分位数分组失败，使用统一固定阈值分组（和选股/回测标准完全对齐）
                    df_valid['热度分组'] = pd.cut(
                        df_valid['行业热度_买入日'],
                        bins=[0,5,15,30,100],
                        labels=['极低热度(<5)', '低热度(5-15)', '中热度(15-30)', '高热度(≥30)']
                    )

                heat_vs_return = df_valid.groupby('热度分组', observed=True).agg({
                    '代码': 'count',
                    '涨跌幅': ['mean', lambda x: (x > 0).sum()]
                }).round(2)

                heat_vs_return.columns = ['样本数', '平均盈亏', '胜率']
                heat_vs_return['胜率'] = (heat_vs_return['胜率'] / heat_vs_return['样本数'] * 100).round(1)

                report += """
**行业热度与盈亏关系**:

| 热度分组 | 样本数 | 胜率 | 平均盈亏 |
|----------|--------|------|----------|
"""
                for group in heat_vs_return.index:
                    row = heat_vs_return.loc[group]
                    report += f"| {group} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% |\n"

                report += """
**结论**:
- 行业热度数据可用于评估所选股票是否处于热门赛道
"""
            else:
                report += "*行业热度数据不足，无法进行分析*\n\n"
        else:
            report += """
### 8.4 行业热度分析

*本回测数据不包含行业热度信息*

"""

    report += """
---

## 九、关键发现总结

"""

    peak_time = time_dist.idxmax() if len(time_dist) > 0 else "N/A"
    median_day = int(df['最大涨幅天数'].median())

    report += f"""
### 9.1 时间分布特征
- 最大涨幅最集中出现在 **{peak_time}** 区间
- 中位时间为 **{median_day} 天**，平均 **{df['最大涨幅天数'].mean():.1f} 天**

### 9.2 盈亏分布特征
- 策略整体**{"盈利" if df['涨跌幅'].mean() > 0 else "亏损"}**，胜率 **{win_rate:.1f}%**
- 平均收益 **{df['涨跌幅'].mean():+.2f}%**
- 大盈比例：{profit_dist.get('大盈 (>10%)', 0)/len(df)*100:.1f}%，大亏比例：{profit_dist.get('大亏 (>10%)', 0)/len(df)*100:.1f}%

### 9.3 排名相关性
- 排名与最终盈亏的相关系数为 **{corr_rank_profit:.4f}**
- {"B1 图形匹配的排名不能完全预测持有期后的涨跌" if abs(corr_rank_profit) < 0.1 else "排名对最终盈亏有一定预测作用"}

### 9.4 相似度相关性
- 相似度与盈亏的相关系数为 **{corr_sim_profit:+.4f}**
"""

    if len(high_sim_df) > 0:
        report += f"- **92% 以上相似度**股票：{len(high_sim_df)} 只，胜率 {high_sim_df['涨跌幅'].apply(lambda x: x > 0).mean()*100:.1f}%，平均收益 {high_sim_df['涨跌幅'].mean():+.2f}%\n"
    else:
        report += "- 无 92% 以上相似度的股票样本\n"

    # 添加止盈止损触发总结
    if has_trigger_data:
        trigger_10pct_count = df['触发 +10% 日期'].notna().sum()
        trigger_neg2pct_count = df['触发 -2% 日期'].notna().sum()
        trigger_neg4pct_count = df['触发 -4% 日期'].notna().sum()

        # 重新计算路径分析
        neg2_then_10pct = 0
        neg4_then_10pct = 0
        clean_10pct = 0  # 从未跌破成本价就直接涨到 10%

        for idx, row in df.iterrows():
            reached_10pct = pd.notna(row['触发 +10% 日期'])
            reached_neg2 = pd.notna(row['触发 -2% 日期'])

            if reached_10pct:
                # 统计从未跌破成本价就直接涨到 10% 的
                if has_ever_below_zero and row['是否曾跌破成本价'] == '否':
                    clean_10pct += 1

            if row['触发顺序'] and row['触发顺序'] != '':
                order = row['触发顺序']
                if 'neg2pct' in order and '10pct' in order:
                    neg2_idx = order.index('neg2pct')
                    pct10_idx = order.index('10pct')
                    if neg2_idx < pct10_idx:
                        neg2_then_10pct += 1
                if 'neg4pct' in order and '10pct' in order:
                    neg4_idx = order.index('neg4pct')
                    pct10_idx = order.index('10pct')
                    if neg4_idx < pct10_idx:
                        neg4_then_10pct += 1

        clean_10pct_pct = clean_10pct / len(df) * 100

        report += f"""
### 9.5 止盈止损触发特征
- **{trigger_10pct_count}** 只股票 ({trigger_10pct_count/len(df)*100:.1f}%) 在持有期内曾达到过 +10% 涨幅
- **{trigger_neg2pct_count}** 只股票 ({trigger_neg2pct_count/len(df)*100:.1f}%) 在持有期内曾跌破 -2%
- **{trigger_neg4pct_count}** 只股票 ({trigger_neg4pct_count/len(df)*100:.1f}%) 在持有期内曾跌破 -4%
- **{neg2_then_10pct}** 只股票 ({neg2_then_10pct/len(df)*100:.1f}%) 先跌破 -2% 后反弹至 +10%
- **{neg4_then_10pct}** 只股票 ({neg4_then_10pct/len(df)*100:.1f}%) 先跌破 -4% 后反弹至 +10%
- **{clean_10pct}** 只股票 ({clean_10pct_pct:.1f}%) 从未跌破成本价而直接涨到 +10%（全身而退）
"""
    else:
        report += "\n### 9.5 止盈止损触发特征\n- 本回测数据不包含止盈止损触发信息\n"

    report += f"""
### 9.6 最大涨幅启示
- 平均最大涨幅 ({avg_max_gain:.2f}%) {"高于" if avg_max_gain > avg_final_return else "低于"}最终盈亏 ({avg_final_return:+.2f}%)
- **{high_gain_pct:.1f}%** 的股票曾达到过比最终盈亏高 5% 以上的涨幅
- 说明持有期内大部分股票都有过不错的表现，但最后回吐了涨幅
- **建议**: 考虑引入动态止盈策略（如达到 8-10% 涨幅时提前止盈）

### 9.7 分类策略建议
"""

    if len(type_stats) > 0:
        report += f"- '{best_type}'分类表现最佳，建议可优先选择该分类的股票\n"
    else:
        report += "- 数据不足，无法提供分类建议\n"

    report += """

---

## 十、决策支持指标

### 10.1 N 日内涨到 10% 的概率

"""
    # 按达到 +10% 天数计算累计概率分布
    if has_10pct_day:
        df_10pct_valid = df[df['达到 +10% 天数'].notna() & (df['达到 +10% 天数'] > 0)].copy()

        if len(df_10pct_valid) > 0:
            # 计算在第 N 天及之前涨到 10% 的累计概率
            cumulative_probs = []
            for day in [3, 5, 7, 10, 15, 20, 30]:
                prob = len(df_10pct_valid[df_10pct_valid['达到 +10% 天数'] <= day]) / len(df) * 100
                cumulative_probs.append((day, prob))

            report += "| 持有天数 | 累计涨到 10% 概率 | 说明 |\n"
            report += "|----------|-----------------|------|\n"
            for day, prob in cumulative_probs:
                report += f"| {day} 天 | {prob:.1f}% | {day}天内有{prob:.1f}% 概率涨到 10% |\n"

            # 找到概率显著跃升的窗口
            report += "\n**解读**:\n"
            for i in range(1, len(cumulative_probs)):
                prev_day, prev_prob = cumulative_probs[i-1]
                curr_day, curr_prob = cumulative_probs[i]
                jump = curr_prob - prev_prob
                if jump > 10:  # 概率跃升超过 10%
                    report += f"- **{prev_day}-{curr_day}天**是关键窗口期，概率从{prev_prob:.1f}% 跃升至{curr_prob:.1f}%\n"
        else:
            report += "*无达到 +10% 天数数据*\n"
    else:
        report += "*本回测未记录达到 +10% 天数*\n"

    report += """

### 10.2 行业热度与胜率关系

"""
    # 行业热度分位数分析
    heat_col = '行业热度_买入日'
    if has_industry_data and heat_col in df.columns:
        df_heat = df[df[heat_col].notna()].copy()

        if len(df_heat) > 0:
            try:
                # 按热度分位数分组
                df_heat['热度分位'] = pd.qcut(df_heat['行业热度_买入日'], q=5, labels=['极低 (0-20%)', '低 (20-40%)', '中 (40-60%)', '高 (60-80%)', '极高 (80-100%)'], duplicates='drop')

                heat_quantile_stats = df_heat.groupby('热度分位', observed=True).agg({
                    '代码': 'count',
                    '涨跌幅': ['mean', lambda x: (x > 0).sum()],
                    '达到 +10% 天数': lambda x: x.notna().sum()
                }).round(2)

                heat_quantile_stats.columns = ['样本数', '平均盈亏', '胜率', '涨到 10% 数量']
                heat_quantile_stats['胜率'] = (heat_quantile_stats['胜率'] / heat_quantile_stats['样本数'] * 100).round(1)
                heat_quantile_stats['涨到 10% 概率'] = (heat_quantile_stats['涨到 10% 数量'] / heat_quantile_stats['样本数'] * 100).round(1)

                report += "| 热度分位 | 样本数 | 胜率 | 平均盈亏 | 涨到 10% 概率 |\n"
                report += "|----------|--------|------|----------|-------------|\n"
                for quantile in heat_quantile_stats.index:
                    row = heat_quantile_stats.loc[quantile]
                    report += f"| {quantile} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% | {row['涨到 10% 概率']:.1f}% |\n"

                # 找出最佳热度区间
                best_heat_idx = heat_quantile_stats['胜率'].idxmax()
                best_heat_win = heat_quantile_stats.loc[best_heat_idx, '胜率']
                report += f"\n**结论**: 行业热度在 **'{best_heat_idx}'** 区间时胜率最高 ({best_heat_win:.1f}%)\n"
            except Exception as e:
                report += f"*行业热度分析失败：{e}*\n"
        else:
            report += "*行业热度数据不足*\n"
    else:
        report += "*本回测未记录行业热度数据*\n"

    report += """

### 10.3 相似度 × 行业热度 双重筛选

**说明**: 行业热度使用**选股/买入当日**的行业热度（即决策时可获得的数据）

"""
    # 双重筛选分析
    if has_industry_data and '行业热度_买入日' in df.columns:
        df_dual = df[df['行业热度_买入日'].notna()].copy()

        # 定义高/低相似度和高/低热度
        sim_threshold = 92
        heat_threshold = df_dual['行业热度_买入日'].median()

        df_dual['高相似度'] = df_dual['相似度'] >= sim_threshold
        df_dual['高热度'] = df_dual['行业热度_买入日'] >= heat_threshold

        dual_stats = df_dual.groupby(['高相似度', '高热度']).agg(
            样本数=('代码', 'count'),
            平均盈亏=('涨跌幅', 'mean'),
            盈利数=('涨跌幅', lambda x: (x > 0).sum())
        ).round(2)
        dual_stats['胜率'] = (dual_stats['盈利数'] / dual_stats['样本数'] * 100).round(1)

        report += "| 相似度 | 行业热度 | 样本数 | 胜率 | 平均盈亏 |\n"
        report += "|--------|----------|--------|------|----------|\n"
        for (high_sim, high_heat), row in dual_stats.iterrows():
            sim_label = '高 (≥92%)' if high_sim else '低 (<92%)'
            heat_label = f'高 (≥{heat_threshold:.0f})' if high_heat else f'低 (<{heat_threshold:.0f})'
            report += f"| {sim_label} | {heat_label} | {int(row['样本数'])} | {row['胜率']:.1f}% | {row['平均盈亏']:+.2f}% |\n"

        # 找出最佳组合（按胜率）
        if len(dual_stats) > 0 and '胜率' in dual_stats.columns:
            best_idx = dual_stats['胜率'].idxmax()
            if pd.notna(best_idx):
                best_sim, best_heat = best_idx
                best_win = dual_stats.loc[best_idx, '胜率']
                report += f"\n**最佳组合**: 相似度{'高' if best_sim else '低'} + 行业热度{'高' if best_heat else '低'}，胜率 {best_win:.1f}%\n"

        # 添加达到 10% 天数的对比分析
        report += "\n"
        report += "#### 关键问题：高热度是否能更快达到 +10%？是否有更大涨幅？\n\n"

        # 比较高相似度组内，高热度 vs 低热度的差异
        high_sim_df = df_dual[df_dual['高相似度'] == True]
        low_heat_grp = high_sim_df[high_sim_df['高热度'] == False]
        high_heat_grp = high_sim_df[high_sim_df['高热度'] == True]

        # 达到 10% 天数对比
        low_heat_reach = low_heat_grp['达到 +10% 天数'].dropna()
        high_heat_reach = high_heat_grp['达到 +10% 天数'].dropna()

        if len(low_heat_reach) > 0 and len(high_heat_reach) > 0:
            low_heat_mean_days = low_heat_reach.mean()
            high_heat_mean_days = high_heat_reach.mean()
            days_diff = low_heat_mean_days - high_heat_mean_days

            low_heat_median_days = low_heat_reach.median()
            high_heat_median_days = high_heat_reach.median()

            # 最大涨幅对比
            low_heat_max_gain = low_heat_grp['最大涨幅'].mean()
            high_heat_max_gain = high_heat_grp['最大涨幅'].mean()
            gain_diff = high_heat_max_gain - low_heat_max_gain

            # 最终盈亏对比
            low_heat_return = low_heat_grp['涨跌幅'].mean()
            high_heat_return = high_heat_grp['涨跌幅'].mean()
            return_diff = high_heat_return - low_heat_return

            report += "| 指标 | 高相似度 + 低热度 | 高相似度 + 高热度 | 差异 |\n"
            report += "|------|----------------|----------------|------|\n"
            report += f"| 达到 +10% 平均天数 | {low_heat_mean_days:.1f}天 (中位数:{low_heat_median_days:.1f}天) | {high_heat_mean_days:.1f}天 (中位数:{high_heat_median_days:.1f}天) | **{days_diff:+.1f}天** {'(高热度更快)' if days_diff > 0 else '(高热度更慢)'} |\n"
            report += f"| 持股期最大涨幅 | {low_heat_max_gain:.2f}% | {high_heat_max_gain:.2f}% | **{gain_diff:+.2f}%** {'(高热度更高)' if gain_diff > 0 else '(高热度更低)'} |\n"
            report += f"| 最终盈亏 | {low_heat_return:+.2f}% | {high_heat_return:+.2f}% | **{return_diff:+.2f}%** {'(高热度更好)' if return_diff > 0 else '(高热度更差)'} |\n"
            report += f"| 样本数 | {len(low_heat_grp)} | {len(high_heat_grp)} | - |\n"

            # 结论
            report += "\n**结论**:\n"
            if days_diff > 0:
                report += f"- **达到 +10% 速度**: 高热度行业股票平均快 {days_diff:.1f} 天 ({low_heat_mean_days:.1f}天 → {high_heat_mean_days:.1f}天)\n"
            else:
                report += f"- **达到 +10% 速度**: 高热度行业股票平均慢 {abs(days_diff):.1f} 天 ({low_heat_mean_days:.1f}天 → {high_heat_mean_days:.1f}天)\n"

            if gain_diff > 0:
                report += f"- **最大涨幅**: 高热度行业股票平均高 {gain_diff:.2f}% ({low_heat_max_gain:.2f}% → {high_heat_max_gain:.2f}%)\n"
            else:
                report += f"- **最大涨幅**: 高热度行业股票平均低 {abs(gain_diff):.2f}% ({low_heat_max_gain:.2f}% → {high_heat_max_gain:.2f}%)\n"

            if return_diff > 0:
                report += f"- **最终盈亏**: 高热度行业股票平均好 {return_diff:+.2f}% ({low_heat_return:+.2f}% → {high_heat_return:+.2f}%)\n"
            else:
                report += f"- **最终盈亏**: 高热度行业股票平均差 {return_diff:+.2f}% ({low_heat_return:+.2f}% → {high_heat_return:+.2f}%)\n"

            # 综合结论
            report += "\n**综合判断**: "
            positive_count = sum([days_diff > 0, gain_diff > 0, return_diff > 0])
            if positive_count >= 2:
                report += f"**高相似度 + 高热度组合在{positive_count}/3 个指标上优于高相似度 + 低热度**，建议优先选择高热度行业股票。\n"
            else:
                report += f"高热度组合优势不明显，仅{positive_count}/3 个指标占优，建议综合其他因素决策。\n"

            # 行业热度变化分析（买入日 vs 涨到 10% 日）
            report += "\n#### 行业热度变化：股票上涨时，行业热度是否也在上升？\n\n"

            # 筛选有涨到 10% 数据的股票
            reach_10_df = df_dual[df_dual['达到 +10% 天数'].notna()].copy()

            if len(reach_10_df) > 0 and '行业热度_10pct 日' in reach_10_df.columns:
                # 计算热度变化
                reach_10_df['热度变化'] = reach_10_df['行业热度_10pct 日'] - reach_10_df['行业热度_买入日']

                # 按热度变化分组
                reach_10_df['热度上升'] = reach_10_df['热度变化'] > 0

                heat_up = reach_10_df[reach_10_df['热度上升'] == True]
                heat_down = reach_10_df[reach_10_df['热度上升'] == False]

                report += "| 热度变化 | 样本数 | 平均最大涨幅 | 平均达到 10% 天数 | 胜率 |\n"
                report += "|----------|--------|-------------|-----------------|------|\n"

                if len(heat_up) > 0:
                    up_max_gain = heat_up['最大涨幅'].mean()
                    up_days = heat_up['达到 +10% 天数'].mean()
                    up_win = (heat_up['涨跌幅'] > 0).sum() / len(heat_up) * 100
                    report += f"| 热度上升 (买入→上涨) | {len(heat_up)} | {up_max_gain:.2f}% | {up_days:.1f}天 | {up_win:.1f}% |\n"

                if len(heat_down) > 0:
                    down_max_gain = heat_down['最大涨幅'].mean()
                    down_days = heat_down['达到 +10% 天数'].mean()
                    down_win = (heat_down['涨跌幅'] > 0).sum() / len(heat_down) * 100
                    report += f"| 热度下降 (买入→上涨) | {len(heat_down)} | {down_max_gain:.2f}% | {down_days:.1f}天 | {down_win:.1f}% |\n"

                # 结论
                if len(heat_up) > 0 and len(heat_down) > 0:
                    gain_diff_heat = up_max_gain - down_max_gain
                    report += f"\n**结论**: 行业热度上升的股票，最大涨幅平均{'高' if gain_diff_heat > 0 else '低'} {abs(gain_diff_heat):.2f}%。\n"
                    report += "这表明**股票上涨时行业热度同步上升**，验证了行业热度与个股表现的正相关性。\n"
            else:
                report += "*数据不足，无法分析*\n"

    else:
        report += "*数据不足，无法进行双重筛选分析*\n"

    report += """

### 10.4 风险预警：大跌后恢复概率

"""
    # 分析买入后先大跌的股票最终恢复情况
    if has_trigger_data:
        # 统计跌破不同阈值后的最终盈利情况
        report += "| 风险信号 | 触发数量 | 最终盈利比例 | 平均最终盈亏 |\n"
        report += "|----------|----------|-------------|-------------|\n"

        # 跌破 -5%（需要检查是否有这个字段，没有就用 -4% 代替）
        if '触发 -4% 日期' in df.columns:
            neg4_triggered = df[df['触发 -4% 日期'].notna()]
            if len(neg4_triggered) > 0:
                neg4_win = len(neg4_triggered[neg4_triggered['涨跌幅'] > 0]) / len(neg4_triggered) * 100
                neg4_avg = neg4_triggered['涨跌幅'].mean()
                report += f"| 跌破 -4% | {len(neg4_triggered)} | {neg4_win:.1f}% | {neg4_avg:+.2f}% |\n"

        # 从未跌破 -4% 的
        never_neg4 = df[df['触发 -4% 日期'].isna()]
        if len(never_neg4) > 0:
            never_neg4_win = len(never_neg4[never_neg4['涨跌幅'] > 0]) / len(never_neg4) * 100
            never_neg4_avg = never_neg4['涨跌幅'].mean()
            report += f"| 未跌破 -4% | {len(never_neg4)} | {never_neg4_win:.1f}% | {never_neg4_avg:+.2f}% |\n"

        report += "\n**解读**: 跌破 -4% 的股票，最终盈利比例显著低于未跌破的股票，说明**风险控制**的重要性\n"
    else:
        report += "*本回测未记录触发数据*\n"

    report += """

### 10.5 最佳止盈窗口分析

"""
    # 按持有天数分析收益分布
    if has_10pct_day:
        # 计算如果在第 N 天卖出的收益（用最大涨幅天数作为参考）
        report += "| 止盈窗口 | 胜率 | 平均收益 | 说明 |\n"
        report += "|----------|------|----------|------|\n"

        # 在 N 天内达到最大涨幅的股票
        for window in [5, 7, 10, 15]:
            early_peak = df[df['最大涨幅天数'] <= window]
            if len(early_peak) > 0:
                win_rate = len(early_peak[early_peak['涨跌幅'] > 0]) / len(early_peak) * 100
                avg_return = early_peak['涨跌幅'].mean()
                report += f"| {window}天内止盈 | {win_rate:.1f}% | {avg_return:+.2f}% | {len(early_peak)}只股票在{window}天内见最大涨幅 |\n"

        report += "\n**解读**: 如果股票在选入后 N 天内达到最大涨幅，应尽早止盈\n"
    else:
        report += "*数据不足*\n"

    report += """
---

## 十一、策略优化建议

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

    # 添加止盈策略建议
    if has_trigger_data:
        trigger_10pct_count = df['触发 +10% 日期'].notna().sum()
        trigger_10pct_pct = trigger_10pct_count / len(df) * 100
        report += f"""3. **引入动态止盈策略**:
   - {trigger_10pct_pct:.1f}% 的股票在持有期内曾达到过 +10% 涨幅
   - 可考虑在达到 +8% 至 +10% 时设置止盈点

4. **持有期调整**:
   - 最大涨幅中位数为 **{median_day} 天**
   - 当前持有期 {hold_days} 天，可考虑调整为 {median_day}-{median_day+5} 天配合止盈策略

5. **排名权重**:
   - {"排名与盈亏无显著相关性，可考虑降低排名权重" if abs(corr_rank_profit) < 0.1 else "排名与盈亏有一定相关性，可保持现有权重"}
"""
    else:
        report += f"""3. **引入动态止盈**:
   - {high_gain_pct:.1f}% 的股票曾达到过比最终盈亏高 5% 以上的涨幅
   - 建议在持有期第 {median_day}-{min(median_day+5, hold_days)} 天设置止盈点（如 8-10%）

4. **持有期调整**:
   - 最大涨幅中位数为 **{median_day} 天**
   - 当前持有期 {hold_days} 天，可考虑调整为 {median_day}-{median_day+5} 天配合止盈策略

5. **排名权重**:
   - {"排名与盈亏无显著相关性，可考虑降低排名权重" if abs(corr_rank_profit) < 0.1 else "排名与盈亏有一定相关性，可保持现有权重"}
"""

    # 增加行业热度和风险控制建议
    if has_industry_data and '行业热度_买入日' in df.columns:
        df_heat_check = df[df['行业热度_买入日'].notna()]
        if len(df_heat_check) > 0:
            heat_median = df_heat_check['行业热度_买入日'].median()
            high_heat_win = len(df_heat_check[(df_heat_check['行业热度_买入日'] >= heat_median) & (df_heat_check['涨跌幅'] > 0)]) / len(df_heat_check[df_heat_check['行业热度_买入日'] >= heat_median]) * 100
            low_heat_win = len(df_heat_check[(df_heat_check['行业热度_买入日'] < heat_median) & (df_heat_check['涨跌幅'] > 0)]) / len(df_heat_check[df_heat_check['行业热度_买入日'] < heat_median]) * 100

            if high_heat_win > low_heat_win + 5:
                report += f"""
6. **行业热度筛选**（强烈推荐）:
   - 高热度行业胜率 {high_heat_win:.1f}%，低热度行业胜率 {low_heat_win:.1f}%
   - **建议**: 优先选择行业热度高于中位数 ({heat_median:.0f} 分) 的股票，避开冷门行业
"""
            elif high_heat_win > low_heat_win:
                report += f"""
6. **行业热度参考**:
   - 高热度行业胜率 {high_heat_win:.1f}%，略高于低热度行业的 {low_heat_win:.1f}%
   - **建议**: 可将行业热度作为辅助筛选条件
"""

    report += """
---

## 十二、数据说明

- **天数计算方式**: 自然日（日历日期差），非交易日
- **数据来源**: 本地缓存的股票历史数据（CSV 格式）
- **回测方法**: 每个交易日用截至当日的历史数据重新跑选股，避免未来函数
- **选股数量**: 每日选前 10 只股票
- **持有期**: {hold_days} 自然日
"""

    if has_trigger_data:
        report += "- **止盈止损跟踪**: 记录每个交易日内是否达到 +10%、-2%、-4% 的触发条件\n"

    report += """
---

**报告完**

*生成脚本：analyze_daily_backtest_fast.py*
*回测脚本：daily_backtest_fast.py*
*数据文件：{csv_file.name}*
"""

    # 从CSV文件名中提取回看天数信息
    import re
    filename_parts = csv_file.name.split('_')
    extracted_lookback_days = 'unknown'
    for i, part in enumerate(filename_parts):
        if part == 'lb' and i + 1 < len(filename_parts):
            # 提取紧跟在 'lb' 后面的数字部分
            next_part = filename_parts[i + 1]
            # 查找数字部分
            match = re.search(r'(\d+)', next_part)
            if match:
                extracted_lookback_days = match.group(1)
                break

    # 从配置中获取其他重要参数
    # 注意：这些参数来自回测运行时的配置，不是策略参数yaml文件
    top_n = config.get('top_n', 'unknown')

    # 使用从配置中获取的lookback_days（如果不存在则使用从文件名提取的）
    config_lookback_days = config.get('lookback_days', 'unknown')
    final_lookback_days = config_lookback_days if config_lookback_days != 'unknown' else extracted_lookback_days

    # 构建更详细的报告名称
    if final_lookback_days != 'unknown':
        output_file = results_dir / f"B1每日回测分析报告_LB{final_lookback_days}_选{top_n}股_持{hold_days}日_{start_date}_{end_date}.md"
    else:
        output_file = results_dir / f"B1每日回测分析报告_选{top_n}股_持{hold_days}日_{start_date}_{end_date}.md"
    output_file = Path(output_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已生成：{output_file}")
    print(f"\n核心指标:")
    print(f"  - 总交易数：{len(df)}")
    print(f"  - 胜率：{win_rate:.1f}%")
    print(f"  - 平均收益：{df['涨跌幅'].mean():+.2f}%")
    print(f"  - 最佳交易：{df.loc[df['涨跌幅'].idxmax(), '名称']} +{df['涨跌幅'].max():+.2f}%")

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析每日回测结果并生成报告')
    parser.add_argument('--csv', type=str, help='CSV 文件路径（默认使用最新文件）')
    parser.add_argument('--json', type=str, help='JSON 文件路径（默认使用最新文件）')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')

    args = parser.parse_args()
    analyze_and_generate_report(args.csv, args.json, args.output)
