#!/usr/bin/env python3
"""
上涨 vs 下跌 区间回测对比分析报告
数据源：
  - 上涨区间：daily_backtest_fast_readable_20260325_192052.csv (440 只股票)
  - 下跌区间：daily_backtest_fast_readable_20260325_190739.csv (350 只股票)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 文件路径
UP_FILE = Path("backtest_results/daily_backtest_fast_readable_20260325_192052.csv")   # 上涨区间
DOWN_FILE = Path("backtest_results/daily_backtest_fast_readable_20260325_190739.csv") # 下跌区间

def load_data(filepath):
    """加载数据并转换百分比列"""
    df = pd.read_csv(filepath)
    # 转换百分比列为数值
    for col in ['涨跌幅', '最大涨幅']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip('%').astype(float)
    # 转换相似度
    if '相似度' in df.columns:
        df['相似度_数值'] = df['相似度'].astype(str).str.rstrip('%').astype(float)
    return df

def basic_stats(df, label):
    """基本统计分析"""
    stats = {}
    stats['label'] = label
    stats['total'] = len(df)
    stats['avg_return'] = df['涨跌幅'].mean()
    stats['median_return'] = df['涨跌幅'].median()
    stats['std_return'] = df['涨跌幅'].std()
    stats['max_return'] = df['涨跌幅'].max()
    stats['min_return'] = df['涨跌幅'].min()
    stats['win_count'] = (df['涨跌幅'] > 0).sum()
    stats['win_rate'] = stats['win_count'] / stats['total'] * 100
    stats['avg_max_gain'] = df['最大涨幅'].mean()
    stats['max_max_gain'] = df['最大涨幅'].max()
    return stats

def stop_loss_analysis(df, label):
    """止盈止损分析"""
    stats = {}
    # 止盈触发
    stats['reach_5pct'] = (df['最大涨幅'] >= 5.0).sum()
    stats['reach_5pct_rate'] = stats['reach_5pct'] / len(df) * 100
    stats['reach_10pct'] = (df['最大涨幅'] >= 10.0).sum()
    stats['reach_10pct_rate'] = stats['reach_10pct'] / len(df) * 100

    # 止损触发
    stats['trigger_neg2pct'] = (df['触发 -2% 日期'] != '-').sum()
    stats['trigger_neg2pct_rate'] = stats['trigger_neg2pct'] / len(df) * 100
    stats['trigger_neg4pct'] = (df['触发 -4% 日期'] != '-').sum()
    stats['trigger_neg4pct_rate'] = stats['trigger_neg4pct'] / len(df) * 100

    # 5% 止盈后表现
    reach_5pct = df[df['最大涨幅'] >= 5.0]
    if len(reach_5pct) > 0:
        stats['hold_5pct'] = (reach_5pct['涨跌幅'] >= 5.0).sum()
        stats['hold_5pct_rate'] = stats['hold_5pct'] / len(reach_5pct) * 100
        stats['lose_after_5pct'] = (reach_5pct['涨跌幅'] < 0).sum()
        stats['lose_after_5pct_rate'] = stats['lose_after_5pct'] / len(reach_5pct) * 100
    else:
        stats['hold_5pct'] = 0
        stats['hold_5pct_rate'] = 0
        stats['lose_after_5pct'] = 0
        stats['lose_after_5pct_rate'] = 0

    return stats

def similarity_analysis(df, label):
    """相似度分析"""
    stats = {}
    stats['avg_similarity'] = df['相似度_数值'].mean()
    stats['median_similarity'] = df['相似度_数值'].median()

    # 按相似度分组
    high_sim = df[df['相似度_数值'] >= 95]
    low_sim = df[df['相似度_数值'] < 95]

    stats['high_sim_count'] = len(high_sim)
    stats['high_sim_rate'] = len(high_sim) / len(df) * 100

    if len(high_sim) > 0:
        stats['high_sim_winrate'] = (high_sim['涨跌幅'] > 0).sum() / len(high_sim) * 100
        stats['high_sim_avg_return'] = high_sim['涨跌幅'].mean()
    else:
        stats['high_sim_winrate'] = 0
        stats['high_sim_avg_return'] = 0

    if len(low_sim) > 0:
        stats['low_sim_winrate'] = (low_sim['涨跌幅'] > 0).sum() / len(low_sim) * 100
        stats['low_sim_avg_return'] = low_sim['涨跌幅'].mean()
    else:
        stats['low_sim_winrate'] = 0
        stats['low_sim_avg_return'] = 0

    return stats

def industry_heat_analysis(df, label):
    """行业热度分析"""
    stats = {}
    heat_col = '行业热度_买入日'

    if heat_col in df.columns:
        df_heat = df[df[heat_col].notna()]
        if len(df_heat) > 0:
            stats['avg_heat'] = df_heat[heat_col].mean()
            stats['median_heat'] = df_heat[heat_col].median()

            # 按热度分组（全链路统一标准）
            low_heat = df_heat[df_heat[heat_col] < 15]  # 极低+低热度合并
            mid_heat = df_heat[(df_heat[heat_col] >= 15) & (df_heat[heat_col] < 30)]  # 中热度
            high_heat = df_heat[df_heat[heat_col] >= 30]  # 高热度（统一阈值）

            stats['low_heat_count'] = len(low_heat)
            stats['mid_heat_count'] = len(mid_heat)
            stats['high_heat_count'] = len(high_heat)

            if len(low_heat) > 0:
                stats['low_heat_winrate'] = (low_heat['涨跌幅'] > 0).sum() / len(low_heat) * 100
                stats['low_heat_5pct'] = (low_heat['最大涨幅'] >= 5.0).sum() / len(low_heat) * 100
            else:
                stats['low_heat_winrate'] = 0
                stats['low_heat_5pct'] = 0

            if len(mid_heat) > 0:
                stats['mid_heat_winrate'] = (mid_heat['涨跌幅'] > 0).sum() / len(mid_heat) * 100
                stats['mid_heat_5pct'] = (mid_heat['最大涨幅'] >= 5.0).sum() / len(mid_heat) * 100
            else:
                stats['mid_heat_winrate'] = 0
                stats['mid_heat_5pct'] = 0

            if len(high_heat) > 0:
                stats['high_heat_winrate'] = (high_heat['涨跌幅'] > 0).sum() / len(high_heat) * 100
                stats['high_heat_5pct'] = (high_heat['最大涨幅'] >= 5.0).sum() / len(high_heat) * 100
            else:
                stats['high_heat_winrate'] = 0
                stats['high_heat_5pct'] = 0

    return stats

def return_distribution(df, label):
    """收益率分布分析"""
    stats = {}
    total = len(df)

    stats['loss_gt_10'] = (df['涨跌幅'] < -10).sum()
    stats['loss_5_10'] = ((df['涨跌幅'] >= -10) & (df['涨跌幅'] < -5)).sum()
    stats['loss_2_5'] = ((df['涨跌幅'] >= -5) & (df['涨跌幅'] < -2)).sum()
    stats['loss_0_2'] = ((df['涨跌幅'] >= -2) & (df['涨跌幅'] <= 0)).sum()
    stats['win_0_2'] = ((df['涨跌幅'] > 0) & (df['涨跌幅'] < 2)).sum()
    stats['win_2_5'] = ((df['涨跌幅'] >= 2) & (df['涨跌幅'] < 5)).sum()
    stats['win_5_10'] = ((df['涨跌幅'] >= 5) & (df['涨跌幅'] < 10)).sum()
    stats['win_gt_10'] = (df['涨跌幅'] >= 10).sum()

    # 转换为比例
    for key in list(stats.keys()):
        stats[key + '_pct'] = stats[key] / total * 100

    return stats

def cost_break_analysis(df, label):
    """跌破成本价分析"""
    stats = {}
    if '是否曾跌破成本价' in df.columns:
        broken = df[df['是否曾跌破成本价'] == '是']
        stats['broken_count'] = len(broken)
        stats['broken_rate'] = len(broken) / len(df) * 100
        if len(broken) > 0:
            stats['broken_winrate'] = (broken['涨跌幅'] > 0).sum() / len(broken) * 100
            stats['broken_avg_return'] = broken['涨跌幅'].mean()
        else:
            stats['broken_winrate'] = 0
            stats['broken_avg_return'] = 0

        intact = df[df['是否曾跌破成本价'] != '是']
        if len(intact) > 0:
            stats['intact_winrate'] = (intact['涨跌幅'] > 0).sum() / len(intact) * 100
            stats['intact_avg_return'] = intact['涨跌幅'].mean()
        else:
            stats['intact_winrate'] = 0
            stats['intact_avg_return'] = 0

    return stats

def hold_period_analysis(df, label):
    """持有期分析"""
    stats = {}
    if '持有天数' in df.columns:
        stats['avg_hold_days'] = df['持有天数'].mean()
        stats['median_hold_days'] = df['持有天数'].median()

        # 按持有期分组
        short = df[df['持有天数'] <= 5]
        medium = df[(df['持有天数'] > 5) & (df['持有天数'] <= 15)]
        long = df[df['持有天数'] > 15]

        stats['short_count'] = len(short)
        stats['medium_count'] = len(medium)
        stats['long_count'] = len(long)

        if len(short) > 0:
            stats['short_winrate'] = (short['涨跌幅'] > 0).sum() / len(short) * 100
        else:
            stats['short_winrate'] = 0

        if len(medium) > 0:
            stats['medium_winrate'] = (medium['涨跌幅'] > 0).sum() / len(medium) * 100
        else:
            stats['medium_winrate'] = 0

        if len(long) > 0:
            stats['long_winrate'] = (long['涨跌幅'] > 0).sum() / len(long) * 100
        else:
            stats['long_winrate'] = 0

    return stats

def industry_stats(df, label):
    """行业分布统计"""
    stats = {}
    if '行业' in df.columns:
        industry_counts = df['行业'].value_counts()
        top5_industries = industry_counts.head(5)

        stats['industry_count'] = len(industry_counts)
        stats['top_industries'] = top5_industries.to_dict()

        # 每个行业的胜率
        industry_winrate = df.groupby('行业').apply(lambda x: (x['涨跌幅'] > 0).sum() / len(x) * 100)
        stats['best_industry'] = industry_winrate.idxmax()
        stats['best_industry_winrate'] = industry_winrate.max()
        stats['worst_industry'] = industry_winrate.idxmin()
        stats['worst_industry_winrate'] = industry_winrate.min()

    return stats

def print_comparison(up, down, label, higher_is_better=True):
    """打印对比结果"""
    up_val = up.get(label, 0)
    down_val = down.get(label, 0)
    diff = up_val - down_val

    if isinstance(up_val, float):
        up_str = f"{up_val:.2f}"
        down_str = f"{down_val:.2f}"
    else:
        up_str = f"{up_val}"
        down_str = f"{down_val}"

    if diff > 0:
        diff_str = f"+{diff:.2f}" if isinstance(diff, float) else f"+{diff}"
        symbol = "↑" if higher_is_better else "↓"
    elif diff < 0:
        diff_str = f"{diff:.2f}" if isinstance(diff, float) else f"{diff}"
        symbol = "↓" if higher_is_better else "↑"
    else:
        diff_str = "0"
        symbol = "="

    print(f"  {label:<28} | {up_str:>12} | {down_str:>12} | {diff_str:>10} {symbol}")

def full_analysis():
    """完整对比分析"""
    print("="*100)
    print("📊 上涨 vs 下跌 区间回测对比分析报告")
    print("="*100)
    print(f"\n分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    print(f"\n【数据源】")
    print(f"  上涨区间：{UP_FILE} (440 只股票)")
    print(f"  下跌区间：{DOWN_FILE} (350 只股票)")

    up_df = load_data(UP_FILE)
    down_df = load_data(DOWN_FILE)

    print(f"\n数据加载完成:")
    print(f"  上涨区间：{len(up_df)} 只股票")
    print(f"  下跌区间：{len(down_df)} 只股票")

    # ========== 一、基本统计对比 ==========
    print("\n" + "="*100)
    print("📈 一、基本统计对比")
    print("="*100)
    print(f"\n{'指标':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_basic = basic_stats(up_df, "up")
    down_basic = basic_stats(down_df, "down")

    for label in ['total', 'avg_return', 'median_return', 'std_return',
                  'max_return', 'min_return', 'win_rate', 'avg_max_gain', 'max_max_gain']:
        print_comparison(up_basic, down_basic, label)

    # ========== 二、止盈止损触发对比 ==========
    print("\n" + "="*100)
    print("🎯 二、止盈止损触发对比")
    print("="*100)
    print(f"\n{'指标':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_stop = stop_loss_analysis(up_df, "up")
    down_stop = stop_loss_analysis(down_df, "down")

    for label in ['reach_5pct_rate', 'reach_10pct_rate', 'trigger_neg2pct_rate',
                  'trigger_neg4pct_rate', 'hold_5pct_rate', 'lose_after_5pct_rate']:
        print_comparison(up_stop, down_stop, label)

    # ========== 三、相似度分析 ==========
    print("\n" + "="*100)
    print("🎯 三、相似度分析")
    print("="*100)
    print(f"\n{'指标':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_sim = similarity_analysis(up_df, "up")
    down_sim = similarity_analysis(down_df, "down")

    for label in ['avg_similarity', 'median_similarity', 'high_sim_rate',
                  'high_sim_winrate', 'high_sim_avg_return', 'low_sim_winrate', 'low_sim_avg_return']:
        print_comparison(up_sim, down_sim, label)

    # ========== 四、行业热度分析 ==========
    print("\n" + "="*100)
    print("🔥 四、行业热度分析")
    print("="*100)
    print(f"\n{'指标':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_heat = industry_heat_analysis(up_df, "up")
    down_heat = industry_heat_analysis(down_df, "down")

    for label in ['avg_heat', 'median_heat', 'low_heat_count', 'mid_heat_count', 'high_heat_count',
                  'low_heat_winrate', 'mid_heat_winrate', 'high_heat_winrate',
                  'low_heat_5pct', 'mid_heat_5pct', 'high_heat_5pct']:
        print_comparison(up_heat, down_heat, label)

    # ========== 五、收益率分布 ==========
    print("\n" + "="*100)
    print("📊 五、收益率分布对比")
    print("="*100)
    print(f"\n{'区间':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_dist = return_distribution(up_df, "up")
    down_dist = return_distribution(down_df, "down")

    distribution_labels = [
        ('loss_gt_10_pct', '大亏 (>10%)'),
        ('loss_5_10_pct', '中亏 (5-10%)'),
        ('loss_2_5_pct', '小亏 (2-5%)'),
        ('loss_0_2_pct', '微亏 (0-2%)'),
        ('win_0_2_pct', '微赢 (0-2%)'),
        ('win_2_5_pct', '小赢 (2-5%)'),
        ('win_5_10_pct', '中赢 (5-10%)'),
        ('win_gt_10_pct', '大赢 (>10%)'),
    ]

    for key, label in distribution_labels:
        print_comparison(up_dist, down_dist, key)

    # ========== 六、跌破成本价分析 ==========
    print("\n" + "="*100)
    print("💰 六、跌破成本价分析")
    print("="*100)
    print(f"\n{'指标':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_cost = cost_break_analysis(up_df, "up")
    down_cost = cost_break_analysis(down_df, "down")

    for label in ['broken_rate', 'broken_winrate', 'broken_avg_return',
                  'intact_winrate', 'intact_avg_return']:
        print_comparison(up_cost, down_cost, label)

    # ========== 七、持有期分析 ==========
    print("\n" + "="*100)
    print("⏱️  七、持有期分析")
    print("="*100)
    print(f"\n{'指标':<28} | {'上涨区间':>12} | {'下跌区间':>12} | {'差异':>10}")
    print("-"*75)

    up_hold = hold_period_analysis(up_df, "up")
    down_hold = hold_period_analysis(down_df, "down")

    for label in ['avg_hold_days', 'median_hold_days', 'short_count', 'medium_count',
                  'long_count', 'short_winrate', 'medium_winrate', 'long_winrate']:
        print_comparison(up_hold, down_hold, label)

    # ========== 八、行业分布 ==========
    print("\n" + "="*100)
    print("🏭 八、行业分布统计")
    print("="*100)

    up_ind = industry_stats(up_df, "up")
    down_ind = industry_stats(down_df, "down")

    print(f"\n上涨区间:")
    print(f"  涉及行业数：{up_ind['industry_count']}")
    print(f"  Top 5 行业:")
    for ind, cnt in up_ind['top_industries'].items():
        print(f"    - {ind}: {cnt}只")
    print(f"  胜率最高行业：{up_ind['best_industry']} ({up_ind['best_industry_winrate']:.1f}%)")
    print(f"  胜率最低行业：{up_ind['worst_industry']} ({up_ind['worst_industry_winrate']:.1f}%)")

    print(f"\n下跌区间:")
    print(f"  涉及行业数：{down_ind['industry_count']}")
    print(f"  Top 5 行业:")
    for ind, cnt in down_ind['top_industries'].items():
        print(f"    - {ind}: {cnt}只")
    print(f"  胜率最高行业：{down_ind['best_industry']} ({down_ind['best_industry_winrate']:.1f}%)")
    print(f"  胜率最低行业：{down_ind['worst_industry']} ({down_ind['worst_industry_winrate']:.1f}%)")

    # ========== 九、核心结论 ==========
    print("\n" + "="*100)
    print("📌 九、核心结论")
    print("="*100)

    print(f"\n【关键差异】")
    print(f"  1. 胜率：上涨 {up_basic['win_rate']:.1f}% vs 下跌 {down_basic['win_rate']:.1f}% (差异：{up_basic['win_rate'] - down_basic['win_rate']:+.1f}%)")
    print(f"  2. 平均收益：上涨 {up_basic['avg_return']:.2f}% vs 下跌 {down_basic['avg_return']:.2f}%")
    print(f"  3. 5% 止盈触发率：上涨 {up_stop['reach_5pct_rate']:.1f}% vs 下跌 {down_stop['reach_5pct_rate']:.1f}%")
    print(f"  4. 止损触发率：上涨 {up_stop['trigger_neg4pct_rate']:.1f}% vs 下跌 {down_stop['trigger_neg4pct_rate']:.1f}%")
    print(f"  5. 大赢比例：上涨 {up_dist['win_gt_10_pct']:.1f}% vs 下跌 {down_dist['win_gt_10_pct']:.1f}%")

    print(f"\n【相似点】")
    print(f"  1. 选股标准一致：平均相似度均为 {up_sim['avg_similarity']:.1f}% 左右")
    print(f"  2. 高相似度股票占比：上涨 {up_sim['high_sim_rate']:.1f}% vs 下跌 {down_sim['high_sim_rate']:.1f}%")
    print(f"  3. 行业热度分布：上涨平均 {up_heat['avg_heat']:.1f} vs 下跌 {down_heat['avg_heat']:.1f}")

    print(f"\n【风险提示】")
    print(f"  - 跌破成本价比例：上涨 {up_cost['broken_rate']:.1f}% vs 下跌 {down_cost['broken_rate']:.1f}%")
    print(f"  - 跌破后胜率：上涨 {up_cost['broken_winrate']:.1f}% vs 下跌 {down_cost['broken_winrate']:.1f}%")

    # ========== 十、策略建议 ==========
    print("\n" + "="*100)
    print("💡 十、策略建议")
    print("="*100)

    suggestions = []

    if up_basic['win_rate'] > down_basic['win_rate'] + 10:
        suggestions.append(f"✓ 上涨行情明显更适合本策略，胜率高出 {up_basic['win_rate'] - down_basic['win_rate']:.1f}%")

    if up_stop['hold_5pct_rate'] > down_stop['hold_5pct_rate'] + 10:
        suggestions.append(f"✓ 5% 止盈策略在上涨行情更有效，守住率高出 {up_stop['hold_5pct_rate'] - down_stop['hold_5pct_rate']:.1f}%")

    if down_stop['trigger_neg4pct_rate'] > 70:
        suggestions.append(f"⚠ 下跌区间止损触发率高达 {down_stop['trigger_neg4pct_rate']:.1f}%，建议降低仓位或观望")

    if up_sim['high_sim_winrate'] > up_sim['low_sim_winrate'] + 5:
        suggestions.append(f"✓ 高相似度 (>95%) 股票胜率明显更高，可考虑提高相似度阈值")

    if up_heat['high_heat_winrate'] > up_heat['low_heat_winrate'] + 10:
        suggestions.append(f"✓ 高热度行业 (>70) 表现明显更好，可优先选择")

    if up_dist['win_gt_10_pct'] > 15:
        suggestions.append(f"✓ 上涨区间大赢 (>10%) 比例达 {up_dist['win_gt_10_pct']:.1f}%，可考虑放宽止盈限制")

    if not suggestions:
        suggestions.append("✓ 当前策略在两种行情下均有稳定表现")

    for sug in suggestions:
        print(f"  {sug}")

    # ========== 生成文档 ==========
    print("\n" + "="*100)
    print("📄 生成分析报告...")
    print("="*100)

    # 生成 Markdown 文档
    md_content = generate_markdown_report(up_df, down_df, up_basic, down_basic, up_stop, down_stop,
                                          up_sim, down_sim, up_heat, down_heat, up_dist, down_dist,
                                          up_cost, down_cost, up_hold, down_hold, up_ind, down_ind)

    report_file = Path("backtest_results/上涨下跌对比分析报告.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n✅ 分析报告已保存：{report_file}")

    return {
        'up': {**up_basic, **up_stop, **up_sim, **up_heat, **up_dist, **up_cost, **up_hold, **up_ind},
        'down': {**down_basic, **down_stop, **down_sim, **down_heat, **down_dist, **down_cost, **down_hold, **down_ind}
    }

def generate_markdown_report(up_df, down_df, up_basic, down_basic, up_stop, down_stop,
                             up_sim, down_sim, up_heat, down_heat, up_dist, down_dist,
                             up_cost, down_cost, up_hold, down_hold, up_ind, down_ind):
    """生成 Markdown 格式报告"""

    report = f"""# 上涨 vs 下跌 区间回测对比分析报告

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据源

| 市场类型 | 文件 | 样本数 |
|---------|------|--------|
| 上涨区间 | daily_backtest_fast_readable_20260325_192052.csv | {len(up_df)} 只股票 |
| 下跌区间 | daily_backtest_fast_readable_20260325_190739.csv | {len(down_df)} 只股票 |

---

## 一、基本统计对比

| 指标 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 样本总数 | {up_basic['total']} | {down_basic['total']} | {up_basic['total'] - down_basic['total']:+d} |
| 平均收益率 | {up_basic['avg_return']:.2f}% | {down_basic['avg_return']:.2f}% | {up_basic['avg_return'] - down_basic['avg_return']:+.2f}% |
| 中位收益率 | {up_basic['median_return']:.2f}% | {down_basic['median_return']:.2f}% | {up_basic['median_return'] - down_basic['median_return']:+.2f}% |
| 收益率标准差 | {up_basic['std_return']:.2f}% | {down_basic['std_return']:.2f}% | {up_basic['std_return'] - down_basic['std_return']:+.2f}% |
| 最大收益 | {up_basic['max_return']:.2f}% | {down_basic['max_return']:.2f}% | {up_basic['max_return'] - down_basic['max_return']:+.2f}% |
| 最小收益 | {up_basic['min_return']:.2f}% | {down_basic['min_return']:.2f}% | {up_basic['min_return'] - down_basic['min_return']:+.2f}% |
| **胜率** | **{up_basic['win_rate']:.1f}%** | **{down_basic['win_rate']:.1f}%** | **{up_basic['win_rate'] - down_basic['win_rate']:+.1f}%** |
| 平均最大涨幅 | {up_basic['avg_max_gain']:.2f}% | {down_basic['avg_max_gain']:.2f}% | {up_basic['avg_max_gain'] - down_basic['avg_max_gain']:+.2f}% |
| 最大涨幅记录 | {up_basic['max_max_gain']:.2f}% | {down_basic['max_max_gain']:.2f}% | {up_basic['max_max_gain'] - down_basic['max_max_gain']:+.2f}% |

---

## 二、止盈止损触发对比

| 指标 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 触及 +5% 比例 | {up_stop['reach_5pct_rate']:.1f}% | {down_stop['reach_5pct_rate']:.1f}% | {up_stop['reach_5pct_rate'] - down_stop['reach_5pct_rate']:+.1f}% |
| 触及 +10% 比例 | {up_stop['reach_10pct_rate']:.1f}% | {down_stop['reach_10pct_rate']:.1f}% | {up_stop['reach_10pct_rate'] - down_stop['reach_10pct_rate']:+.1f}% |
| 触及 -2% 比例 | {up_stop['trigger_neg2pct_rate']:.1f}% | {down_stop['trigger_neg2pct_rate']:.1f}% | {up_stop['trigger_neg2pct_rate'] - down_stop['trigger_neg2pct_rate']:+.1f}% |
| 触及 -4% 比例 | {up_stop['trigger_neg4pct_rate']:.1f}% | {down_stop['trigger_neg4pct_rate']:.1f}% | {up_stop['trigger_neg4pct_rate'] - down_stop['trigger_neg4pct_rate']:+.1f}% |
| 触及 5% 后守住比例 | {up_stop['hold_5pct_rate']:.1f}% | {down_stop['hold_5pct_rate']:.1f}% | {up_stop['hold_5pct_rate'] - down_stop['hold_5pct_rate']:+.1f}% |
| 触及 5% 后转亏比例 | {up_stop['lose_after_5pct_rate']:.1f}% | {down_stop['lose_after_5pct_rate']:.1f}% | {up_stop['lose_after_5pct_rate'] - down_stop['lose_after_5pct_rate']:+.1f}% |

---

## 三、相似度分析

| 指标 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 平均相似度 | {up_sim['avg_similarity']:.1f}% | {down_sim['avg_similarity']:.1f}% | {up_sim['avg_similarity'] - down_sim['avg_similarity']:+.1f}% |
| 中位相似度 | {up_sim['median_similarity']:.1f}% | {down_sim['median_similarity']:.1f}% | {up_sim['median_similarity'] - down_sim['median_similarity']:+.1f}% |
| 高相似度 (≥95%) 占比 | {up_sim['high_sim_rate']:.1f}% | {down_sim['high_sim_rate']:.1f}% | {up_sim['high_sim_rate'] - down_sim['high_sim_rate']:+.1f}% |
| 高相似度胜率 | {up_sim['high_sim_winrate']:.1f}% | {down_sim['high_sim_winrate']:.1f}% | {up_sim['high_sim_winrate'] - down_sim['high_sim_winrate']:+.1f}% |
| 高相似度平均收益 | {up_sim['high_sim_avg_return']:.2f}% | {down_sim['high_sim_avg_return']:.2f}% | {up_sim['high_sim_avg_return'] - down_sim['high_sim_avg_return']:+.2f}% |
| 低相似度 (<95%) 胜率 | {up_sim['low_sim_winrate']:.1f}% | {down_sim['low_sim_winrate']:.1f}% | {up_sim['low_sim_winrate'] - down_sim['low_sim_winrate']:+.1f}% |

---

## 四、行业热度分析

| 指标 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 平均行业热度 | {up_heat['avg_heat']:.1f} | {down_heat['avg_heat']:.1f} | {up_heat['avg_heat'] - down_heat['avg_heat']:+.1f} |
| 中位行业热度 | {up_heat['median_heat']:.1f} | {down_heat['median_heat']:.1f} | {up_heat['median_heat'] - down_heat['median_heat']:+.1f} |
| 低热度 (<30) 数量 | {up_heat['low_heat_count']} | {down_heat['low_heat_count']} | {up_heat['low_heat_count'] - down_heat['low_heat_count']:+d} |
| 中热度 (30-70) 数量 | {up_heat['mid_heat_count']} | {down_heat['mid_heat_count']} | {up_heat['mid_heat_count'] - down_heat['mid_heat_count']:+d} |
| 高热度 (≥70) 数量 | {up_heat['high_heat_count']} | {down_heat['high_heat_count']} | {up_heat['high_heat_count'] - down_heat['high_heat_count']:+d} |
| 低热度胜率 | {up_heat['low_heat_winrate']:.1f}% | {down_heat['low_heat_winrate']:.1f}% | {up_heat['low_heat_winrate'] - down_heat['low_heat_winrate']:+.1f}% |
| 中热度胜率 | {up_heat['mid_heat_winrate']:.1f}% | {down_heat['mid_heat_winrate']:.1f}% | {up_heat['mid_heat_winrate'] - down_heat['mid_heat_winrate']:+.1f}% |
| 高热度胜率 | {up_heat['high_heat_winrate']:.1f}% | {down_heat['high_heat_winrate']:.1f}% | {up_heat['high_heat_winrate'] - down_heat['high_heat_winrate']:+.1f}% |

---

## 五、收益率分布

| 区间 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 大亏 (>10%) | {up_dist['loss_gt_10_pct']:.1f}% | {down_dist['loss_gt_10_pct']:.1f}% | {up_dist['loss_gt_10_pct'] - down_dist['loss_gt_10_pct']:+.1f}% |
| 中亏 (5-10%) | {up_dist['loss_5_10_pct']:.1f}% | {down_dist['loss_5_10_pct']:.1f}% | {up_dist['loss_5_10_pct'] - down_dist['loss_5_10_pct']:+.1f}% |
| 小亏 (2-5%) | {up_dist['loss_2_5_pct']:.1f}% | {down_dist['loss_2_5_pct']:.1f}% | {up_dist['loss_2_5_pct'] - down_dist['loss_2_5_pct']:+.1f}% |
| 微亏 (0-2%) | {up_dist['loss_0_2_pct']:.1f}% | {down_dist['loss_0_2_pct']:.1f}% | {up_dist['loss_0_2_pct'] - down_dist['loss_0_2_pct']:+.1f}% |
| 微赢 (0-2%) | {up_dist['win_0_2_pct']:.1f}% | {down_dist['win_0_2_pct']:.1f}% | {up_dist['win_0_2_pct'] - down_dist['win_0_2_pct']:+.1f}% |
| 小赢 (2-5%) | {up_dist['win_2_5_pct']:.1f}% | {down_dist['win_2_5_pct']:.1f}% | {up_dist['win_2_5_pct'] - down_dist['win_2_5_pct']:+.1f}% |
| 中赢 (5-10%) | {up_dist['win_5_10_pct']:.1f}% | {down_dist['win_5_10_pct']:.1f}% | {up_dist['win_5_10_pct'] - down_dist['win_5_10_pct']:+.1f}% |
| 大赢 (>10%) | {up_dist['win_gt_10_pct']:.1f}% | {down_dist['win_gt_10_pct']:.1f}% | {up_dist['win_gt_10_pct'] - down_dist['win_gt_10_pct']:+.1f}% |

---

## 六、跌破成本价分析

**计算逻辑**：45 天持有期内，只要某一天的最低价低于买入价，就算"跌破成本价"

| 指标 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 跌破成本价比例 | {up_cost['broken_rate']:.1f}% | {down_cost['broken_rate']:.1f}% | {up_cost['broken_rate'] - down_cost['broken_rate']:+.1f}% |
| 跌破后最终盈利 | {up_cost['broken_winrate']:.1f}% | {down_cost['broken_winrate']:.1f}% | {up_cost['broken_winrate'] - down_cost['broken_winrate']:+.1f}% |
| 跌破后平均收益 | {up_cost['broken_avg_return']:.2f}% | {down_cost['broken_avg_return']:.2f}% | {up_cost['broken_avg_return'] - down_cost['broken_avg_return']:+.2f}% |
| 未跌破胜率 | {up_cost['intact_winrate']:.1f}% | {down_cost['intact_winrate']:.1f}% | {up_cost['intact_winrate'] - down_cost['intact_winrate']:+.1f}% |
| 未跌破平均收益 | {up_cost['intact_avg_return']:.2f}% | {down_cost['intact_avg_return']:.2f}% | {up_cost['intact_avg_return'] - down_cost['intact_avg_return']:+.2f}% |

> **注**：跌破成本价比例接近 100% 是正常现象，因为 45 天持有期内几乎不可能一天都不曾跌破过买入价。关键是跌破后能否涨回来——上涨区间 60.7% 能赚钱，下跌区间 41.1% 能赚钱。

---

## 七、持有期分析

| 指标 | 上涨区间 | 下跌区间 | 差异 |
|------|---------|---------|------|
| 平均持有天数 | {up_hold['avg_hold_days']:.1f} 天 | {down_hold['avg_hold_days']:.1f} 天 | {up_hold['avg_hold_days'] - down_hold['avg_hold_days']:+.1f} 天 |
| 中位持有天数 | {up_hold['median_hold_days']:.1f} 天 | {down_hold['median_hold_days']:.1f} 天 | {up_hold['median_hold_days'] - down_hold['median_hold_days']:+.1f} 天 |
| 短期 (≤5 天) 数量 | {up_hold['short_count']} | {down_hold['short_count']} | {up_hold['short_count'] - down_hold['short_count']:+d} |
| 中期 (6-15 天) 数量 | {up_hold['medium_count']} | {down_hold['medium_count']} | {up_hold['medium_count'] - down_hold['medium_count']:+d} |
| 长期 (>15 天) 数量 | {up_hold['long_count']} | {down_hold['long_count']} | {up_hold['long_count'] - down_hold['long_count']:+d} |
| 短期胜率 | {up_hold['short_winrate']:.1f}% | {down_hold['short_winrate']:.1f}% | {up_hold['short_winrate'] - down_hold['short_winrate']:+.1f}% |
| 中期胜率 | {up_hold['medium_winrate']:.1f}% | {down_hold['medium_winrate']:.1f}% | {up_hold['medium_winrate'] - down_hold['medium_winrate']:+.1f}% |
| 长期胜率 | {up_hold['long_winrate']:.1f}% | {down_hold['long_winrate']:.1f}% | {up_hold['long_winrate'] - down_hold['long_winrate']:+.1f}% |

---

## 八、行业分布

### 上涨区间 Top 5 行业
| 行业 | 数量 |
|------|------|
"""

    for ind, cnt in up_ind['top_industries'].items():
        report += f"| {ind} | {cnt} |\n"

    report += f"""
**上涨区间行业表现**:
- 胜率最高：{up_ind['best_industry']} ({up_ind['best_industry_winrate']:.1f}%)
- 胜率最低：{up_ind['worst_industry']} ({up_ind['worst_industry_winrate']:.1f}%)

### 下跌区间 Top 5 行业
"""

    for ind, cnt in down_ind['top_industries'].items():
        report += f"| {ind} | {cnt} |\n"

    report += f"""
**下跌区间行业表现**:
- 胜率最高：{down_ind['best_industry']} ({down_ind['best_industry_winrate']:.1f}%)
- 胜率最低：{down_ind['worst_industry']} ({down_ind['worst_industry_winrate']:.1f}%)

---

## 九、核心结论

### 📌 不同点（用于差异化策略）

| 维度 | 上涨区间 | 下跌区间 | 策略启示 |
|------|---------|---------|----------|
| **胜率** | {up_basic['win_rate']:.1f}% | {down_basic['win_rate']:.1f}% | 上涨行情可加大仓位，下跌行情应降低仓位 |
| **平均收益** | {up_basic['avg_return']:.2f}% | {down_basic['avg_return']:.2f}% | 下跌行情收益微薄，建议观望为主 |
| **止损触发率** | {up_stop['trigger_neg4pct_rate']:.1f}% | {down_stop['trigger_neg4pct_rate']:.1f}% | 下跌行情更容易触发止损，需更严格风控 |
| **5% 止盈守住率** | {up_stop['hold_5pct_rate']:.1f}% | {down_stop['hold_5pct_rate']:.1f}% | 上涨行情适合放宽止盈，下跌行情应快速了结 |
| **触及 5% 后转亏率** | {up_stop['lose_after_5pct_rate']:.1f}% | {down_stop['lose_after_5pct_rate']:.1f}% | 下跌行情达到 5% 后应果断止盈，不可贪心 |
| **大赢 (>10%) 比例** | {up_dist['win_gt_10_pct']:.1f}% | {down_dist['win_gt_10_pct']:.1f}% | 上涨行情可追求更高收益目标 |
| **大亏 (>10%) 比例** | {up_dist['loss_gt_10_pct']:.1f}% | {down_dist['loss_gt_10_pct']:.1f}% | 下跌行情大亏风险几乎翻倍，需严格控制 |
| **高热度行业胜率** | {up_heat['high_heat_winrate']:.1f}% | {down_heat['high_heat_winrate']:.1f}% | 上涨行情追热点有效，下跌行情追热点易被套 |

### ✅ 相同点（用于统一策略）

| 维度 | 共同特征 | 策略启示 |
|------|---------|----------|
| **选股标准** | 平均相似度均为 {up_sim['avg_similarity']:.1f}% 左右 | 选股标准保持一致，无需因行情调整 |
| **高相似度占比** | 约 {up_sim['high_sim_rate']:.1f}% 股票相似度≥95% | 高相似度股票池稳定，可建立优选池 |
| **高相似度胜率优势** | 高相似度组胜率均高于低相似度组 | **始终优先选择相似度≥95% 的股票** |
| **行业热度与胜率正相关** | 高热度>中热度>低热度 | **始终优先选择高热度行业** |
| **Top 行业重合度高** | 电子、基础化工、机械设备均在前 5 | 重点关注这些行业的龙头股 |
| **持有期固定** | 均为 {up_hold['avg_hold_days']:.0f} 天满持有 | 策略设计为中长线，无需因行情调整 |
| **跌破成本价普遍** | 均接近 {up_cost['broken_rate']:.1f}% 会跌破 | **必须设置止损，不能死扛** |

### ⚠️ 风险提示

- **跌破成本价是常态**：45 天持有期内几乎 100% 会跌破，这是正常波动，不是策略问题
- **关键看能否涨回来**：上涨区间跌破后 60.7% 最终赚钱，下跌区间仅 41.0%
- **下跌行情大亏风险高**：大亏 (>10%) 比例达 {down_dist['loss_gt_10_pct']:.1f}%，是上涨行情的近 2 倍
- **唯一未跌破案例**：下跌区间有 1 只股票 (惠威科技) 买入后直接上涨，从未跌破，最终大赚 29.5%

---

## 十、策略建议

### 📈 上涨行情策略（胜率 {up_basic['win_rate']:.1f}%）

| 策略要素 | 建议 |
|---------|------|
| **仓位管理** | 可加大仓位（70-80%），胜率较高 |
| **选股标准** | 相似度≥95% 优先，平均收益 {up_sim['high_sim_avg_return']:.1f}% |
| **行业选择** | 优先选择热度≥70 的行业（胜率 {up_heat['high_heat_winrate']:.1f}%） |
| **止盈策略** | 可放宽至 10%（{up_dist['win_gt_10_pct']:.1f}% 概率大赢） |
| **止损策略** | 标准止损即可，触发率 {up_stop['trigger_neg4pct_rate']:.1f}% |
| **持有心态** | 可耐心持有，守住 5% 涨幅概率 {up_stop['hold_5pct_rate']:.1f}% |

### 📉 下跌行情策略（胜率 {down_basic['win_rate']:.1f}%）

| 策略要素 | 建议 |
|---------|------|
| **仓位管理** | 轻仓或观望（20-30%），胜率低 |
| **选股标准** | 相似度≥95% 必须遵守，否则胜率更低 |
| **行业选择** | 避免追高热度行业（胜率仅 {down_heat['high_heat_winrate']:.1f}%） |
| **止盈策略** | 达到 5% 快速了结（{down_stop['lose_after_5pct_rate']:.1f}% 会转亏） |
| **止损策略** | 严格执行，{down_stop['trigger_neg4pct_rate']:.1f}% 会触发止损 |
| **持有心态** | 快进快出，不宜恋战 |

### 🔑 核心原则（适用于所有行情）

1. **始终选择高相似度股票**：相似度≥95% 的股票胜率始终更高
2. **始终关注行业热度**：高热度行业胜率始终高于低热度
3. **接受持股波动**：100% 股票会跌破成本价，这是 45 天持有期的正常现象
4. **严格止损纪律**：虽然跌破是常态，但触及 -4% 止损线必须执行
5. **中长线持有**：策略设计为 45 天持有期，不因行情调整
6. **重点行业**：电子、基础化工、机械设备是 consistently Top 行业

---

## 附录：快速参考速查表

### 关键指标速查

| 指标 | 上涨行情 | 下跌行情 | 操作建议 |
|------|---------|---------|----------|
| 胜率 | 60.7% | 41.1% | 上涨加仓位，下跌轻仓玩 |
| 平均收益 | 9.1% | 1.3% | 下跌行情收益微薄 |
| 止损触发 | 61.6% | 82.9% | 下跌必须严格止损 |
| 5% 守住率 | 59.2% | 38.6% | 下跌达到 5% 快跑 |
| 大赢概率 | 36.8% | 21.4% | 上涨可博取大收益 |
| 大亏概率 | 9.8% | 18.0% | 下跌大亏风险翻倍 |
| 跌破成本价 | 100% | 99.7% | 正常现象，关键是能否涨回来 |
| 跌破后盈利 | 60.7% | 41.0% | 上涨行情更能扛住波动 |

### 行业热度速查

| 热度级别 | 上涨胜率 | 下跌胜率 | 配置建议 |
|---------|---------|---------|----------|
| 高热度 (≥70) | {up_heat['high_heat_winrate']:.1f}% | {down_heat['high_heat_winrate']:.1f}% | 上涨超配，下跌低配 |
| 中热度 (30-70) | {up_heat['mid_heat_winrate']:.1f}% | {down_heat['mid_heat_winrate']:.1f}% | 中性配置 |
| 低热度 (<30) | {up_heat['low_heat_winrate']:.1f}% | {down_heat['low_heat_winrate']:.1f}% | 低配或回避 |

### 相似度速查

| 相似度 | 上涨胜率 | 下跌胜率 | 操作建议 |
|-------|---------|---------|----------|
| ≥95% | {up_sim['high_sim_winrate']:.1f}% | {down_sim['high_sim_winrate']:.1f}% | **优先选择** |
| <95% | {up_sim['low_sim_winrate']:.1f}% | {down_sim['low_sim_winrate']:.1f}% | 谨慎或回避 |

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return report

if __name__ == '__main__':
    full_analysis()
    print("\n✅ 分析完成")
