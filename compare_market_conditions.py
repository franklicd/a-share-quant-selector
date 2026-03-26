#!/usr/bin/env python3
"""
对比分析不同市场行情下的股票表现
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 文件路径
BULL_FILE = Path("backtest_results/daily_backtest_fast_readable_20260325_083014.csv")  # 上涨行情
BEAR_FILE = Path("backtest_results/daily_backtest_fast_readable_20260325_084008.csv")  # 下跌行情

def load_data(filepath):
    """加载数据"""
    df = pd.read_csv(filepath)
    # 转换百分比列为数值
    for col in ['涨跌幅', '最大涨幅']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip('%').astype(float)
    return df

def analyze_performance(df, market_type):
    """分析股票表现"""
    print(f"\n{'='*70}")
    print(f"📊 {market_type}")
    print(f"{'='*70}")

    # 基本统计
    total_trades = len(df)
    print(f"\n📈 基本统计")
    print(f"  总交易数：{total_trades}")

    # 收益率分析
    avg_return = df['涨跌幅'].mean()
    median_return = df['涨跌幅'].median()
    std_return = df['涨跌幅'].std()
    max_return = df['涨跌幅'].max()
    min_return = df['涨跌幅'].min()

    print(f"\n💰 收益率统计")
    print(f"  平均收益：{avg_return:.2f}%")
    print(f"  中位收益：{median_return:.2f}%")
    print(f"  标准差：{std_return:.2f}%")
    print(f"  最佳收益：{max_return:.2f}%")
    print(f"  最差收益：{min_return:.2f}%")

    # 胜率分析
    win_count = (df['涨跌幅'] > 0).sum()
    win_rate = win_count / total_trades * 100
    print(f"\n🎯 胜率分析")
    print(f"  盈利交易：{win_count} ({win_rate:.1f}%)")
    print(f"  亏损交易：{total_trades - win_count} ({100-win_rate:.1f}%)")

    # 收益率分布
    print(f"\n📊 收益率分布")
    loss_gt_10 = (df['涨跌幅'] < -10).sum()
    loss_5_10 = ((df['涨跌幅'] >= -10) & (df['涨跌幅'] < -5)).sum()
    loss_2_5 = ((df['涨跌幅'] >= -5) & (df['涨跌幅'] < -2)).sum()
    win_2_5 = ((df['涨跌幅'] >= 2) & (df['涨跌幅'] < 5)).sum()
    win_5_10 = ((df['涨跌幅'] >= 5) & (df['涨跌幅'] < 10)).sum()
    win_gt_10 = (df['涨跌幅'] >= 10).sum()

    print(f"  大亏 (>10%): {loss_gt_10} ({loss_gt_10/total_trades*100:.1f}%)")
    print(f"  中亏 (5-10%): {loss_5_10} ({loss_5_10/total_trades*100:.1f}%)")
    print(f"  小亏 (2-5%): {loss_2_5} ({loss_2_5/total_trades*100:.1f}%)")
    print(f"  小赢 (2-5%): {win_2_5} ({win_2_5/total_trades*100:.1f}%)")
    print(f"  中赢 (5-10%): {win_5_10} ({win_5_10/total_trades*100:.1f}%)")
    print(f"  大赢 (>10%): {win_gt_10} ({win_gt_10/total_trades*100:.1f}%)")

    # 最大涨幅分析
    print(f"\n🚀 最大涨幅统计")
    avg_max_gain = df['最大涨幅'].mean()
    max_max_gain = df['最大涨幅'].max()
    print(f"  平均最大涨幅：{avg_max_gain:.2f}%")
    print(f"  最高最大涨幅：{max_max_gain:.2f}%")

    # 曾跌破成本价分析
    if '是否曾跌破成本价' in df.columns:
        below_cost = (df['是否曾跌破成本价'] == '是').sum()
        print(f"\n📉 跌破成本价分析")
        print(f"  曾跌破成本价：{below_cost} ({below_cost/total_trades*100:.1f}%)")
        print(f"  未跌破成本价：{total_trades - below_cost} ({100-below_cost/total_trades*100:.1f}%)")

        # 跌破成本价的交易最终收益率
        below_cost_df = df[df['是否曾跌破成本价'] == '是']
        above_cost_df = df[df['是否曾跌破成本价'] != '是']
        if len(below_cost_df) > 0:
            print(f"  跌破成本价的平均收益：{below_cost_df['涨跌幅'].mean():.2f}%")
        if len(above_cost_df) > 0:
            print(f"  未跌破成本价的平均收益：{above_cost_df['涨跌幅'].mean():.2f}%")

    # 行业热度分析（如果有）
    if '行业热度_买入日' in df.columns:
        print(f"\n🔥 行业热度统计")
        avg_heat = df['行业热度_买入日'].mean()
        median_heat = df['行业热度_买入日'].median()
        print(f"  平均行业热度：{avg_heat:.2f}")
        print(f"  中位行业热度：{median_heat:.2f}")

        # 按热度分组
        heat_bins = [0, 30, 50, 70, 100]
        heat_labels = ['低热度 (0-30)', '中低热度 (30-50)', '中高热度 (50-70)', '高热度 (70-100)']
        df_valid = df[df['行业热度_买入日'].notna()].copy()
        if len(df_valid) > 0:
            df_valid['热度分组'] = pd.cut(df_valid['行业热度_买入日'], bins=heat_bins, labels=heat_labels)
            heat_perf = df_valid.groupby('热度分组', observed=True)['涨跌幅'].agg(['mean', 'count'])
            print(f"\n  不同热度下的表现:")
            for idx, row in heat_perf.iterrows():
                print(f"    {idx}: 平均收益 {row['mean']:.2f}% (样本数 {int(row['count'])})")

    # 行业表现
    if '行业' in df.columns:
        print(f"\n🏭 行业表现 TOP5 / BOTTOM5")
        industry_perf = df.groupby('行业')['涨跌幅'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(f"  表现最好的行业:")
        for idx, row in industry_perf.head(5).iterrows():
            print(f"    {idx}: {row['mean']:.2f}% (样本数 {int(row['count'])})")
        print(f"  表现最差的行业:")
        for idx, row in industry_perf.tail(5).iterrows():
            print(f"    {idx}: {row['mean']:.2f}% (样本数 {int(row['count'])})")

    # 持有天数分析
    if '持有天数' in df.columns:
        print(f"\n⏱️ 持有天数统计")
        avg_hold = df['持有天数'].mean()
        median_hold = df['持有天数'].median()
        print(f"  平均持有天数：{avg_hold:.1f}天")
        print(f"  中位持有天数：{median_hold:.1f}天")

    return {
        'total_trades': total_trades,
        'avg_return': avg_return,
        'median_return': median_return,
        'std_return': std_return,
        'win_rate': win_rate,
        'avg_max_gain': avg_max_gain,
        'below_cost_pct': below_cost/total_trades*100 if '是否曾跌破成本价' in df.columns else None
    }


def compare_analysis(bull_stats, bear_stats):
    """对比分析"""
    print(f"\n{'='*70}")
    print(f"📊 对比分析总结")
    print(f"{'='*70}")

    print(f"\n{'指标':<20} | {'上涨行情':<15} | {'下跌行情':<15} | {'差异':<10}")
    print(f"{'-'*70}")

    metrics = [
        ('总交易数', 'total_trades', '{:.0f}'),
        ('平均收益 (%)', 'avg_return', '{:.2f}'),
        ('中位收益 (%)', 'median_return', '{:.2f}'),
        ('收益标准差 (%)', 'std_return', '{:.2f}'),
        ('胜率 (%)', 'win_rate', '{:.1f}'),
        ('平均最大涨幅 (%)', 'avg_max_gain', '{:.2f}'),
    ]

    for label, key, fmt in metrics:
        bull_val = bull_stats.get(key, 0)
        bear_val = bear_stats.get(key, 0)
        diff = bull_val - bear_val
        diff_str = f"+{fmt.format(diff)}" if diff > 0 else fmt.format(diff)
        print(f"{label:<20} | {fmt.format(bull_val):<15} | {fmt.format(bear_val):<15} | {diff_str:<10}")

    if bull_stats.get('below_cost_pct') and bear_stats.get('below_cost_pct'):
        bull_bc = bull_stats['below_cost_pct']
        bear_bc = bear_stats['below_cost_pct']
        diff = bull_bc - bear_bc
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        print(f"跌破成本价比例 (%) | {bull_bc:<15.1f} | {bear_bc:<15.1f} | {diff_str:<10}")

    # 结论
    print(f"\n📌 核心结论:")

    if bull_stats['avg_return'] > bear_stats['avg_return']:
        print(f"  ✓ 上涨行情平均收益高出 {bull_stats['avg_return'] - bear_stats['avg_return']:.2f}%")
    else:
        print(f"  ✓ 下跌行情平均收益高出 {bear_stats['avg_return'] - bull_stats['avg_return']:.2f}%")

    if bull_stats['win_rate'] > bear_stats['win_rate']:
        print(f"  ✓ 上涨行情胜率高出 {bull_stats['win_rate'] - bear_stats['win_rate']:.1f}%")
    else:
        print(f"  ✓ 下跌行情胜率高出 {bear_stats['win_rate'] - bull_stats['win_rate']:.1f}%")

    if bull_stats['std_return'] > bear_stats['std_return']:
        print(f"  ✓ 上涨行情波动更大 (标准差高 {bull_stats['std_return'] - bear_stats['std_return']:.2f}%)")
    else:
        print(f"  ✓ 下跌行情波动更大 (标准差高 {bear_stats['std_return'] - bull_stats['std_return']:.2f}%)")


if __name__ == '__main__':
    print("="*70)
    print("📊 不同市场行情下的股票表现对比分析")
    print("="*70)

    # 加载数据
    print(f"\n加载数据...")
    print(f"  上涨行情：{BULL_FILE}")
    print(f"  下跌行情：{BEAR_FILE}")

    bull_df = load_data(BULL_FILE)
    bear_df = load_data(BEAR_FILE)

    # 分析
    bull_stats = analyze_performance(bull_df, "📈 上涨行情 (2025-09-09 选股)")
    bear_stats = analyze_performance(bear_df, "📉 下跌行情 (2025-09-05 选股)")

    # 对比
    compare_analysis(bull_stats, bear_stats)

    print(f"\n✅ 分析完成")
