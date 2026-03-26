#!/usr/bin/env python3
"""
对比分析不同市场行情下的股票表现 - 全面对比版
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 文件路径
BULL_FILE = Path("backtest_results/daily_backtest_fast_readable_20260325_100414.csv")  # 上涨行情
BEAR_FILE = Path("backtest_results/daily_backtest_fast_readable_20260325_101306.csv")  # 下跌行情

def load_data(filepath):
    """加载数据"""
    df = pd.read_csv(filepath)
    # 转换百分比列为数值
    for col in ['涨跌幅', '最大涨幅']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip('%').astype(float)
    # 转换相似度
    if '相似度' in df.columns:
        df['相似度_数值'] = df['相似度'].astype(str).str.rstrip('%').astype(float)
    return df

def basic_stats(df, market_type):
    """基本统计分析"""
    stats = {}
    stats['market_type'] = market_type
    stats['total_trades'] = len(df)
    stats['avg_return'] = df['涨跌幅'].mean()
    stats['median_return'] = df['涨跌幅'].median()
    stats['std_return'] = df['涨跌幅'].std()
    stats['max_return'] = df['涨跌幅'].max()
    stats['min_return'] = df['涨跌幅'].min()
    stats['win_count'] = (df['涨跌幅'] > 0).sum()
    stats['win_rate'] = stats['win_count'] / stats['total_trades'] * 100
    stats['avg_max_gain'] = df['最大涨幅'].mean()
    stats['max_max_gain'] = df['最大涨幅'].max()
    return stats

def trigger_analysis(df, market_type):
    """止盈止损触发分析"""
    stats = {}

    # 各止盈级别触发
    stats['reach_5pct'] = (df['最大涨幅'] >= 5.0).sum()
    stats['reach_5pct_rate'] = stats['reach_5pct'] / len(df) * 100
    stats['reach_10pct'] = (df['最大涨幅'] >= 10.0).sum()
    stats['reach_10pct_rate'] = stats['reach_10pct'] / len(df) * 100

    # 止损触发
    stats['trigger_neg4pct'] = (df['触发 -4% 日期'] != '-').sum()
    stats['trigger_neg4pct_rate'] = stats['trigger_neg4pct'] / len(df) * 100

    # 5% 止盈后走势
    reach_5pct_df = df[df['最大涨幅'] >= 5.0]
    if len(reach_5pct_df) > 0:
        stats['hold_5pct'] = (reach_5pct_df['涨跌幅'] >= 5.0).sum()
        stats['hold_5pct_rate'] = stats['hold_5pct'] / len(reach_5pct_df) * 100
        stats['lose_after_5pct'] = (reach_5pct_df['涨跌幅'] < 0).sum()
        stats['lose_after_5pct_rate'] = stats['lose_after_5pct'] / len(reach_5pct_df) * 100
    else:
        stats['hold_5pct'] = 0
        stats['hold_5pct_rate'] = 0
        stats['lose_after_5pct'] = 0
        stats['lose_after_5pct_rate'] = 0

    return stats

def similarity_analysis(df, market_type):
    """相似度分析"""
    stats = {}

    stats['avg_similarity'] = df['相似度_数值'].mean()
    stats['median_similarity'] = df['相似度_数值'].median()

    # 按相似度分组
    df_high = df[df['相似度_数值'] >= 95]
    df_low = df[df['相似度_数值'] < 95]

    stats['high_sim_count'] = len(df_high)
    stats['high_sim_rate'] = len(df_high) / len(df) * 100
    if len(df_high) > 0:
        stats['high_sim_winrate'] = (df_high['涨跌幅'] > 0).sum() / len(df_high) * 100
        stats['high_sim_avg_return'] = df_high['涨跌幅'].mean()
    else:
        stats['high_sim_winrate'] = 0
        stats['high_sim_avg_return'] = 0

    if len(df_low) > 0:
        stats['low_sim_winrate'] = (df_low['涨跌幅'] > 0).sum() / len(df_low) * 100
        stats['low_sim_avg_return'] = df_low['涨跌幅'].mean()
    else:
        stats['low_sim_winrate'] = 0
        stats['low_sim_avg_return'] = 0

    return stats

def industry_heat_analysis(df, market_type):
    """行业热度分析"""
    stats = {}

    heat_col = '行业热度_买入日'
    if heat_col in df.columns:
        df_heat = df[df[heat_col].notna()]
        if len(df_heat) > 0:
            stats['avg_heat'] = df_heat[heat_col].mean()
            stats['median_heat'] = df_heat[heat_col].median()

            # 按热度分组
            low_heat = df_heat[df_heat[heat_col] < 30]
            mid_heat = df_heat[(df_heat[heat_col] >= 30) & (df_heat[heat_col] < 70)]
            high_heat = df_heat[df_heat[heat_col] >= 70]

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

def return_distribution(df, market_type):
    """收益率分布分析"""
    stats = {}

    stats['loss_gt_10'] = (df['涨跌幅'] < -10).sum()
    stats['loss_5_10'] = ((df['涨跌幅'] >= -10) & (df['涨跌幅'] < -5)).sum()
    stats['loss_2_5'] = ((df['涨跌幅'] >= -5) & (df['涨跌幅'] < -2)).sum()
    stats['win_2_5'] = ((df['涨跌幅'] >= 2) & (df['涨跌幅'] < 5)).sum()
    stats['win_5_10'] = ((df['涨跌幅'] >= 5) & (df['涨跌幅'] < 10)).sum()
    stats['win_gt_10'] = (df['涨跌幅'] >= 10).sum()

    # 转换为比例
    total = len(df)
    for key in list(stats.keys()):
        stats[key + '_pct'] = stats[key] / total * 100

    return stats

def print_comparison(bull, bear, label, higher_is_better=True):
    """打印对比结果"""
    bull_val = bull.get(label, 0)
    bear_val = bear.get(label, 0)
    diff = bull_val - bear_val

    if isinstance(bull_val, float):
        bull_str = f"{bull_val:.2f}"
        bear_str = f"{bear_val:.2f}"
    else:
        bull_str = f"{bull_val}"
        bear_str = f"{bear_val}"

    if diff > 0:
        diff_str = f"+{diff:.2f}" if isinstance(diff, float) else f"+{diff}"
        symbol = "↑" if higher_is_better else "↓"
    elif diff < 0:
        diff_str = f"{diff:.2f}" if isinstance(diff, float) else f"{diff}"
        symbol = "↓" if higher_is_better else "↑"
    else:
        diff_str = "0"
        symbol = "="

    print(f"  {label:<25} | {bull_str:>12} | {bear_str:>12} | {diff_str:>10} {symbol}")

def full_comparison():
    """完整对比分析"""
    print("="*90)
    print("📊 不同市场行情下的股票表现对比分析")
    print("="*90)

    # 加载数据
    print(f"\n加载数据:")
    print(f"  上涨行情：{BULL_FILE}")
    print(f"  下跌行情：{BEAR_FILE}")

    bull_df = load_data(BULL_FILE)
    bear_df = load_data(BEAR_FILE)

    print(f"\n数据加载完成:")
    print(f"  上涨行情：{len(bull_df)} 只股票")
    print(f"  下跌行情：{len(bear_df)} 只股票")

    # 基本统计
    print("\n" + "="*90)
    print("📈 一、基本统计对比")
    print("="*90)
    print(f"\n{'指标':<25} | {'上涨行情':>12} | {'下跌行情':>12} | {'差异':>10}")
    print("-"*70)

    bull_basic = basic_stats(bull_df, "bull")
    bear_basic = basic_stats(bear_df, "bear")

    for label in ['total_trades', 'avg_return', 'median_return', 'std_return',
                  'max_return', 'min_return', 'win_rate', 'avg_max_gain', 'max_max_gain']:
        print_comparison(bull_basic, bear_basic, label)

    # 止盈止损触发
    print("\n" + "="*90)
    print("🎯 二、止盈止损触发对比")
    print("="*90)
    print(f"\n{'指标':<25} | {'上涨行情':>12} | {'下跌行情':>12} | {'差异':>10}")
    print("-"*70)

    bull_trigger = trigger_analysis(bull_df, "bull")
    bear_trigger = trigger_analysis(bear_df, "bear")

    for label in ['reach_5pct_rate', 'reach_10pct_rate', 'trigger_neg4pct_rate',
                  'hold_5pct_rate', 'lose_after_5pct_rate']:
        print_comparison(bull_trigger, bear_trigger, label)

    # 相似度分析
    print("\n" + "="*90)
    print("🎯 三、相似度分析对比")
    print("="*90)
    print(f"\n{'指标':<25} | {'上涨行情':>12} | {'下跌行情':>12} | {'差异':>10}")
    print("-"*70)

    bull_sim = similarity_analysis(bull_df, "bull")
    bear_sim = similarity_analysis(bear_df, "bear")

    for label in ['avg_similarity', 'median_similarity', 'high_sim_rate',
                  'high_sim_winrate', 'high_sim_avg_return', 'low_sim_winrate']:
        print_comparison(bull_sim, bear_sim, label)

    # 行业热度分析
    print("\n" + "="*90)
    print("🔥 四、行业热度分析对比")
    print("="*90)
    print(f"\n{'指标':<25} | {'上涨行情':>12} | {'下跌行情':>12} | {'差异':>10}")
    print("-"*70)

    bull_heat = industry_heat_analysis(bull_df, "bull")
    bear_heat = industry_heat_analysis(bear_df, "bear")

    for label in ['avg_heat', 'median_heat', 'low_heat_winrate', 'mid_heat_winrate',
                  'high_heat_winrate', 'low_heat_5pct', 'mid_heat_5pct', 'high_heat_5pct']:
        print_comparison(bull_heat, bear_heat, label)

    # 收益率分布
    print("\n" + "="*90)
    print("📊 五、收益率分布对比")
    print("="*90)
    print(f"\n{'区间':<25} | {'上涨行情':>12} | {'下跌行情':>12} | {'差异':>10}")
    print("-"*70)

    bull_dist = return_distribution(bull_df, "bull")
    bear_dist = return_distribution(bear_df, "bear")

    distribution_labels = [
        ('loss_gt_10_pct', '大亏 (>10%)'),
        ('loss_5_10_pct', '中亏 (5-10%)'),
        ('loss_2_5_pct', '小亏 (2-5%)'),
        ('win_2_5_pct', '小赢 (2-5%)'),
        ('win_5_10_pct', '中赢 (5-10%)'),
        ('win_gt_10_pct', '大赢 (>10%)'),
    ]

    for key, label in distribution_labels:
        print_comparison(bull_dist, bear_dist, key)

    # 相同点分析
    print("\n" + "="*90)
    print("✅ 六、相同点分析")
    print("="*90)

    print("\n1. 相似度分布:")
    print(f"   上涨行情平均相似度：{bull_sim['avg_similarity']:.1f}%")
    print(f"   下跌行情平均相似度：{bear_sim['avg_similarity']:.1f}%")
    print(f"   → 两个市场行情下，选股相似度基本一致")

    print("\n2. 高相似度股票占比:")
    print(f"   上涨行情>=95% 占比：{bull_sim['high_sim_rate']:.1f}%")
    print(f"   下跌行情>=95% 占比：{bear_sim['high_sim_rate']:.1f}%")
    print(f"   → 两种行情下高相似度股票比例相近")

    print("\n3. 5% 止盈触发后守住的比例:")
    print(f"   上涨行情：{bull_trigger['hold_5pct_rate']:.1f}%")
    print(f"   下跌行情：{bear_trigger['hold_5pct_rate']:.1f}%")
    print(f"   → 达到 5% 后能守住的比例差异反映了市场波动性")

    # 核心结论
    print("\n" + "="*90)
    print("📌 七、核心结论")
    print("="*90)

    print(f"\n【差异点】")
    print(f"  1. 胜率差异：上涨 {bull_basic['win_rate']:.1f}% vs 下跌 {bear_basic['win_rate']:.1f}%")
    print(f"  2. 5% 触发率：上涨 {bull_trigger['reach_5pct_rate']:.1f}% vs 下跌 {bear_trigger['reach_5pct_rate']:.1f}%")
    print(f"  3. 止损触发率：上涨 {bull_trigger['trigger_neg4pct_rate']:.1f}% vs 下跌 {bear_trigger['trigger_neg4pct_rate']:.1f}%")
    print(f"  4. 大赢比例：上涨 {bull_dist['win_gt_10_pct']:.1f}% vs 下跌 {bear_dist['win_gt_10_pct']:.1f}%")

    print(f"\n【相同点】")
    print(f"  1. 选股标准一致：平均相似度均为 {bull_sim['avg_similarity']:.1f}% 左右")
    print(f"  2. 高相似度股票占比稳定：约 {bull_sim['high_sim_rate']:.1f}%")
    print(f"  3. 行业热度分布相似：平均热度上涨 {bull_heat['avg_heat']:.1f} vs 下跌 {bear_heat['avg_heat']:.1f}")

    # 策略建议
    print(f"\n【策略建议】")
    if bull_basic['win_rate'] > bear_basic['win_rate'] + 10:
        print(f"  ✓ 上涨行情明显更适合本策略，胜率高出 {bull_basic['win_rate'] - bear_basic['win_rate']:.1f}%")
    if bull_trigger['hold_5pct_rate'] > bear_trigger['hold_5pct_rate'] + 10:
        print(f"  ✓ 5% 止盈策略在上涨行情更有效，守住率高出 {bull_trigger['hold_5pct_rate'] - bear_trigger['hold_5pct_rate']:.1f}%")
    if bear_trigger['trigger_neg4pct_rate'] > 70:
        print(f"  ⚠ 下跌行情止损触发率高达 {bear_trigger['trigger_neg4pct_rate']:.1f}%，建议降低仓位或观望")

    return {
        'bull': {**bull_basic, **bull_trigger, **bull_sim, **bull_heat, **bull_dist},
        'bear': {**bear_basic, **bear_trigger, **bear_sim, **bear_heat, **bear_dist}
    }

if __name__ == '__main__':
    full_comparison()
    print("\n✅ 分析完成")
