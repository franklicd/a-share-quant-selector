#!/usr/bin/env python3
"""
对比分析不同市场行情下的股票表现 - 加入相似度分析
策略：+10% 止盈，-4% 止损
分析维度：相似度 + 行业热度 对止盈止损的影响
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

    # 提取相似度数值（去掉%）
    if '相似度' in df.columns:
        df['相似度_数值'] = df['相似度'].astype(str).str.rstrip('%').astype(float)

    return df

def analyze_by_similarity_heat(df, market_type, stop_profit=10.0, stop_loss=-4.0):
    """按相似度和行业热度分组分析"""
    print(f"\n{'='*70}")
    print(f"📊 {market_type} - 相似度 + 行业热度 分析")
    print(f"{'='*70}")

    total = len(df)

    # 相似度分组
    df_valid = df[df['相似度_数值'].notna() & df['行业热度_买入日'].notna()].copy()

    if len(df_valid) == 0:
        print("  数据不足，跳过分析")
        return None

    # 相似度分组 (按分位数)
    try:
        df_valid['相似度分组'] = pd.qcut(
            df_valid['相似度_数值'],
            q=4,
            labels=['低相似度 (0-25%)', '中低相似度 (25-50%)', '中高相似度 (50-75%)', '高相似度 (75-100%)'],
            duplicates='drop'
        )
    except ValueError:
        df_valid['相似度分组'] = pd.cut(
            df_valid['相似度_数值'],
            bins=4,
            labels=['低相似度 (0-25%)', '中低相似度 (25-50%)', '中高相似度 (50-75%)', '高相似度 (75-100%)']
        )

    # 行业热度分组
    heat_bins = [0, 30, 50, 70, 100]
    heat_labels = ['低热度 (0-30)', '中低热度 (30-50)', '中高热度 (50-70)', '高热度 (70-100)']
    df_valid['热度分组'] = pd.cut(df_valid['行业热度_买入日'], bins=heat_bins, labels=heat_labels)

    print(f"\n📈 基本统计")
    print(f"  有效样本数：{len(df_valid)} / {total}")

    # 1. 按相似度分组分析止盈止损
    print(f"\n📊 按相似度分组分析")
    print(f"  {'相似度分组':<20} | {'样本数':>8} | {'止盈率':>10} | {'止损率':>10} | {'净胜率':>10}")
    print(f"  {'-'*65}")

    sim_stats = []
    for sim_group in df_valid['相似度分组'].cat.categories:
        group_df = df_valid[df_valid['相似度分组'] == sim_group]
        n = len(group_df)
        profit_rate = (group_df['最大涨幅'] >= stop_profit).sum() / n * 100
        loss_rate = (group_df['触发 -4% 日期'] != '-').sum() / n * 100
        net_rate = profit_rate - loss_rate
        sim_stats.append({
            '分组': str(sim_group),
            '样本数': n,
            '止盈率': profit_rate,
            '止损率': loss_rate,
            '净胜率': net_rate
        })
        print(f"  {str(sim_group):<20} | {n:>8} | {profit_rate:>9.1f}% | {loss_rate:>9.1f}% | {net_rate:>+9.1f}%")

    # 2. 按行业热度分组分析止盈止损
    print(f"\n📊 按行业热度分组分析")
    print(f"  {'热度分组':<20} | {'样本数':>8} | {'止盈率':>10} | {'止损率':>10} | {'净胜率':>10}")
    print(f"  {'-'*65}")

    heat_stats = []
    for heat_group in df_valid['热度分组'].cat.categories:
        group_df = df_valid[df_valid['热度分组'] == heat_group]
        n = len(group_df)
        profit_rate = (group_df['最大涨幅'] >= stop_profit).sum() / n * 100
        loss_rate = (group_df['触发 -4% 日期'] != '-').sum() / n * 100
        net_rate = profit_rate - loss_rate
        heat_stats.append({
            '分组': str(heat_group),
            '样本数': n,
            '止盈率': profit_rate,
            '止损率': loss_rate,
            '净胜率': net_rate
        })
        print(f"  {str(heat_group):<20} | {n:>8} | {profit_rate:>9.1f}% | {loss_rate:>9.1f}% | {net_rate:>+9.1f}%")

    # 3. 相似度 + 热度 组合分析 (9 宫格)
    print(f"\n📊 相似度 + 行业热度 组合分析 (止盈率 - 止损率)")
    print(f"  净胜率 = 止盈率 - 止损率，正值表示更容易止盈，负值表示更容易止损")
    print()

    # 只取前 3 个相似度和前 3 个热度（避免样本太少）
    sim_categories = df_valid['相似度分组'].cat.categories[:3].tolist()
    heat_categories = df_valid['热度分组'].cat.categories[:3].tolist()

    # 打印表头
    print(f"  {'相似度 \\ 热度':<18}", end='')
    for h in heat_categories:
        print(f" | {str(h)[:12]:>14}", end='')
    print()
    print(f"  {'-'*18}", end='')
    for _ in heat_categories:
        print(f"-+-{'-'*14}", end='')
    print()

    combo_stats = []
    for sim in sim_categories:
        print(f"  {str(sim)[:18]:<18}", end='')
        for heat in heat_categories:
            combo_df = df_valid[(df_valid['相似度分组'] == sim) & (df_valid['热度分组'] == heat)]
            n = len(combo_df)
            if n >= 5:  # 至少 5 个样本才显示
                profit_rate = (combo_df['最大涨幅'] >= stop_profit).sum() / n * 100
                loss_rate = (combo_df['触发 -4% 日期'] != '-').sum() / n * 100
                net_rate = profit_rate - loss_rate
                combo_stats.append({
                    '相似度': str(sim),
                    '热度': str(heat),
                    '样本数': n,
                    '止盈率': profit_rate,
                    '止损率': loss_rate,
                    '净胜率': net_rate
                })
                print(f" | {net_rate:>+6.1f}% ({n:>2})", end='')
            else:
                print(f" | {'N/A':>14}", end='')
        print()

    # 4. 找出最佳组合
    print(f"\n🏆 最佳组合 TOP5 (按净胜率排序)")
    if combo_stats:
        combo_df_stats = pd.DataFrame(combo_stats).sort_values('净胜率', ascending=False)
        for idx, row in combo_df_stats.head(5).iterrows():
            print(f"  {idx+1}. {row['相似度']} + {row['热度']}: 净胜率 {row['净胜率']:+.1f}% "
                  f"(止盈{row['止盈率']:.0f}%, 止损{row['止损率']:.0f}%, 样本{int(row['样本数'])})")

    # 5. 相关性分析
    print(f"\n📈 相关性分析")
    corr_cols = ['相似度_数值', '行业热度_买入日', '最大涨幅', '涨跌幅']
    corr_df = df_valid[corr_cols].copy()
    corr_matrix = corr_df.corr()

    print(f"  相似度与最大涨幅的相关性：{corr_matrix.loc['相似度_数值', '最大涨幅']:.3f}")
    print(f"  行业热度与最大涨幅的相关性：{corr_matrix.loc['行业热度_买入日', '最大涨幅']:.3f}")
    print(f"  相似度与行业热度的相关性：{corr_matrix.loc['相似度_数值', '行业热度_买入日']:.3f}")

    # 判断相关性方向
    sim_corr = corr_matrix.loc['相似度_数值', '最大涨幅']
    heat_corr = corr_matrix.loc['行业热度_买入日', '最大涨幅']

    if abs(sim_corr) > abs(heat_corr):
        print(f"  → 相似度对涨幅的预测力强于行业热度")
    else:
        print(f"  → 行业热度对涨幅的预测力强于相似度")

    return {
        'sim_stats': sim_stats,
        'heat_stats': heat_stats,
        'combo_stats': combo_stats,
        'sim_corr': sim_corr,
        'heat_corr': heat_corr
    }


def analyze_similarity_threshold(df, market_type, stop_profit=10.0, stop_loss=-4.0):
    """分析不同相似度阈值下的表现"""
    print(f"\n{'='*70}")
    print(f"📊 {market_type} - 相似度阈值分析")
    print(f"{'='*70}")

    df_valid = df[df['相似度_数值'].notna()].copy()

    thresholds = [85, 90, 92, 95, 97]

    print(f"\n  {'相似度阈值':>12} | {'样本数':>8} | {'止盈率':>10} | {'止损率':>10} | {'净胜率':>10} | {'胜率':>10}")
    print(f"  {'-'*70}")

    threshold_stats = []
    for thresh in thresholds:
        high_sim_df = df_valid[df_valid['相似度_数值'] >= thresh]
        n = len(high_sim_df)
        if n > 0:
            profit_rate = (high_sim_df['最大涨幅'] >= stop_profit).sum() / n * 100
            loss_rate = (high_sim_df['触发 -4% 日期'] != '-').sum() / n * 100
            net_rate = profit_rate - loss_rate
            # 胜率 = 止盈 / (止盈 + 止损)
            profit_count = (high_sim_df['最大涨幅'] >= stop_profit).sum()
            loss_count = (high_sim_df['触发 -4% 日期'] != '-').sum()
            win_rate = profit_count / (profit_count + loss_count) * 100 if (profit_count + loss_count) > 0 else 0
            threshold_stats.append({
                '阈值': thresh,
                '样本数': n,
                '止盈率': profit_rate,
                '止损率': loss_rate,
                '净胜率': net_rate,
                '胜率': win_rate
            })
            print(f"  >= {thresh:.0f}%        | {n:>8} | {profit_rate:>9.1f}% | {loss_rate:>9.1f}% | {net_rate:>+9.1f}% | {win_rate:>9.1f}%")

    return threshold_stats


def compare_threshold_analysis(bull_thresh, bear_thresh):
    """对比不同阈值下的表现"""
    print(f"\n{'='*70}")
    print(f"📊 相似度阈值对比 (上涨 vs 下跌)")
    print(f"{'='*70}")

    print(f"\n  {'阈值':>8} | {'上涨净胜率':>12} | {'下跌净胜率':>12} | {'上涨胜率':>10} | {'下跌胜率':>10}")
    print(f"  {'-'*65}")

    for bull_stat in bull_thresh:
        thresh = bull_stat['阈值']
        bear_stat = next((s for s in bear_thresh if s['阈值'] == thresh), None)
        if bear_stat:
            bull_net = bull_stat['净胜率']
            bear_net = bear_stat['净胜率']
            bull_win = bull_stat['胜率']
            bear_win = bear_stat['胜率']
            print(f"  >= {thresh:.0f}%    | {bull_net:>+11.1f}% | {bear_net:>+11.1f}% | {bull_win:>9.1f}% | {bear_win:>9.1f}%")


if __name__ == '__main__':
    print("="*70)
    print("📊 相似度 + 行业热度 对止盈止损的影响分析")
    print("="*70)

    # 加载数据
    print(f"\n加载数据...")
    print(f"  上涨行情：{BULL_FILE}")
    print(f"  下跌行情：{BEAR_FILE}")

    bull_df = load_data(BULL_FILE)
    bear_df = load_data(BEAR_FILE)

    # 1. 相似度 + 热度组合分析
    bull_combo = analyze_by_similarity_heat(bull_df, "📈 上涨行情", stop_profit=10.0, stop_loss=-4.0)
    bear_combo = analyze_by_similarity_heat(bear_df, "📉 下跌行情", stop_profit=10.0, stop_loss=-4.0)

    # 2. 相似度阈值分析
    bull_thresh = analyze_similarity_threshold(bull_df, "📈 上涨行情", stop_profit=10.0, stop_loss=-4.0)
    bear_thresh = analyze_similarity_threshold(bear_df, "📉 下跌行情", stop_profit=10.0, stop_loss=-4.0)

    # 3. 对比
    compare_threshold_analysis(bull_thresh, bear_thresh)

    # 4. 总结
    print(f"\n{'='*70}")
    print(f"📌 核心结论")
    print(f"{'='*70}")

    if bull_combo and bear_combo:
        print(f"\n1️⃣ 相似度与涨幅的相关性:")
        print(f"   上涨行情：{bull_combo['sim_corr']:.3f}")
        print(f"   下跌行情：{bear_combo['sim_corr']:.3f}")

        print(f"\n2️⃣ 行业热度与涨幅的相关性:")
        print(f"   上涨行情：{bull_combo['heat_corr']:.3f}")
        print(f"   下跌行情：{bear_combo['heat_corr']:.3f}")

        # 找出最佳组合
        if bull_combo['combo_stats']:
            bull_best = max(bull_combo['combo_stats'], key=lambda x: x['净胜率'])
            print(f"\n3️⃣ 上涨行情最佳组合:")
            print(f"   {bull_best['相似度']} + {bull_best['热度']}")
            print(f"   净胜率 {bull_best['净胜率']:+.1f}% (止盈{bull_best['止盈率']:.0f}%, 止损{bull_best['止损率']:.0f}%)")

        if bear_combo['combo_stats']:
            bear_best = max(bear_combo['combo_stats'], key=lambda x: x['净胜率'])
            print(f"\n4️⃣ 下跌行情最佳组合:")
            print(f"   {bear_best['相似度']} + {bear_best['热度']}")
            print(f"   净胜率 {bear_best['净胜率']:+.1f}% (止盈{bear_best['止盈率']:.0f}%, 止损{bear_best['止损率']:.0f}%)")

    print(f"\n✅ 分析完成")
