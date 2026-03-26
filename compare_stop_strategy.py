#!/usr/bin/env python3
"""
对比分析不同市场行情下的股票表现 - 止盈止损策略版
策略：+10% 止盈，-4% 止损，先触发者退出
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

def analyze_stop_strategy(df, market_type, stop_profit=10.0, stop_loss=-4.0):
    """分析止盈止损策略表现"""
    print(f"\n{'='*70}")
    print(f"📊 {market_type}")
    print(f"{'='*70}")
    print(f"策略：+{stop_profit}% 止盈，{stop_loss}% 止损")

    total_trades = len(df)

    # 分析每个交易的退出情况
    stop_profit_count = 0  # 触发止盈
    stop_loss_count = 0    # 触发止损
    neither_count = 0      # 都未触发
    both_count = 0         # 都触发（按先触发者计算）

    # 记录触发顺序
    profit_first = 0   # 先触发止盈
    loss_first = 0     # 先触发止损

    for idx, row in df.iterrows():
        max_gain = row['最大涨幅']
        final_return = row['涨跌幅']

        # 检查是否曾达到止盈线
        reached_profit = max_gain >= stop_profit
        # 检查是否曾达到止损线（最大涨幅为负或最终收益）
        # 需要看是否有触发 -4% 的记录
        neg4_triggered = row.get('触发 -4% 日期', '-') != '-'

        if reached_profit and neg4_triggered:
            # 都触发，判断谁先
            both_count += 1
            # 根据触发顺序判断
            trigger_order = row.get('触发顺序', '')
            if '10pct' in str(trigger_order):
                # 找到 10pct 和 neg4pct 的位置
                order_str = str(trigger_order)
                if order_str.find('10pct') < order_str.find('neg4pct'):
                    profit_first += 1
                    stop_profit_count += 1
                else:
                    loss_first += 1
                    stop_loss_count += 1
            else:
                # 无法判断，按最终结果
                if final_return >= stop_profit:
                    stop_profit_count += 1
                else:
                    stop_loss_count += 1
        elif reached_profit:
            stop_profit_count += 1
        elif neg4_triggered:
            stop_loss_count += 1
        else:
            # 都未触发，按最终收益判断
            neither_count += 1
            if final_return >= stop_profit:
                stop_profit_count += 1
            elif final_return <= stop_loss:
                stop_loss_count += 1

    # 实际止盈/止损次数（包含都触发的情况）
    actual_profit = stop_profit_count
    actual_loss = stop_loss_count

    # 计算策略胜率（止盈 / (止盈 + 止损)）
    triggered_total = actual_profit + actual_loss
    strategy_win_rate = actual_profit / triggered_total * 100 if triggered_total > 0 else 0

    # 止盈触发率 = 止盈数 / 总交易数
    profit_trigger_rate = actual_profit / total_trades * 100
    # 止损触发率 = 止损数 / 总交易数
    loss_trigger_rate = actual_loss / total_trades * 100

    print(f"\n📈 基本统计")
    print(f"  总交易数：{total_trades}")

    print(f"\n🎯 止盈止损触发情况")
    print(f"  触发止盈 (+{stop_profit}%): {actual_profit} ({profit_trigger_rate:.1f}%)")
    print(f"  触发止损 ({stop_loss}%): {actual_loss} ({loss_trigger_rate:.1f}%)")
    print(f"  都未触发：{neither_count} ({neither_count/total_trades*100:.1f}%)")
    print(f"  止盈止损都触发：{both_count} ({both_count/total_trades*100:.1f}%)")
    if both_count > 0:
        print(f"    - 先止盈后止损：{profit_first} ({profit_first/both_count*100:.1f}%)")
        print(f"    - 先止损后止盈：{loss_first} ({loss_first/both_count*100:.1f}%)")

    print(f"\n📊 策略表现")
    print(f"  策略胜率：{strategy_win_rate:.1f}%")
    print(f"  止盈触发率：{profit_trigger_rate:.1f}%")
    print(f"  止损触发率：{loss_trigger_rate:.1f}%")
    print(f"  盈亏比：{actual_profit/actual_loss:.2f}:1" if actual_loss > 0 else f"  盈亏比：N/A (无止损)")

    # 收益率分析
    avg_return = df['涨跌幅'].mean()
    median_return = df['涨跌幅'].median()
    print(f"\n💰 实际收益率统计")
    print(f"  平均收益：{avg_return:.2f}%")
    print(f"  中位收益：{median_return:.2f}%")

    # 按行业分析止盈触发率
    if '行业' in df.columns:
        print(f"\n🏭 行业止盈触发率 TOP5 / BOTTOM5")
        industry_stats = []
        for industry in df['行业'].unique():
            ind_df = df[df['行业'] == industry]
            ind_count = len(ind_df)
            ind_profit = (ind_df['最大涨幅'] >= stop_profit).sum()
            ind_loss = (ind_df['触发 -4% 日期'] != '-').sum()
            industry_stats.append({
                '行业': industry,
                '样本数': ind_count,
                '止盈触发率': ind_profit / ind_count * 100 if ind_count > 0 else 0,
                '止损触发率': ind_loss / ind_count * 100 if ind_count > 0 else 0
            })
        industry_df = pd.DataFrame(industry_stats)

        print(f"  止盈触发率最高:")
        for idx, row in industry_df.nlargest(5, '止盈触发率').iterrows():
            print(f"    {row['行业']}: {row['止盈触发率']:.1f}% (样本数 {int(row['样本数'])})")
        print(f"  止盈触发率最低:")
        for idx, row in industry_df.nsmallest(5, '止盈触发率').iterrows():
            print(f"    {row['行业']}: {row['止盈触发率']:.1f}% (样本数 {int(row['样本数'])})")

    # 按行业热度分析
    if '行业热度_买入日' in df.columns:
        print(f"\n🔥 不同热度下的止盈触发率")
        heat_bins = [0, 30, 50, 70, 100]
        heat_labels = ['低热度 (0-30)', '中低热度 (30-50)', '中高热度 (50-70)', '高热度 (70-100)']
        df_valid = df[df['行业热度_买入日'].notna()].copy()
        if len(df_valid) > 0:
            df_valid['热度分组'] = pd.cut(df_valid['行业热度_买入日'], bins=heat_bins, labels=heat_labels)
            for heat_label in heat_labels:
                heat_df = df_valid[df_valid['热度分组'] == heat_label]
                if len(heat_df) > 0:
                    profit_rate = (heat_df['最大涨幅'] >= stop_profit).sum() / len(heat_df) * 100
                    loss_rate = (heat_df['触发 -4% 日期'] != '-').sum() / len(heat_df) * 100
                    print(f"    {heat_label}: 止盈 {profit_rate:.1f}%, 止损 {loss_rate:.1f}% (样本数 {len(heat_df)})")

    return {
        'total_trades': total_trades,
        'stop_profit_count': actual_profit,
        'stop_loss_count': actual_loss,
        'strategy_win_rate': strategy_win_rate,
        'profit_trigger_rate': profit_trigger_rate,
        'loss_trigger_rate': loss_trigger_rate,
        'avg_return': avg_return,
        'median_return': median_return
    }


def compare_analysis(bull_stats, bear_stats, stop_profit=10.0, stop_loss=-4.0):
    """对比分析"""
    print(f"\n{'='*70}")
    print(f"📊 对比分析总结 (止盈{stop_profit}% / 止损{stop_loss}%)")
    print(f"{'='*70}")

    print(f"\n{'指标':<20} | {'上涨行情':<15} | {'下跌行情':<15} | {'差异':<10}")
    print(f"{'-'*70}")

    metrics = [
        ('总交易数', 'total_trades', '{:.0f}'),
        ('止盈触发数', 'stop_profit_count', '{:.0f}'),
        ('止损触发数', 'stop_loss_count', '{:.0f}'),
        ('策略胜率 (%)', 'strategy_win_rate', '{:.1f}'),
        ('止盈触发率 (%)', 'profit_trigger_rate', '{:.1f}'),
        ('止损触发率 (%)', 'loss_trigger_rate', '{:.1f}'),
        ('平均收益 (%)', 'avg_return', '{:.2f}'),
        ('中位收益 (%)', 'median_return', '{:.2f}'),
    ]

    for label, key, fmt in metrics:
        bull_val = bull_stats.get(key, 0)
        bear_val = bear_stats.get(key, 0)
        diff = bull_val - bear_val
        diff_str = f"+{fmt.format(diff)}" if diff > 0 else fmt.format(diff)
        print(f"{label:<20} | {fmt.format(bull_val):<15} | {fmt.format(bear_val):<15} | {diff_str:<10}")

    # 结论
    print(f"\n📌 核心结论:")

    if bull_stats['strategy_win_rate'] > bear_stats['strategy_win_rate']:
        print(f"  ✓ 上涨行情策略胜率高出 {bull_stats['strategy_win_rate'] - bear_stats['strategy_win_rate']:.1f}%")
    else:
        print(f"  ✓ 下跌行情策略胜率高出 {bear_stats['strategy_win_rate'] - bull_stats['strategy_win_rate']:.1f}%")

    if bull_stats['profit_trigger_rate'] > bear_stats['profit_trigger_rate']:
        print(f"  ✓ 上涨行情止盈触发率高出 {bull_stats['profit_trigger_rate'] - bear_stats['profit_trigger_rate']:.1f}%")
    else:
        print(f"  ✓ 下跌行情止盈触发率高出 {bear_stats['profit_trigger_rate'] - bull_stats['profit_trigger_rate']:.1f}%")

    if bull_stats['loss_trigger_rate'] < bear_stats['loss_trigger_rate']:
        print(f"  ✓ 上涨行情止损触发率低 {bear_stats['loss_trigger_rate'] - bull_stats['loss_trigger_rate']:.1f}%")
    else:
        print(f"  ⚠ 下跌行情止损触发率反而低 {bull_stats['loss_trigger_rate'] - bear_stats['loss_trigger_rate']:.1f}%")

    # 期望收益估算
    # 假设止盈赚 10%，止损亏 4%
    bull_expected = bull_stats['profit_trigger_rate'] * 10 / 100 + bull_stats['loss_trigger_rate'] * (-4) / 100
    bear_expected = bear_stats['profit_trigger_rate'] * 10 / 100 + bear_stats['loss_trigger_rate'] * (-4) / 100

    print(f"\n📈 策略期望收益估算 (按止盈 +10%, 止损 -4% 计算):")
    print(f"  上涨行情期望收益：{bull_expected:.2f}% 每笔")
    print(f"  下跌行情期望收益：{bear_expected:.2f}% 每笔")

    if bull_expected > 0:
        print(f"  ✓ 上涨行情策略期望为正，可执行")
    else:
        print(f"  ⚠ 上涨行情策略期望为负，需谨慎")

    if bear_expected > 0:
        print(f"  ✓ 下跌行情策略期望为正，可执行")
    else:
        print(f"  ⚠ 下跌行情策略期望为负，需谨慎")


if __name__ == '__main__':
    print("="*70)
    print("📊 止盈止损策略对比分析 (+10% 止盈，-4% 止损)")
    print("="*70)

    # 加载数据
    print(f"\n加载数据...")
    print(f"  上涨行情：{BULL_FILE}")
    print(f"  下跌行情：{BEAR_FILE}")

    bull_df = load_data(BULL_FILE)
    bear_df = load_data(BEAR_FILE)

    # 分析
    bull_stats = analyze_stop_strategy(bull_df, "📈 上涨行情", stop_profit=10.0, stop_loss=-4.0)
    bear_stats = analyze_stop_strategy(bear_df, "📉 下跌行情", stop_profit=10.0, stop_loss=-4.0)

    # 对比
    compare_analysis(bull_stats, bear_stats, stop_profit=10.0, stop_loss=-4.0)

    print(f"\n✅ 分析完成")
