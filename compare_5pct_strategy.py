#!/usr/bin/env python3
"""
对比分析不同市场行情下的股票表现 - 止盈止损策略版
策略：+10% 止盈，+5% 止盈，-4% 止损，先触发者退出
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

def analyze_stop_strategy(df, market_type, stop_profit_levels=[10.0, 5.0], stop_loss=-4.0):
    """分析止盈止损策略表现 - 支持多个止盈级别"""
    print(f"\n{'='*70}")
    print(f"📊 {market_type}")
    print(f"{'='*70}")
    print(f"策略：+{stop_profit_levels[0]}%/+{stop_profit_levels[1]}% 止盈，{stop_loss}% 止损")

    total_trades = len(df)

    # 分析每个交易的退出情况
    stats = {}
    for sp in stop_profit_levels:
        stats[f'stop_profit_{sp}'] = 0
    stats['stop_loss'] = 0
    stats['neither'] = 0

    # 记录触发顺序（用于分析 5% 止盈后是否反转）
    reach_5pct_then_10pct = 0  # 先达到 5% 后达到 10%
    reach_5pct_then_neg4pct = 0  # 先达到 5% 后达到 -4%
    reach_5pct_only = 0  # 只达到 5%，未达到 10%

    for idx, row in df.iterrows():
        max_gain = row['最大涨幅']
        final_return = row['涨跌幅']

        # 检查各止盈止损是否触发
        reached_10pct = max_gain >= stop_profit_levels[0]
        reached_5pct = max_gain >= stop_profit_levels[1]
        neg4_triggered = row.get('触发 -4% 日期', '-') != '-'
        trigger_5pct_date = row.get('触发 +5% 日期', '-')
        trigger_10pct_date = row.get('触发 +10% 日期', '-')
        trigger_neg4pct_date = row.get('触发 -4% 日期', '-')

        # 统计各止盈级别触发情况
        for sp in stop_profit_levels:
            if sp == 10.0:
                if reached_10pct:
                    stats[f'stop_profit_{sp}'] += 1
            elif sp == 5.0:
                if reached_5pct:
                    stats[f'stop_profit_{sp}'] += 1

        if neg4_triggered:
            stats['stop_loss'] += 1

        # 5% 止盈后的走势分析
        if reached_5pct:
            if reached_10pct:
                reach_5pct_then_10pct += 1  # 5% 后继续上涨到 10%
            else:
                reach_5pct_only += 1  # 只到 5%，未到 10%

            if neg4_triggered:
                # 检查触发顺序
                trigger_order = row.get('触发顺序', '')
                if '5pct' in str(trigger_order) and 'neg4pct' in str(trigger_order):
                    order_str = str(trigger_order)
                    if order_str.find('5pct') < order_str.find('neg4pct'):
                        reach_5pct_then_neg4pct += 1  # 先 5% 后 -4%（坐过山车）

    # 5% 止盈能全身而退的比例（达到 5% 且最终收益>=5%）
    exit_at_5pct = (df[(df['最大涨幅'] >= 5.0) & (df['涨跌幅'] >= 5.0)])
    exit_at_5pct_count = len(exit_at_5pct)

    # 5% 止盈但未能保住的（达到 5% 但最终收益<5%）
    not_hold_5pct = df[(df['最大涨幅'] >= 5.0) & (df['涨跌幅'] < 5.0)]
    not_hold_5pct_count = len(not_hold_5pct)

    # 5% 止盈但最终亏损的（达到 5% 但最终收益<0）
    loss_after_5pct = df[(df['最大涨幅'] >= 5.0) & (df['涨跌幅'] < 0)]
    loss_after_5pct_count = len(loss_after_5pct)

    # 5% 止盈但最终止损的（达到 5% 但最终触发 -4%）
    stop_loss_after_5pct = df[(df['最大涨幅'] >= 5.0) & (df['触发 -4% 日期'] != '-')]
    stop_loss_after_5pct_count = len(stop_loss_after_5pct)

    print(f"\n📈 基本统计")
    print(f"  总交易数：{total_trades}")

    print(f"\n🎯 各止盈级别触发情况")
    for sp in stop_profit_levels:
        count = stats[f'stop_profit_{sp}']
        rate = count / total_trades * 100
        print(f"  触发 +{sp}% 止盈：{count} ({rate:.1f}%)")
    print(f"  触发 {stop_loss}% 止损：{stats['stop_loss']} ({stats['stop_loss']/total_trades*100:.1f}%)")

    print(f"\n💰 5% 止盈策略深度分析")
    print(f"  曾达到 +5% 的交易数：{(df['最大涨幅'] >= 5.0).sum()} ({(df['最大涨幅'] >= 5.0).sum()/total_trades*100:.1f}%)")

    print(f"\n  📊 5% 止盈后走势分析:")
    print(f"    达到 5% 后继续冲到 10%: {reach_5pct_then_10pct} ({reach_5pct_then_10pct/total_trades*100:.1f}%)")
    print(f"    达到 5% 但未到 10%: {reach_5pct_only} ({reach_5pct_only/total_trades*100:.1f}%)")
    print(f"    达到 5% 后跌破 -4%（坐过山车）: {reach_5pct_then_neg4pct} ({reach_5pct_then_neg4pct/total_trades*100:.1f}%)")

    print(f"\n  💵 5% 止盈盈亏分布:")
    print(f"    达到 5% 且最终收益>=5%（全身而退）: {exit_at_5pct_count} ({exit_at_5pct_count/total_trades*100:.1f}%)")
    print(f"    达到 5% 但最终收益<5%（未能守住）: {not_hold_5pct_count} ({not_hold_5pct_count/total_trades*100:.1f}%)")
    print(f"    达到 5% 但最终亏损（盈利变亏损）: {loss_after_5pct_count} ({loss_after_5pct_count/total_trades*100:.1f}%)")
    print(f"    达到 5% 但最终触发 -4%（彻底坐过山车）: {stop_loss_after_5pct_count} ({stop_loss_after_5pct_count/total_trades*100:.1f}%)")

    # 5% 止盈策略期望收益（假设在 5% 时卖出）
    # 对于曾达到 5% 的交易，假设在 5% 时卖出
    hold_5pct_df = df[df['最大涨幅'] >= 5.0].copy()
    if len(hold_5pct_df) > 0:
        # 假设在 5% 时卖出，收益为 5%
        assumed_profit = len(hold_5pct_df) * 5.0
        # 未达到 5% 的交易，按实际收益计算
        rest_df = df[df['最大涨幅'] < 5.0]
        actual_loss = rest_df['涨跌幅'].sum() if len(rest_df) > 0 else 0
        total_assumed_return = assumed_profit + actual_loss
        avg_assumed_return = total_assumed_return / total_trades
        print(f"\n  📈 5% 止盈策略期望收益:")
        print(f"    假设收益（曾达 5% 按 5% 计）: +{assumed_profit:.1f}%")
        print(f"    实际收益（未达 5% 按实际）: {actual_loss:.1f}%")
        print(f"    总期望收益：{total_assumed_return:.1f}%")
        print(f"    平均每笔收益：{avg_assumed_return:.2f}%")

    # 收益率分析
    avg_return = df['涨跌幅'].mean()
    median_return = df['涨跌幅'].median()
    print(f"\n💰 实际收益率统计")
    print(f"  平均收益：{avg_return:.2f}%")
    print(f"  中位收益：{median_return:.2f}%")

    # 按行业分析 5% 止盈触发率
    if '行业' in df.columns:
        print(f"\n🏭 行业 5% 止盈触发率 TOP5 / BOTTOM5")
        industry_stats = []
        for industry in df['行业'].unique():
            ind_df = df[df['行业'] == industry]
            ind_count = len(ind_df)
            ind_5pct = (ind_df['最大涨幅'] >= 5.0).sum()
            ind_10pct = (ind_df['最大涨幅'] >= 10.0).sum()
            industry_stats.append({
                '行业': industry,
                '样本数': ind_count,
                '5% 触发率': ind_5pct / ind_count * 100 if ind_count > 0 else 0,
                '10% 触发率': ind_10pct / ind_count * 100 if ind_count > 0 else 0
            })
        industry_df = pd.DataFrame(industry_stats)

        print(f"  5% 触发率最高:")
        for idx, row in industry_df.nlargest(5, '5% 触发率').iterrows():
            print(f"    {row['行业']}: {row['5% 触发率']:.1f}% (10% 触发率 {row['10% 触发率']:.1f}%, 样本数 {int(row['样本数'])})")
        print(f"  5% 触发率最低:")
        for idx, row in industry_df.nsmallest(5, '5% 触发率').iterrows():
            print(f"    {row['行业']}: {row['5% 触发率']:.1f}% (10% 触发率 {row['10% 触发率']:.1f}%, 样本数 {int(row['样本数'])})")

    # 按行业热度分析
    if '行业热度_买入日' in df.columns:
        print(f"\n🔥 不同热度下的 5% 止盈触发率")
        heat_bins = [0, 30, 50, 70, 100]
        heat_labels = ['低热度 (0-30)', '中低热度 (30-50)', '中高热度 (50-70)', '高热度 (70-100)']
        df_valid = df[df['行业热度_买入日'].notna()].copy()
        if len(df_valid) > 0:
            df_valid['热度分组'] = pd.cut(df_valid['行业热度_买入日'], bins=heat_bins, labels=heat_labels)
            for heat_label in heat_labels:
                heat_df = df_valid[df_valid['热度分组'] == heat_label]
                if len(heat_df) > 0:
                    pct_5 = (heat_df['最大涨幅'] >= 5.0).sum() / len(heat_df) * 100
                    pct_10 = (heat_df['最大涨幅'] >= 10.0).sum() / len(heat_df) * 100
                    print(f"    {heat_label}: 5% 触发 {pct_5:.1f}%, 10% 触发 {pct_10:.1f}% (样本数 {len(heat_df)})")

    return {
        'total_trades': total_trades,
        'reach_5pct_count': (df['最大涨幅'] >= 5.0).sum(),
        'reach_10pct_count': (df['最大涨幅'] >= 10.0).sum(),
        'stop_loss_count': stats['stop_loss'],
        'exit_at_5pct_count': exit_at_5pct_count,
        'not_hold_5pct_count': not_hold_5pct_count,
        'loss_after_5pct_count': loss_after_5pct_count,
        'stop_loss_after_5pct_count': stop_loss_after_5pct_count,
        'reach_5pct_then_10pct': reach_5pct_then_10pct,
        'reach_5pct_then_neg4pct': reach_5pct_then_neg4pct,
        'avg_return': avg_return,
        'median_return': median_return
    }


def compare_5pct_analysis(bull_stats, bear_stats):
    """对比 5% 止盈分析"""
    print(f"\n{'='*70}")
    print(f"📊 5% 止盈策略对比分析 (上涨 vs 下跌)")
    print(f"{'='*70}")

    print(f"\n{'指标':<25} | {'上涨行情':<15} | {'下跌行情':<15} | {'差异':<10}")
    print(f"{'-'*75}")

    metrics = [
        ('总交易数', 'total_trades', '{:.0f}'),
        ('达到 5% 交易数', 'reach_5pct_count', '{:.0f}'),
        ('达到 10% 交易数', 'reach_10pct_count', '{:.0f}'),
        ('触发止损数', 'stop_loss_count', '{:.0f}'),
        ('5% 全身而退数', 'exit_at_5pct_count', '{:.0f}'),
        ('5% 未能守住数', 'not_hold_5pct_count', '{:.0f}'),
        ('5% 变亏损数', 'loss_after_5pct_count', '{:.0f}'),
        ('5% 后触发 -4%', 'stop_loss_after_5pct_count', '{:.0f}'),
        ('5% 后冲到 10%', 'reach_5pct_then_10pct', '{:.0f}'),
        ('5% 后跌到 -4%', 'reach_5pct_then_neg4pct', '{:.0f}'),
        ('平均收益 (%)', 'avg_return', '{:.2f}'),
        ('中位收益 (%)', 'median_return', '{:.2f}'),
    ]

    for label, key, fmt in metrics:
        bull_val = bull_stats.get(key, 0)
        bear_val = bear_stats.get(key, 0)
        diff = bull_val - bear_val
        diff_str = f"+{fmt.format(diff)}" if diff > 0 else fmt.format(diff)
        print(f"{label:<25} | {fmt.format(bull_val):<15} | {fmt.format(bear_val):<15} | {diff_str:<10}")

    # 计算比例
    bull_total = bull_stats['total_trades']
    bear_total = bear_stats['total_trades']

    print(f"\n📊 比例对比:")
    print(f"  5% 触发率：上涨 {bull_stats['reach_5pct_count']/bull_total*100:.1f}% vs 下跌 {bear_stats['reach_5pct_count']/bear_total*100:.1f}%")
    print(f"  5% 全身而退率：上涨 {bull_stats['exit_at_5pct_count']/bull_total*100:.1f}% vs 下跌 {bear_stats['exit_at_5pct_count']/bear_total*100:.1f}%")
    print(f"  5% 坐过山车率：上涨 {bull_stats['stop_loss_after_5pct_count']/bull_total*100:.1f}% vs 下跌 {bear_stats['stop_loss_after_5pct_count']/bear_total*100:.1f}%")

    # 结论
    print(f"\n📌 核心结论:")

    # 5% 止盈有效性
    bull_effective = bull_stats['exit_at_5pct_count'] / bull_stats['reach_5pct_count'] * 100 if bull_stats['reach_5pct_count'] > 0 else 0
    bear_effective = bear_stats['exit_at_5pct_count'] / bear_stats['reach_5pct_count'] * 100 if bear_stats['reach_5pct_count'] > 0 else 0

    print(f"  5% 止盈有效性（达到 5% 且守住）: 上涨 {bull_effective:.1f}% vs 下跌 {bear_effective:.1f}%")

    # 坐过山车风险
    bull_risk = bull_stats['stop_loss_after_5pct_count'] / bull_stats['reach_5pct_count'] * 100 if bull_stats['reach_5pct_count'] > 0 else 0
    bear_risk = bear_stats['stop_loss_after_5pct_count'] / bear_stats['reach_5pct_count'] * 100 if bear_stats['reach_5pct_count'] > 0 else 0

    print(f"  坐过山车风险（5% 后跌破 -4%）: 上涨 {bull_risk:.1f}% vs 下跌 {bear_risk:.1f}%")

    # 策略建议
    print(f"\n📈 策略建议:")
    if bull_effective > 50:
        print(f"  ✓ 上涨行情：5% 止盈策略有效，{bull_effective:.0f}% 的交易能守住盈利")
    else:
        print(f"  ⚠ 上涨行情：5% 止盈策略效果一般，仅 {bull_effective:.0f}% 的交易能守住盈利")

    if bear_risk > 30:
        print(f"  ⚠ 下跌行情：坐过山车风险高，{bear_risk:.0f}% 的 5% 盈利交易最终跌破 -4%")
    else:
        print(f"  ✓ 下跌行情：坐过山车风险可控")


if __name__ == '__main__':
    print("="*70)
    print("📊 5% 止盈策略对比分析")
    print("="*70)

    # 加载数据
    print(f"\n加载数据...")
    print(f"  上涨行情：{BULL_FILE}")
    print(f"  下跌行情：{BEAR_FILE}")

    bull_df = load_data(BULL_FILE)
    bear_df = load_data(BEAR_FILE)

    # 分析
    bull_stats = analyze_stop_strategy(bull_df, "📈 上涨行情", stop_profit_levels=[10.0, 5.0], stop_loss=-4.0)
    bear_stats = analyze_stop_strategy(bear_df, "📉 下跌行情", stop_profit_levels=[10.0, 5.0], stop_loss=-4.0)

    # 对比
    compare_5pct_analysis(bull_stats, bear_stats)

    print(f"\n✅ 分析完成")
