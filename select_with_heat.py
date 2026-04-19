#!/usr/bin/env python3
"""
选股 + 行业热度分析（独立脚本，不修改原有 main.py）

功能：
1. 执行 B1 完美图形匹配选股
2. 为每只入选股票计算行业和行业热度
3. 按相似度 + 行业热度综合排序输出

使用方法:
    python select_with_heat.py

参数:
    --min-similarity 70    # 最小相似度阈值（默认 60）
    --lookback-days 30     # B1 匹配回看天数（默认 25）
    --max-stocks 500       # 限制处理股票数量（快速测试）
    --M-days 35            # 碗口反弹策略回溯天数
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import QuantSystem
from enhance_stock_selection import enhance_with_industry_heat, print_enhanced_results

# 从配置读取默认参数
try:
    from strategy.pattern_config import MIN_SIMILARITY_SCORE, DEFAULT_LOOKBACK_DAYS, TOP_N_RESULTS
except ImportError:
    MIN_SIMILARITY_SCORE = 60.0
    DEFAULT_LOOKBACK_DAYS = 45
    TOP_N_RESULTS = 10


def main():
    parser = argparse.ArgumentParser(
        description='选股 + 行业热度分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python select_with_heat.py                      # 使用默认参数
  python select_with_heat.py --min-similarity 70  # 提高相似度阈值到 70%
  python select_with_heat.py --lookback-days 30   # B1 匹配使用 30 天回看
  python select_with_heat.py --max-stocks 500     # 只处理前 500 只股票（快速测试）
  python select_with_heat.py --M-days 35          # 碗口策略使用 35 天回溯
        '''
    )

    parser.add_argument(
        '--min-similarity',
        type=float,
        default=None,
        help=f'B1 匹配最小相似度阈值 (默认：{MIN_SIMILARITY_SCORE})'
    )

    parser.add_argument(
        '--lookback-days',
        type=int,
        default=None,
        help=f'B1 匹配回看天数 (默认：{DEFAULT_LOOKBACK_DAYS})'
    )

    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='限制处理股票数量（用于快速测试）'
    )

    parser.add_argument(
        '--M-days',
        type=int,
        default=None,
        dest='M_days',
        help='碗口反弹策略的回溯天数 M'
    )

    parser.add_argument(
        '--no-enhance',
        action='store_true',
        help='跳过行业热度增强，仅执行普通选股'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='配置文件路径 (默认：config/config.yaml)'
    )

    args = parser.parse_args()

    # 确定参数
    min_similarity = args.min_similarity if args.min_similarity is not None else MIN_SIMILARITY_SCORE
    lookback_days = args.lookback_days if args.lookback_days is not None else DEFAULT_LOOKBACK_DAYS

    print("=" * 70)
    print("🎯 A 股量化选股系统 - 行业热度增强版")
    print("=" * 70)
    print(f"选股策略：B1 完美图形匹配 + 碗口反弹")
    print(f"相似度阈值：≥{min_similarity}%")
    print(f"回看天数：{lookback_days}天")
    if args.max_stocks:
        print(f"快速测试：只处理前 {args.max_stocks} 只股票")
    if args.M_days:
        print(f"碗口回溯天数：{args.M_days}天 (命令行覆盖)")
    print("=" * 70)

    # 创建系统实例
    quant = QuantSystem(args.config)

    # 1. 更新数据（智能更新：3 点前不更新）
    print("\n[1/4] 更新数据...")
    quant._smart_update(max_stocks=args.max_stocks)

    # 2. 执行选股 + B1 匹配
    print("\n[2/4] 执行选股 + B1 完美图形匹配...")
    result = quant.select_with_b1_match(
        category='all',
        max_stocks=args.max_stocks,
        min_similarity=min_similarity,
        lookback_days=lookback_days,
        M_days=args.M_days
    )

    matched = result.get('matched', [])
    total_selected = result.get('total_selected', 0)

    if not matched:
        print("\n⚠️ 没有符合 B1 匹配条件的股票")
        print(f"   提示：可以降低相似度阈值试试，如 --min-similarity {min_similarity - 10}")
        return

    print(f"\n✓ 选出 {total_selected} 只股票，其中 {len(matched)} 只符合 B1 匹配条件")

    if args.no_enhance:
        # 跳过行业热度增强，直接输出
        print("\n[跳过行业热度增强]")
        print_enhanced_results(matched, top_n=TOP_N_RESULTS)
        return

    # 3. 重建 stock_data_dict（用于计算行业热度）
    print("\n[3/4] 准备股票数据（用于行业热度计算）...")
    import pandas as pd
    from utils.csv_manager import CSVManager

    csv_manager = CSVManager(Path("data"))
    stock_data_dict = {}

    for r in matched:
        code = r['stock_code']
        df = csv_manager.read_stock(code)
        if not df.empty:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            stock_data_dict[code] = df

    print(f"✓ 加载 {len(stock_data_dict)} 只股票数据")

    # 4. 增强行业热度信息
    print("\n[4/4] 计算行业热度并排序...")
    enhanced = enhance_with_industry_heat(
        matched,
        stock_data_dict,
        "data"
    )

    # 5. 输出结果
    print_enhanced_results(enhanced, top_n=TOP_N_RESULTS)

    # 6. 保存结果到文件
    output_file = Path("backtest_results") / f"select_with_heat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"A 股选股结果 - 行业热度增强版\n")
        f.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"相似度阈值：≥{min_similarity}%\n")
        f.write(f"回看天数：{lookback_days}天\n")
        f.write(f"匹配股票数：{len(enhanced)}只\n")
        f.write("=" * 70 + "\n\n")

        for i, r in enumerate(enhanced, 1):
            heat = r.get('industry_heat')
            heat_str = f"{heat}分" if heat is not None else "N/A"
            f.write(f"{i}. {r['stock_code']} {r['stock_name']}\n")
            f.write(f"   相似度：{r['similarity_score']}% | 行业：{r['industry']} ({heat_str})\n")
            f.write(f"   匹配：{r['matched_case']} ({r['matched_date']})\n")
            f.write(f"   策略：{r['category']} | 价格：{r['close']} | J 值：{r['J']}\n\n")

    print(f"\n💾 结果已保存到：{output_file}")

    # 7. 显示决策建议
    print("\n" + "=" * 70)
    print("💡 决策建议")
    print("=" * 70)

    # 找出高相似度 + 高热度的股票
    high_quality = [r for r in enhanced
                   if r['similarity_score'] >= 90 and
                   (r['industry_heat'] is None or r['industry_heat'] >= 70)]

    if high_quality:
        print(f"\n✅ 强烈推荐 ({len(high_quality)}只): 高相似度 (≥90%) + 高热度 (≥70 分)")
        for r in high_quality[:5]:
            print(f"   - {r['stock_code']} {r['stock_name']}: "
                  f"相似度{r['similarity_score']}%, 行业热度{r['industry_heat']}分")

    # 找出需要谨慎的股票
    low_quality = [r for r in enhanced
                  if r['similarity_score'] < 75 or
                  (r['industry_heat'] is not None and r['industry_heat'] < 40)]

    if low_quality:
        print(f"\n⚠️ 谨慎关注 ({len(low_quality)}只): 低相似度 (<75%) 或 冷门行业 (<40 分)")
        for r in low_quality[:5]:
            heat_str = f"{r['industry_heat']}分" if r['industry_heat'] is not None else "N/A"
            print(f"   - {r['stock_code']} {r['stock_name']}: "
                  f"相似度{r['similarity_score']}%, 行业热度{heat_str}")

    print("\n" + "-" * 70)
    print("提示：以上建议仅供参考，请结合其他因素综合判断")
    print("=" * 70)


if __name__ == '__main__':
    main()
