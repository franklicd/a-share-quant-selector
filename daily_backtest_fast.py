#!/usr/bin/env python3
"""
每日回测脚本 - B1 完美图形匹配版 (优化版)

功能：
1. 支持输入任意日期范围
2. 使用原有 backtest.py 的优化逻辑（指标缓存）
3. 输出详细分析报告

使用方法:
    python daily_backtest_fast.py --start 2025-10-01 --end 2025-12-30
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.csv_manager import CSVManager
from utils.technical import MA, EMA, LLV, HHV, REF, EXIST, KDJ, calculate_zhixing_trend
from strategy.pattern_feature_extractor import PatternFeatureExtractor
from strategy.pattern_matcher import PatternMatcher
from strategy.pattern_config import SIMILARITY_WEIGHTS, B1_PERFECT_CASES
from strategy.bowl_rebound import BowlReboundStrategy


# 构建案例查找表：case_id -> 案例信息
CASE_INFO = {case["id"]: case for case in B1_PERFECT_CASES}


class FastDailyBacktestConfig:
    """快速每日回测配置"""
    def __init__(self, start_date, end_date, top_n=10, hold_days=30, cap_threshold=4e9):
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.hold_days = hold_days
        self.cap_threshold = cap_threshold


class FastDailyBacktestStrategy(BowlReboundStrategy):
    """快速每日回测策略 - 复用碗口反弹策略的优化逻辑"""

    def __init__(self, params=None):
        super().__init__(params)
        self.feature_extractor = PatternFeatureExtractor()
        self.matcher = PatternMatcher(SIMILARITY_WEIGHTS)
        self.cases = {}  # B1 案例库

    def _build_case_library(self, csv_manager):
        """构建 B1 案例库"""
        print("  正在构建 B1 完美图形案例库...")

        for case in B1_PERFECT_CASES:
            try:
                df = csv_manager.read_stock(case["code"])
                if df.empty:
                    continue

                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                breakout_dt = pd.to_datetime(case["breakout_date"])
                mask = df['date'] < breakout_dt
                window_df = df[mask].head(case.get("lookback_days", 25)).reset_index(drop=True)

                if window_df.empty or len(window_df) < 10:
                    continue

                window_df = window_df.sort_values('date').reset_index(drop=True)
                features = self.feature_extractor.extract(window_df)

                self.cases[case["id"]] = {
                    "meta": case,
                    "features": features,
                }
            except Exception as e:
                continue

        print(f"  ✓ 案例库加载完成：{len(self.cases)} 个案例")

    def rank_stocks_single_date(self, stock_data_dict, target_date, precomputed_indicators=None):
        """
        对单日进行选股排名

        :param stock_data_dict: 股票数据字典 {code: (name, df)}
        :param target_date: 选股日期
        :param precomputed_indicators: 预计算的指标字典（可选）
        """
        if not self.cases:
            self._build_case_library(self.csv_manager)

        candidates = []
        target_pd = pd.to_datetime(target_date)

        for code, (name, df) in stock_data_dict.items():
            if df.empty:
                continue

            # 使用截至 target_date 的历史数据（保持倒序，最新在前）
            historical_df = df[df['date'] <= target_pd].copy().reset_index(drop=True)

            if len(historical_df) < 25:
                continue

            # 找最接近 target_date 的日期（数据是倒序的，head(1) 是最大的日期，即最接近 target_date 的）
            actual_date = historical_df['date'].iloc[0]
            if pd.isna(actual_date):
                continue

            # 获取最近 25 天数据（倒序，最新在前）
            candidate_df = historical_df.head(25).reset_index(drop=True)

            if len(candidate_df) < 10:
                continue

            # 检查是否符合碗口反弹策略条件
            try:
                # 检查是否已经预计算过指标
                cache_key = f'bounce_indicators_{self.params["M1"]}_{self.params["M2"]}_{self.params["M3"]}_{self.params["M4"]}'

                if cache_key in candidate_df.attrs:
                    indicators_df = candidate_df.attrs[cache_key]
                else:
                    indicators_df = self.calculate_indicators(candidate_df)

                # 取第一天（target_date，最新的数据）的指标
                latest = indicators_df.iloc[0]

                if not latest.get('trend_above', False):
                    continue
                if not latest.get('j_low', False):
                    continue
            except Exception:
                continue

            # 提取特征并匹配（按日期正序排列）
            candidate_df_sorted = candidate_df.sort_values('date').reset_index(drop=True)
            candidate_features = self.feature_extractor.extract(candidate_df_sorted)

            best_score = 0
            best_match = None
            best_match_date = None

            for case_id, case_data in self.cases.items():
                result = self.matcher.match(candidate_features, case_data["features"])
                similarity = result.get('total_score', 0) if isinstance(result, dict) else result
                if similarity > best_score:
                    best_score = similarity
                    best_match = case_id
                    best_match_date = case_data["meta"]["breakout_date"]

            candidates.append({
                'code': code,
                'name': name,
                'actual_date': str(actual_date.date()) if hasattr(actual_date, 'date') else str(actual_date),
                'similarity_score': best_score,
                'best_match_case': best_match,
                'best_match_date': str(best_match_date) if best_match_date else None,
            })

        # 按相似度降序排列
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

        # 添加排名和分类信息
        for i, cand in enumerate(candidates):
            cand['rank'] = i + 1
            # 获取匹配案例的中文信息
            case_info = CASE_INFO.get(cand['best_match_case'], {})
            cand['category'] = case_info.get('tags', ['未知'])[0] if case_info else '未知'  # 使用 tags 第一个作为类型
            cand['match_description'] = case_info.get('description', '')  # 匹配案例的描述
            cand['reasons'] = [f"匹配{cand['best_match_case']}: {case_info.get('name', '')} - {cand['match_description']}"] if cand['best_match_case'] else []

        return candidates


class FastDailyBacktester:
    """快速每日回测执行器"""

    def __init__(self, config):
        self.config = config
        self.csv_manager = CSVManager(Path(__file__).parent / 'data')
        self.strategy = FastDailyBacktestStrategy()
        self.strategy.csv_manager = self.csv_manager
        self.stock_names = {}

    def _load_stock_names(self):
        """加载股票名称"""
        names_file = Path(__file__).parent / 'data' / 'stock_names.json'
        if names_file.exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                self.stock_names = json.load(f)

    def _get_trading_dates(self):
        """获取日期范围内的交易日（从缓存数据中提取）"""
        # 从任意股票获取所有交易日期
        stock_codes = self._get_all_stock_codes()
        all_dates = set()

        for code in stock_codes[:100]:  # 取前 100 只股票
            df = self.csv_manager.read_stock(code)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                all_dates.update(df['date'].dt.date.tolist())

        # 过滤日期范围
        start = self.config.start_date
        end = self.config.end_date
        trading_dates = sorted([d for d in all_dates if start <= d <= end])

        return trading_dates

    def _calculate_return_and_max_gain(self, code, buy_date, hold_days, cached_df=None):
        """计算持有期收益率和期间最大涨幅"""
        df = cached_df if cached_df is not None else self.csv_manager.read_stock(code)
        if df is None or df.empty:
            return None, None, None, None, None, None

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 找到买入日数据
        buy_mask = df['date'].dt.date == buy_date
        buy_rows = df[buy_mask]

        if buy_rows.empty:
            buy_row = df[df['date'] <= pd.to_datetime(buy_date)].head(1)
            if buy_row.empty:
                return None, None, None, None, None, None
            buy_price = buy_row['close'].iloc[0]
            actual_buy_date = buy_row['date'].iloc[0].date()
        else:
            buy_price = buy_rows['close'].iloc[0]
            actual_buy_date = buy_date

        # 找到卖出日
        sell_date = buy_date + timedelta(days=hold_days)
        sell_mask = df['date'].dt.date == sell_date
        sell_rows = df[sell_mask]

        if sell_rows.empty:
            # 找最接近的卖出日（数据是倒序的，所以用 tail(1) 取最小的大于等于 sell_date 的日期）
            sell_df = df[df['date'] >= pd.to_datetime(sell_date)]
            if sell_df.empty:
                return None, None, None, None, None, None
            sell_price = sell_df['close'].iloc[-1]  # tail(1) = 最小的大于等于 sell_date 的日期
            actual_sell_date = sell_df['date'].iloc[-1].date()
        else:
            sell_price = sell_rows['close'].iloc[0]
            actual_sell_date = sell_date

        return_pct = (sell_price - buy_price) / buy_price * 100

        # 计算最大涨幅（修复：重置索引确保连续，使用 iloc 避免索引跳跃问题）
        df_sorted = df.sort_values('date').reset_index(drop=True)
        buy_mask = df_sorted['date'].dt.date == actual_buy_date
        sell_mask = df_sorted['date'].dt.date == actual_sell_date

        if buy_mask.any() and sell_mask.any():
            start_pos = buy_mask.argmax()  # 获取买入位置
            end_pos = sell_mask.argmax()   # 获取卖出位置
            period_df = df_sorted.iloc[start_pos:end_pos+1]  # 包含两端
            max_high = period_df['high'].max()
            max_gain_pct = (max_high - buy_price) / buy_price * 100

            max_high_row = period_df[period_df['high'] == max_high]
            if len(max_high_row) > 0:
                max_high_date = max_high_row['date'].iloc[0].date()
                max_gain_day = (max_high_date - actual_buy_date).days
            else:
                max_gain_day = 0
        else:
            max_gain_pct = return_pct
            max_gain_day = hold_days

        return buy_price, sell_price, return_pct, actual_sell_date, max_gain_pct, max_gain_day

    def run_backtest(self):
        """执行每日回测（优化版）"""
        print("=" * 95)
        print("📊 每日回测报告 - B1 完美图形匹配版 (优化版)")
        print("=" * 95)
        print(f"回测参数:")
        print(f"  - 日期范围：{self.config.start_date} 至 {self.config.end_date}")
        print(f"  - 每次选股：Top {self.config.top_n}")
        print(f"  - 持有期：{self.config.hold_days}天")
        print(f"  - 市值门槛：{self.config.cap_threshold / 1e8:.0f}亿")
        print("=" * 95)

        # 获取交易日（而不是所有日期）
        trading_dates = self._get_trading_dates()
        print(f"\n交易日数量：{len(trading_dates)}天")

        # 预加载股票数据
        print("\n预加载所有股票数据到缓存...")
        stock_codes = self._get_all_stock_codes()
        stock_data_dict = {}
        for i, code in enumerate(stock_codes):
            df = self.csv_manager.read_stock(code)
            if df is not None and not df.empty:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                name = self.stock_names.get(code, '未知')
                stock_data_dict[code] = (name, df)
            if (i + 1) % 1000 == 0:
                print(f"  已缓存 {i + 1}/{len(stock_codes)}...")
        print(f"✓ 缓存完成，共 {len(stock_data_dict)} 只股票\n")

        all_results = []
        daily_summary = []

        for idx, sel_date in enumerate(trading_dates, 1):
            print(f"[{idx}/{len(trading_dates)}] 选股日：{sel_date}")

            # 选股
            ranked = self.strategy.rank_stocks_single_date(stock_data_dict, sel_date)

            if not ranked:
                print(f"  ⚠️ 该日期未选出任何股票")
                continue

            top_stocks = ranked[:self.config.top_n]
            print(f"  ✓ 选出 Top {len(top_stocks)} 只股票")

            day_results = []
            for stock in top_stocks:
                actual_date = datetime.strptime(stock['actual_date'], '%Y-%m-%d').date()
                buy_price, sell_price, return_pct, sell_date, max_gain_pct, max_gain_day = \
                    self._calculate_return_and_max_gain(
                        stock['code'], actual_date, self.config.hold_days,
                        stock_data_dict.get(stock['code'], (None, None))[1]
                    )

                if buy_price is None:
                    continue

                day_results.append({
                    'selection_date': str(sel_date),
                    'code': stock['code'],
                    'name': stock['name'],
                    'rank': stock['rank'],
                    'similarity_score': stock['similarity_score'],
                    'best_match_case': stock['best_match_case'],
                    'best_match_date': stock['best_match_date'],
                    'category': stock['category'],
                    'reasons': stock['reasons'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return_pct': return_pct,
                    'max_gain_pct': max_gain_pct,
                    'max_gain_day': max_gain_day,
                    'sell_date': str(sell_date) if sell_date else '',
                    'hold_days': self.config.hold_days
                })

            if day_results:
                all_results.extend(day_results)

                day_return = sum(r['return_pct'] for r in day_results) / len(day_results)
                win_count = sum(1 for r in day_results if r['return_pct'] > 0)
                win_rate = win_count / len(day_results) * 100

                daily_summary.append({
                    'date': str(sel_date),
                    'count': len(day_results),
                    'win_count': win_count,
                    'win_rate': win_rate,
                    'avg_return': day_return
                })

                print(f"  平均收益：{day_return:+.2f}%, 胜率：{win_rate:.1f}%")

        self._save_results(all_results, daily_summary)
        return all_results, daily_summary

    def _get_all_stock_codes(self):
        """获取所有股票代码"""
        return self.csv_manager.list_all_stocks()

    def print_report(self, all_results, daily_summary):
        """打印回测报告"""
        print("\n" + "=" * 95)
        print("📈 回测结果汇总")
        print("=" * 95)

        if not all_results:
            print("❌ 无有效数据")
            return

        total_trades = len(all_results)
        wins = sum(1 for r in all_results if r['return_pct'] > 0)
        win_rate = wins / total_trades * 100
        avg_return = sum(r['return_pct'] for r in all_results) / total_trades

        print(f"\n总交易数：{total_trades}")
        print(f"盈利：{wins} ({win_rate:.1f}%)")
        print(f"亏损：{total_trades - wins} ({100 - win_rate:.1f}%)")
        print(f"平均收益：{avg_return:+.2f}%")

        if daily_summary:
            best_day = max(daily_summary, key=lambda x: x['avg_return'])
            worst_day = min(daily_summary, key=lambda x: x['avg_return'])
            print(f"\n最佳交易日：{best_day['date']} ({best_day['avg_return']:+.2f}%)")
            print(f"最差交易日：{worst_day['date']} ({worst_day['avg_return']:+.2f}%)")

        best_trade = max(all_results, key=lambda x: x['return_pct'])
        worst_trade = min(all_results, key=lambda x: x['return_pct'])
        print(f"\n最佳交易：{best_trade['name']} ({best_trade['code']}) {best_trade['return_pct']:+.2f}%")
        print(f"最差交易：{worst_trade['name']} ({worst_trade['code']}) {worst_trade['return_pct']:+.2f}%")

    def _save_results(self, all_results, daily_summary):
        """保存回测结果"""
        output_dir = Path(__file__).parent / 'backtest_results'
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        json_file = output_dir / f'daily_backtest_fast_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'start_date': str(self.config.start_date),
                    'end_date': str(self.config.end_date),
                    'top_n': self.config.top_n,
                    'hold_days': self.config.hold_days
                },
                'summary': {
                    'total_trades': len(all_results),
                    'win_rate': sum(1 for r in all_results if r['return_pct'] > 0) / len(all_results) * 100 if all_results else 0,
                    'avg_return': sum(r['return_pct'] for r in all_results) / len(all_results) if all_results else 0
                },
                'daily_summary': daily_summary,
                'all_results': all_results
            }, f, ensure_ascii=False, indent=2)

        csv_file = output_dir / f'daily_backtest_fast_{timestamp}.csv'
        df = pd.DataFrame(all_results)
        df.to_csv(csv_file, index=False)

        simple_rows = []
        for r in all_results:
            # 获取匹配案例的中文信息
            case_info = CASE_INFO.get(r['best_match_case'], {}) if r['best_match_case'] else {}
            case_name = case_info.get('name', '')
            case_desc = case_info.get('description', '')
            simple_rows.append({
                '选股日期': r['selection_date'],
                '排名': r['rank'],
                '代码': r['code'],
                '名称': r['name'],
                '相似度': f"{r['similarity_score']:.1f}%",
                '匹配案例': f"{r['best_match_case']} ({case_name})" if r['best_match_case'] else '-',
                '匹配案例描述': case_desc if case_desc else '-',
                '匹配日期': r['best_match_date'] or '-',
                '类型': r['category'] if r['category'] else '-',
                '推荐理由': r['reasons'][0] if r['reasons'] else '-',
                '买入价': r['buy_price'],
                '卖出价': r['sell_price'],
                '涨跌幅': f"{r['return_pct']:+.2f}%",
                '最大涨幅': f"{r['max_gain_pct']:+.2f}%" if r.get('max_gain_pct') is not None else 'N/A',
                '最大涨幅天数': f"第{r['max_gain_day']}天" if r.get('max_gain_day') is not None else 'N/A',
                '卖出日期': r['sell_date'],
                '持有天数': r['hold_days']
            })

        simple_csv = output_dir / f'daily_backtest_fast_readable_{timestamp}.csv'
        df_simple = pd.DataFrame(simple_rows)
        df_simple.to_csv(simple_csv, index=False, encoding='utf-8-sig')

        print(f"\n💾 结果已保存:")
        print(f"  - JSON: {json_file}")
        print(f"  - CSV:  {csv_file}")
        print(f"  - 可读 CSV: {simple_csv}")


def main():
    parser = argparse.ArgumentParser(description='每日快速回测 - B1 完美图形匹配版')
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=10, help='每次选股数量 (默认：10)')
    parser.add_argument('--hold-days', type=int, default=30, help='持有天数 (默认：30)')
    parser.add_argument('--cap', type=float, default=40, help='市值门槛 (单位：亿，默认：40)')

    args = parser.parse_args()

    config = FastDailyBacktestConfig(
        start_date=datetime.strptime(args.start, '%Y-%m-%d').date(),
        end_date=datetime.strptime(args.end, '%Y-%m-%d').date(),
        top_n=args.top_n,
        hold_days=args.hold_days,
        cap_threshold=args.cap * 1e8
    )

    backtester = FastDailyBacktester(config)
    backtester._load_stock_names()
    all_results, daily_summary = backtester.run_backtest()
    backtester.print_report(all_results, daily_summary)


if __name__ == '__main__':
    main()
