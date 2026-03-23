#!/usr/bin/env python3
"""
选股模型回测脚本 - B1 完美图形匹配版

功能：
1. 过去 12 个月，每月 1 日作为选股日
2. 每次用截至选股日的历史数据重新跑选股
3. 使用 B1 完美图形匹配相似度进行排名
4. 选前 10 只股票，追踪 30 日后表现
5. 输出详细清单，包含排名、相似度、匹配案例

使用方法:
    python backtest.py
    python backtest.py --months 6
    python backtest.py --top-n 10
    python backtest.py --hold-days 30
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.csv_manager import CSVManager
from utils.technical import MA, EMA, LLV, HHV, REF, EXIST, KDJ, calculate_zhixing_trend
from strategy.pattern_feature_extractor import PatternFeatureExtractor
from strategy.pattern_matcher import PatternMatcher
from strategy.pattern_config import SIMILARITY_WEIGHTS, B1_PERFECT_CASES


class BacktestConfig:
    """回测配置"""
    def __init__(self, months=12, top_n=10, hold_days=30, cap_threshold=4e9):
        self.months = months
        self.top_n = top_n
        self.hold_days = hold_days
        self.cap_threshold = cap_threshold


class B1BacktestStrategy:
    """B1 完美图形匹配策略"""

    def __init__(self, params=None):
        self.params = params or {
            'N': 4, 'M': 15, 'CAP': 4e9, 'J_VAL': 30,
            'duokong_pct': 3, 'short_pct': 2,
            'M1': 14, 'M2': 28, 'M3': 57, 'M4': 114
        }
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

                # 提取突破日期前的数据
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                breakout_dt = pd.to_datetime(case["breakout_date"])
                mask = df['date'] < breakout_dt
                window_df = df[mask].head(case.get("lookback_days", 25)).reset_index(drop=True)

                if window_df.empty or len(window_df) < 10:
                    continue

                # 按日期正序排列
                window_df = window_df.sort_values('date').reset_index(drop=True)

                # 提取特征
                features = self.feature_extractor.extract(window_df)

                self.cases[case["id"]] = {
                    "meta": case,
                    "features": features,
                }
            except Exception as e:
                continue

        print(f"  ✓ 案例库加载完成：{len(self.cases)} 个案例")

    def calculate_indicators(self, df):
        """计算技术指标"""
        result = df.copy()

        trend_df = calculate_zhixing_trend(
            result,
            m1=self.params['M1'], m2=self.params['M2'],
            m3=self.params['M3'], m4=self.params['M4']
        )
        result['short_term_trend'] = trend_df['short_term_trend']
        result['bull_bear_line'] = trend_df['bull_bear_line']
        result['trend_above'] = result['short_term_trend'] > result['bull_bear_line']

        duokong_pct = self.params['duokong_pct'] / 100
        short_pct = self.params['short_pct'] / 100

        result['fall_in_bowl'] = (
            (result['close'] >= result['bull_bear_line']) &
            (result['close'] <= result['short_term_trend'])
        )
        result['near_duokong'] = (
            (result['close'] >= result['bull_bear_line'] * (1 - duokong_pct)) &
            (result['close'] <= result['bull_bear_line'] * (1 + duokong_pct))
        )
        result['near_short_trend'] = (
            (result['close'] >= result['short_term_trend'] * (1 - short_pct)) &
            (result['close'] <= result['short_term_trend'] * (1 + short_pct))
        )

        kdj_df = KDJ(result, n=9, m1=3, m2=3)
        result['K'] = kdj_df['K']
        result['D'] = kdj_df['D']
        result['J'] = kdj_df['J']

        if 'market_cap' in result.columns:
            result['market_cap_ok'] = result['market_cap'] > self.params['CAP']
        else:
            result['market_cap_ok'] = True

        result['vol_ratio'] = result['volume'] / REF(result['volume'], 1)
        result['vol_surge'] = result['vol_ratio'] >= self.params['N']
        result['positive_candle'] = result['close'] > result['open']
        result['key_candle'] = (
            result['vol_surge'] &
            result['positive_candle'] &
            result['market_cap_ok']
        )

        return result

    def rank_stocks_by_similarity(self, stock_data_dict, csv_manager, target_date, lookback_days=25):
        """
        使用 B1 完美图形匹配相似度进行排名

        :param stock_data_dict: {code: (name, df)} 股票数据字典
        :param csv_manager: CSV 管理器（用于加载案例数据）
        :param target_date: 选股目标日期
        :param lookback_days: 回看天数
        :return: 按相似度降序排列的股票列表
        """
        # 加载案例库（如果还没加载）
        if not self.cases:
            self._build_case_library(csv_manager)

        if not self.cases:
            print("  ⚠️ 警告：案例库为空")
            return []

        results = []

        for code, (name, df) in stock_data_dict.items():
            # 过滤 ST、退市股票
            if 'ST' in name or '退' in name:
                continue

            if df is None or df.empty or len(df) < 60:
                continue

            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])

            # 截取截至目标日期的数据
            target_pd = pd.Timestamp(target_date)
            historical_df = df[df['date'] <= target_pd].reset_index(drop=True)

            if historical_df.empty or len(historical_df) < lookback_days:
                continue

            # 使用最新一天作为选股日
            latest_row = historical_df.iloc[0]
            actual_date = latest_row['date'].date()

            date_diff = abs((actual_date - target_date).days)
            if date_diff > 10:
                continue

            # 计算指标，检查基本条件
            result = self.calculate_indicators(historical_df)
            latest = result.iloc[0]

            # 基本条件筛选
            if not latest.get('trend_above', False):
                continue
            if latest.get('J', 100) > self.params['J_VAL']:
                continue

            # 检查 M 天内是否有放量阳线
            lookback_df = result.head(self.params['M'])
            if lookback_df.empty:
                continue

            # 剔除：最大成交量日是阴线
            max_vol_idx = lookback_df['volume'].idxmax()
            max_vol_row = lookback_df.loc[max_vol_idx]
            if max_vol_row['close'] < max_vol_row['open']:
                continue

            key_candles = lookback_df[
                (lookback_df['volume'] >= REF(lookback_df['volume'], 1) * self.params['N']) &
                (lookback_df['close'] > lookback_df['open'])
            ]
            if key_candles.empty:
                continue

            # 位置检查
            category = None
            if latest.get('fall_in_bowl', False):
                category = 'bowl_center'
            elif latest.get('near_duokong', False):
                category = 'near_duokong'
            elif latest.get('near_short_trend', False):
                category = 'near_short_trend'

            if not category:
                continue

            # ========== B1 完美图形匹配 ==========
            # 提取候选特征（按日期正序排列）
            candidate_df = historical_df.head(lookback_days).sort_values('date').reset_index(drop=True)
            candidate_features = self.feature_extractor.extract(candidate_df)

            # 与所有案例匹配，取最高相似度
            best_match = None
            best_score = 0
            all_matches = []

            for case_id, case_data in self.cases.items():
                try:
                    similarity = self.matcher.match(candidate_features, case_data["features"])
                    score = similarity["total_score"]
                    all_matches.append({
                        "case_id": case_id,
                        "case_name": case_data["meta"]["name"],
                        "score": score
                    })
                    if score > best_score:
                        best_score = score
                        best_match = case_data["meta"]
                except Exception:
                    continue

            # 确定推荐理由（基于匹配结果和位置）
            reasons = []
            if category == 'bowl_center':
                reasons.append('回落碗中')
            elif category == 'near_duokong':
                reasons.append(f'靠近多空线 (±{self.params["duokong_pct"]}%)')
            elif category == 'near_short_trend':
                reasons.append(f'靠近短期趋势线 (±{self.params["short_pct"]}%)')

            if best_match:
                reasons.append(f'匹配{best_match["name"]}')

            results.append({
                'code': code,
                'name': name,
                'similarity_score': round(best_score, 2),
                'best_match_case': best_match["name"] if best_match else '',
                'best_match_date': best_match.get("breakout_date", '') if best_match else '',
                'category': category,
                'close': round(latest['close'], 2),
                'J': round(latest['J'], 2),
                'volume_ratio': round(latest.get('vol_ratio', 1.0), 2),
                'reasons': reasons,
                'actual_date': str(actual_date),
                'all_matches': all_matches
            })

        # 按相似度降序排序
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # 添加排名
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return results

    def rank_stocks_by_similarity_cached(self, stock_data_dict, csv_manager, target_date, lookback_days=25):
        """
        使用 B1 完美图形匹配相似度进行排名（缓存版本，直接使用已加载的数据）
        """
        # 加载案例库（如果还没加载）
        if not self.cases:
            self._build_case_library(csv_manager)

        if not self.cases:
            print("  ⚠️ 警告：案例库为空")
            return []

        results = []

        for code, (name, df) in stock_data_dict.items():
            # 过滤 ST、退市股票
            if 'ST' in name or '退' in name:
                continue

            if df is None or df.empty or len(df) < 60:
                continue

            df = df.copy()
            # 数据已经在加载时转换为 datetime

            # 截取截至目标日期的数据
            target_pd = pd.Timestamp(target_date)
            historical_df = df[df['date'] <= target_pd].reset_index(drop=True)

            if historical_df.empty or len(historical_df) < lookback_days:
                continue

            # 使用最新一天作为选股日
            latest_row = historical_df.iloc[0]
            actual_date = latest_row['date'].date()

            date_diff = abs((actual_date - target_date).days)
            if date_diff > 10:
                continue

            # 计算指标，检查基本条件
            result = self.calculate_indicators(historical_df)
            latest = result.iloc[0]

            # 基本条件筛选
            if not latest.get('trend_above', False):
                continue
            if latest.get('J', 100) > self.params['J_VAL']:
                continue

            # 检查 M 天内是否有放量阳线
            lookback_df = result.head(self.params['M'])
            if lookback_df.empty:
                continue

            # 剔除：最大成交量日是阴线
            max_vol_idx = lookback_df['volume'].idxmax()
            max_vol_row = lookback_df.loc[max_vol_idx]
            if max_vol_row['close'] < max_vol_row['open']:
                continue

            key_candles = lookback_df[
                (lookback_df['volume'] >= REF(lookback_df['volume'], 1) * self.params['N']) &
                (lookback_df['close'] > lookback_df['open'])
            ]
            if key_candles.empty:
                continue

            # 位置检查
            category = None
            if latest.get('fall_in_bowl', False):
                category = 'bowl_center'
            elif latest.get('near_duokong', False):
                category = 'near_duokong'
            elif latest.get('near_short_trend', False):
                category = 'near_short_trend'

            if not category:
                continue

            # ========== B1 完美图形匹配 ==========
            # 提取候选特征（按日期正序排列）
            candidate_df = historical_df.head(lookback_days).sort_values('date').reset_index(drop=True)
            candidate_features = self.feature_extractor.extract(candidate_df)

            # 与所有案例匹配，取最高相似度
            best_match = None
            best_score = 0
            all_matches = []

            for case_id, case_data in self.cases.items():
                try:
                    similarity = self.matcher.match(candidate_features, case_data["features"])
                    score = similarity["total_score"]
                    all_matches.append({
                        "case_id": case_id,
                        "case_name": case_data["meta"]["name"],
                        "score": score
                    })
                    if score > best_score:
                        best_score = score
                        best_match = case_data["meta"]
                except Exception:
                    continue

            # 确定推荐理由（基于匹配结果和位置）
            reasons = []
            if category == 'bowl_center':
                reasons.append('回落碗中')
            elif category == 'near_duokong':
                reasons.append(f'靠近多空线 (±{self.params["duokong_pct"]}%)')
            elif category == 'near_short_trend':
                reasons.append(f'靠近短期趋势线 (±{self.params["short_pct"]}%)')

            if best_match:
                reasons.append(f'匹配{best_match["name"]}')

            results.append({
                'code': code,
                'name': name,
                'similarity_score': round(best_score, 2),
                'best_match_case': best_match["name"] if best_match else '',
                'best_match_date': best_match.get("breakout_date", '') if best_match else '',
                'category': category,
                'close': round(latest['close'], 2),
                'J': round(latest['J'], 2),
                'volume_ratio': round(latest.get('vol_ratio', 1.0), 2),
                'reasons': reasons,
                'actual_date': str(actual_date),
                'all_matches': all_matches
            })

        # 按相似度降序排序
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # 添加排名
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return results


class Backtester:
    """回测引擎"""

    def __init__(self, config=None):
        self.config = config or BacktestConfig()
        self.data_dir = Path(__file__).parent / 'data'
        self.csv_manager = CSVManager(str(self.data_dir))
        self.strategy = B1BacktestStrategy()
        self.stock_names = self._load_stock_names()

    def _load_stock_names(self):
        """加载股票名称缓存"""
        names_file = self.data_dir / 'stock_names.json'
        if names_file.exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _get_all_stock_codes(self):
        """获取所有股票代码"""
        return self.csv_manager.list_all_stocks()

    def _get_selection_dates(self):
        """获取选股日期列表（每月 1 日）"""
        today = datetime.now()
        dates = []
        for i in range(self.config.months):
            selection_date = (today - relativedelta(months=i)).replace(day=1)
            dates.append(selection_date.date())
        return dates

    def _load_stock_data(self, code, target_date):
        """加载单只股票的历史数据"""
        df = self.csv_manager.read_stock(code)
        if df is None or df.empty:
            return None, None
        name = self.stock_names.get(code, '未知')
        df['date'] = pd.to_datetime(df['date'])
        return name, df

    def run_selection(self, selection_date):
        """执行单次选股"""
        stock_codes = self._get_all_stock_codes()
        print(f"  正在加载 {len(stock_codes)} 只股票数据...")

        stock_data_dict = {}
        loaded = 0
        for i, code in enumerate(stock_codes):
            name, df = self._load_stock_data(code, selection_date)
            if name and df is not None and not df.empty:
                stock_data_dict[code] = (name, df)
            loaded += 1
            if loaded % 1000 == 0:
                print(f"    已加载 {loaded}/{len(stock_codes)}...")

        print(f"  ✓ 数据加载完成，共 {len(stock_data_dict)} 只股票")
        print(f"  正在执行 B1 完美图形匹配排名...")

        ranked = self.strategy.rank_stocks_by_similarity(
            stock_data_dict, self.csv_manager, selection_date
        )

        print(f"  ✓ 排名完成，共 {len(ranked)} 只股票符合条件")
        return ranked

    def _calculate_return_and_max_gain(self, code, buy_date, hold_days, cached_df=None):
        """
        计算持有期收益率和期间最大涨幅
        :return: (buy_price, sell_price, return_pct, sell_date, max_gain_pct, max_gain_day)
        """
        df = cached_df if cached_df is not None else self.csv_manager.read_stock(code)
        if df is None or df.empty:
            return None, None, None, None, None, None

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 找到买入日数据
        buy_mask = df['date'].dt.date == buy_date
        if not buy_mask.any():
            for delta in range(10):
                check_date = buy_date - timedelta(days=delta)
                buy_mask = df['date'].dt.date == check_date
                if buy_mask.any():
                    break
            else:
                return None, None, None, None, None, None

        buy_price = df.loc[buy_mask, 'close'].iloc[0]
        actual_buy_date = df.loc[buy_mask, 'date'].iloc[0].date()

        # 计算卖出日期
        sell_target = actual_buy_date + timedelta(days=hold_days)

        # 找到卖出日数据
        sell_mask = df['date'].dt.date == sell_target
        sell_date = None

        if not sell_mask.any():
            found = False
            for delta in range(10):
                check_date = sell_target + timedelta(days=delta)
                sell_mask = df['date'].dt.date == check_date
                if sell_mask.any():
                    sell_price = df.loc[sell_mask, 'close'].iloc[0]
                    sell_date = df.loc[sell_mask, 'date'].iloc[0].date()
                    found = True
                    break

            if not found:
                sell_price = df.iloc[0]['close']
                sell_date = df.iloc[0]['date'].date()
        else:
            sell_price = df.loc[sell_mask, 'close'].iloc[0]
            sell_date = df.loc[sell_mask, 'date'].iloc[0].date()

        return_pct = (sell_price - buy_price) / buy_price * 100

        # 计算期间最大涨幅（从买入日到卖出日之间的最高价相对于买入价的涨幅）
        # 找到买入和卖出之间的所有数据
        df_sorted = df.sort_values('date')
        buy_idx = df_sorted[df_sorted['date'].dt.date == actual_buy_date]
        sell_idx = df_sorted[df_sorted['date'].dt.date == sell_date]

        if len(buy_idx) > 0 and len(sell_idx) > 0:
            start_pos = buy_idx.index[0]
            end_pos = sell_idx.index[0]
            period_df = df_sorted.loc[start_pos:end_pos]
            max_high = period_df['high'].max()
            max_gain_pct = (max_high - buy_price) / buy_price * 100

            # 找到最大涨幅发生在第几天（相对于买入日）
            max_high_row = period_df[period_df['high'] == max_high]
            if len(max_high_row) > 0:
                max_high_date = max_high_row['date'].iloc[0].date()
                max_gain_day = (max_high_date - actual_buy_date).days
            else:
                max_gain_day = 0
        else:
            max_gain_pct = return_pct  # 如果找不到区间，用最终收益率
            max_gain_day = hold_days

        return buy_price, sell_price, return_pct, sell_date, max_gain_pct, max_gain_day

    def run_backtest(self):
        """执行完整回测"""
        print("=" * 95)
        print("📊 选股模型回测报告 - B1 完美图形匹配版")
        print("=" * 95)
        print(f"回测参数:")
        print(f"  - 回测月数：{self.config.months}个月")
        print(f"  - 每次选股：Top {self.config.top_n}")
        print(f"  - 持有期：{self.config.hold_days}天")
        print(f"  - 市值门槛：{self.config.cap_threshold / 1e8:.0f}亿")
        print("=" * 95)

        selection_dates = self._get_selection_dates()
        print(f"\n选股日期范围：{selection_dates[-1]} 至 {selection_dates[0]}")

        # 性能优化：提前加载所有股票数据到缓存
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
        monthly_summary = []

        for idx, sel_date in enumerate(selection_dates, 1):
            print(f"\n[{idx}/{self.config.months}] 选股日：{sel_date}")

            # 使用缓存的数据进行选股
            ranked = self.strategy.rank_stocks_by_similarity_cached(
                stock_data_dict, self.csv_manager, sel_date
            )

            if not ranked:
                print(f"  ⚠️ 该日期未选出任何股票")
                continue

            top_stocks = ranked[:self.config.top_n]
            print(f"  ✓ 选出 Top {len(top_stocks)} 只股票")

            month_results = []
            for stock in top_stocks:
                actual_date = datetime.strptime(stock['actual_date'], '%Y-%m-%d').date()
                # 使用缓存数据计算收益
                cached_df = stock_data_dict.get(stock['code'], (None, None))[1]
                buy_price, sell_price, return_pct, sell_date, max_gain_pct, max_gain_day = self._calculate_return_and_max_gain(
                    stock['code'], actual_date, self.config.hold_days, cached_df
                )

                if buy_price is None:
                    continue

                month_results.append({
                    'selection_date': str(sel_date),
                    'actual_selection_date': stock['actual_date'],
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

            if month_results:
                all_results.extend(month_results)

                avg_return = sum(r['return_pct'] for r in month_results) / len(month_results)
                win_count = sum(1 for r in month_results if r['return_pct'] > 0)
                win_rate = win_count / len(month_results) * 100

                monthly_summary.append({
                    'selection_date': str(sel_date),
                    'stock_count': len(month_results),
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'win_count': win_count
                })

                print(f"  平均收益：{avg_return:.2f}%, 胜率：{win_rate:.1f}% ({win_count}/{len(month_results)})")

        return all_results, monthly_summary

    def print_report(self, all_results, monthly_summary):
        """打印回测报告"""
        print("\n" + "=" * 95)
        print("📈 回测结果汇总")
        print("=" * 95)

        if not all_results:
            print("✗ 没有回测数据")
            return

        total_trades = len(all_results)
        avg_return = sum(r['return_pct'] for r in all_results) / total_trades
        win_count = sum(1 for r in all_results if r['return_pct'] > 0)
        win_rate = win_count / total_trades * 100

        best_trade = max(all_results, key=lambda x: x['return_pct'])
        worst_trade = min(all_results, key=lambda x: x['return_pct'])

        print(f"\n总体统计:")
        print(f"  - 总交易次数：{total_trades}")
        print(f"  - 平均收益：{avg_return:.2f}%")
        print(f"  - 胜率：{win_rate:.1f}% ({win_count}/{total_trades})")
        print(f"  - 最佳交易：{best_trade['code']} {best_trade['name']} +{best_trade['return_pct']:.2f}%")
        print(f"  - 最差交易：{worst_trade['code']} {worst_trade['name']} {worst_trade['return_pct']:.2f}%")

        print("\n" + "-" * 95)
        print("月度表现明细:")
        print("-" * 95)
        print(f"{'选股日期':<14} {'股票数':>8} {'平均收益':>14} {'胜率':>10}")
        print("-" * 95)

        for month in monthly_summary:
            print(f"{month['selection_date']:<14} {month['stock_count']:>8} {month['avg_return']:>+13.2f}% {month['win_rate']:>9.1f}%")

        print("\n" + "=" * 95)
        print("📋 详细交易清单")
        print("=" * 95)

        from itertools import groupby
        all_results.sort(key=lambda x: x['selection_date'])

        for sel_date, group in groupby(all_results, key=lambda x: x['selection_date']):
            group_list = list(group)
            print(f"\n【选股日：{sel_date}】")
            print(f"{'排名':>4} {'代码':<10} {'名称':<12} {'相似度':>7} {'匹配案例':>12} {'买入价':>8} {'卖出价':>8} {'涨跌':>8} {'最大涨幅':>8} {'天数':>6}")
            print("-" * 108)

            for r in group_list:
                pct_str = f"{r['return_pct']:+.1f}%"
                match_str = r['best_match_case'][:12] if r['best_match_case'] else '-'
                max_gain_str = f"{r['max_gain_pct']:+.1f}%" if r.get('max_gain_pct') is not None else 'N/A'
                max_gain_day_str = f"第{r['max_gain_day']}天" if r.get('max_gain_day') is not None else '-'
                print(f"{r['rank']:>4} {r['code']:<10} {r['name']:<12} {r['similarity_score']:>6.1f}% "
                      f"{match_str:>12} {r['buy_price']:>8.2f} {r['sell_price']:>8.2f} {pct_str:>8} {max_gain_str:>8} {max_gain_day_str:>6}")

        self._save_results(all_results, monthly_summary)

    def _save_results(self, all_results, monthly_summary):
        """保存回测结果"""
        output_dir = Path(__file__).parent / 'backtest_results'
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_file = output_dir / f'b1_backtest_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'months': self.config.months,
                    'top_n': self.config.top_n,
                    'hold_days': self.config.hold_days,
                    'cap_threshold': self.config.cap_threshold
                },
                'all_results': all_results,
                'monthly_summary': monthly_summary
            }, f, ensure_ascii=False, indent=2)

        csv_file = output_dir / f'b1_backtest_{timestamp}.csv'
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')

        simple_rows = []
        for r in all_results:
            simple_rows.append({
                '选股日期': r['selection_date'],
                '实际选股日': r['actual_selection_date'],
                '排名': r['rank'],
                '代码': r['code'],
                '名称': r['name'],
                '相似度': f"{r['similarity_score']:.1f}%",
                '匹配案例': r['best_match_case'],
                '匹配日期': r['best_match_date'],
                '类别': r['category'],
                '推荐理由': ' | '.join(r['reasons']),
                '买入价': r['buy_price'],
                '卖出价': r['sell_price'],
                '涨跌幅': f"{r['return_pct']:+.2f}%",
                '最大涨幅': f"{r['max_gain_pct']:+.2f}%" if r.get('max_gain_pct') is not None else 'N/A',
                '最大涨幅天数': f"第{r['max_gain_day']}天" if r.get('max_gain_day') is not None else 'N/A',
                '卖出日期': r['sell_date'],
                '持有天数': r['hold_days']
            })

        simple_csv = output_dir / f'b1_backtest_readable_{timestamp}.csv'
        df_simple = pd.DataFrame(simple_rows)
        df_simple.to_csv(simple_csv, index=False, encoding='utf-8-sig')

        print(f"\n💾 结果已保存:")
        print(f"  - JSON: {results_file}")
        print(f"  - CSV:  {csv_file}")
        print(f"  - 可读 CSV: {simple_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='选股模型回测 - B1 完美图形匹配版')
    parser.add_argument('--months', type=int, default=12, help='回测月数 (默认：12)')
    parser.add_argument('--top-n', type=int, default=10, help='每次选股数量 (默认：10)')
    parser.add_argument('--hold-days', type=int, default=30, help='持有天数 (默认：30)')
    parser.add_argument('--cap', type=float, default=40, help='市值门槛 (单位：亿，默认：40)')

    args = parser.parse_args()

    config = BacktestConfig(
        months=args.months,
        top_n=args.top_n,
        hold_days=args.hold_days,
        cap_threshold=args.cap * 1e8
    )

    backtester = Backtester(config)
    all_results, monthly_summary = backtester.run_backtest()
    backtester.print_report(all_results, monthly_summary)


if __name__ == '__main__':
    main()
