#!/usr/bin/env python3
"""
每日回测脚本 - B1 完美图形匹配版 (多进程并行优化版)

功能：
1. 支持输入任意日期范围
2. 使用原有 backtest.py 的优化逻辑（指标缓存）
3. 输出详细分析报告
4. 支持多进程并行计算，大幅提升回测速度

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.csv_manager import CSVManager
from utils.technical import MA, EMA, LLV, HHV, REF, EXIST, KDJ, calculate_zhixing_trend
from utils.industry_fetcher import IndustryFetcher, IndustryHeatCalculator, TurnoverCache
from strategy.pattern_feature_extractor import PatternFeatureExtractor
from strategy.pattern_matcher import PatternMatcher
from strategy.pattern_config import SIMILARITY_WEIGHTS, B1_PERFECT_CASES
from strategy.bowl_rebound import BowlReboundStrategy
from strategy.zge_filter import ZgeFilter


# 构建案例查找表：case_id -> 案例信息
CASE_INFO = {case["id"]: case for case in B1_PERFECT_CASES}


def _process_single_day(sel_date, stock_data_list, config_dict, top_n, hold_days, preloaded_data):
    """
    处理单个交易日的选股和收益计算（用于多进程并行）

    :param sel_date: 选股日期
    :param stock_data_list: 序列化后的股票数据列表
    :param config_dict: 配置字典
    :param top_n: 选股Top N
    :param hold_days: 持有天数
    :param preloaded_data: 预加载数据
    :return: (day_results, day_summary)
    """
    import io

    # 使用主进程预加载的数据（避免重复加载）
    strategy = FastDailyBacktestStrategy()
    strategy.cases = preloaded_data['cases']
    strategy.csv_manager = CSVManager(Path(config_dict['data_path']))

    # 初始化行业数据获取器（使用预加载的行业映射）
    industry_fetcher = IndustryFetcher(Path(config_dict['data_path']) / 'industry_cache')
    industry_fetcher.stock_industry_map = preloaded_data['industry_map']
    industry_fetcher.industry_stocks_map = preloaded_data['industry_stocks_map']

    # 使用预计算的行业热度缓存
    industry_heats_cache = preloaded_data.get('industry_heats', {})

    # 将列表转换回字典
    stock_data_dict = {}
    for item in stock_data_list:
        code, name, df_bytes = item
        buffer = io.BytesIO(df_bytes)
        df = pd.read_pickle(buffer)
        stock_data_dict[code] = (name, df)

    # 选股
    ranked = strategy.rank_stocks_single_date(stock_data_dict, sel_date, lookback_days=config_dict.get('lookback_days', 25), filter_zge_risk=config_dict.get('filter_zge_risk', False))

    if not ranked:
        return ([], None)

    top_stocks = ranked[:top_n]
    day_results = []

    # 计算每只股票的收益
    for stock in top_stocks:
        actual_date = datetime.strptime(stock['actual_date'], '%Y-%m-%d').date()

        # 获取股票数据
        code = stock['code']
        if code not in stock_data_dict:
            continue

        name, df = stock_data_dict[code]
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 计算收益（内联计算逻辑，避免跨进程调用）
        buy_mask = df['date'].dt.date == actual_date
        buy_rows = df[buy_mask]

        if buy_rows.empty:
            buy_row = df[df['date'] <= pd.to_datetime(actual_date)].head(1)
            if buy_row.empty:
                continue
            buy_price = buy_row['close'].iloc[0]
            actual_buy_date = buy_row['date'].iloc[0].date()
        else:
            buy_price = buy_rows['close'].iloc[0]
            actual_buy_date = actual_date

        # 找到卖出日
        sell_date = actual_date + timedelta(days=hold_days)
        sell_mask = df['date'].dt.date == sell_date
        sell_rows = df[sell_mask]

        if sell_rows.empty:
            sell_df = df[df['date'] >= pd.to_datetime(sell_date)]
            if sell_df.empty:
                continue
            sell_price = sell_df['close'].iloc[0]
            actual_sell_date = sell_df['date'].iloc[0].date()
        else:
            sell_price = sell_rows['close'].iloc[0]
            actual_sell_date = sell_date

        return_pct = (sell_price - buy_price) / buy_price * 100

        # 计算最大涨幅和止盈止损触发日期
        df_sorted = df.sort_values('date').reset_index(drop=True)
        buy_mask = df_sorted['date'].dt.date == actual_buy_date
        sell_mask = df_sorted['date'].dt.date == actual_sell_date

        if buy_mask.any() and sell_mask.any():
            start_pos = buy_mask.argmax()
            end_pos = sell_mask.argmax()
            period_df = df_sorted.iloc[start_pos:end_pos+1]

            max_high = period_df['high'].max()
            max_gain_pct = (max_high - buy_price) / buy_price * 100

            max_high_row = period_df[period_df['high'] == max_high]
            if len(max_high_row) > 0:
                max_high_date = max_high_row['date'].iloc[0].date()
                max_gain_day = (max_high_date - actual_buy_date).days
            else:
                max_gain_day = 0

            # 计算持有期内最大跌幅（用最低价）
            min_low = period_df['low'].min()
            max_drop_pct = (min_low - buy_price) / buy_price * 100
            min_low_row = period_df[period_df['low'] == min_low]
            if len(min_low_row) > 0:
                max_drop_date = min_low_row['date'].iloc[0].date()
                max_drop_day = (max_drop_date - actual_buy_date).days
            else:
                max_drop_date = None
                max_drop_day = 0

            # 止盈止损触发日期
            trigger_10pct_date = None
            trigger_5pct_date = None  # 新增：5% 止盈触发日期
            trigger_neg2pct_date = None
            trigger_neg4pct_date = None
            trigger_10pct_day = None  # 达到 +10% 的天数
            trigger_5pct_day = None   # 新增：达到 +5% 的天数
            ever_below_zero = False  # 是否曾跌破成本价（0%）

            price_10pct = buy_price * 1.10
            price_5pct = buy_price * 1.05   # 新增：5% 止盈价格
            price_neg2pct = buy_price * 0.98
            price_neg4pct = buy_price * 0.96

            for idx, row in period_df.iterrows():
                row_date = row['date'].date()
                days_from_buy = (row_date - actual_buy_date).days

                if trigger_10pct_date is None and row['high'] >= price_10pct:
                    trigger_10pct_date = row_date
                    trigger_10pct_day = days_from_buy
                if trigger_5pct_date is None and row['high'] >= price_5pct:  # 新增：5% 止盈检测
                    trigger_5pct_date = row_date
                    trigger_5pct_day = days_from_buy
                if trigger_neg2pct_date is None and row['low'] <= price_neg2pct:
                    trigger_neg2pct_date = row_date
                if trigger_neg4pct_date is None and row['low'] <= price_neg4pct:
                    trigger_neg4pct_date = row_date
                # 检查是否曾跌破成本价
                if row['low'] < buy_price:
                    ever_below_zero = True

                # 所有触发都检测到后退出
                if all([trigger_10pct_date, trigger_5pct_date, trigger_neg2pct_date, trigger_neg4pct_date]):
                    break

            first_reach_order = None
            triggers = []
            if trigger_10pct_date:
                triggers.append(('10pct', trigger_10pct_date))
            if trigger_5pct_date:
                triggers.append(('5pct', trigger_5pct_date))  # 新增：5% 止盈
            if trigger_neg2pct_date:
                triggers.append(('neg2pct', trigger_neg2pct_date))
            if trigger_neg4pct_date:
                triggers.append(('neg4pct', trigger_neg4pct_date))

            if len(triggers) >= 2:
                triggers_sorted = sorted(triggers, key=lambda x: x[1])
                first_reach_order = tuple([t[0] for t in triggers_sorted])
        else:
            max_gain_pct = return_pct
            max_gain_day = hold_days
            max_drop_pct = return_pct
            max_drop_date = None
            max_drop_day = hold_days
            trigger_10pct_date = None
            trigger_5pct_date = None  # 新增
            trigger_neg2pct_date = None
            trigger_neg4pct_date = None
            trigger_10pct_day = None
            trigger_5pct_day = None  # 新增
            first_reach_order = None

        # 计算行业热度（使用预计算缓存）
        # 从内置映射获取行业（不使用网络）
        industry = industry_fetcher.get_industry_for_stock(code, refresh_if_missing=False)

        # 日期格式转换为字符串
        def date_to_str(d):
            if isinstance(d, str):
                return d
            if isinstance(d, pd.Timestamp):
                return d.strftime('%Y-%m-%d')
            if hasattr(d, 'strftime'):
                return d.strftime('%Y-%m-%d')
            return str(d)

        # 提取行业热度获取逻辑为内联函数（消除重复代码）
        def _get_heat(industry, date, industry_heats_cache, preloaded_data, stock_keys):
            if not industry or not date:
                return None
            date_str = date_to_str(date)
            # 优先从预计算缓存获取
            if industry in industry_heats_cache and date_str in industry_heats_cache[industry]:
                return industry_heats_cache[industry][date_str]
            # 尝试从 turnover_cache 实时计算
            try:
                turnover_cache = preloaded_data.get('turnover_cache', {})
                if turnover_cache and date_str in turnover_cache:
                    heat_calc = IndustryHeatCalculator(industry_fetcher)
                    return heat_calc.calculate_industry_heat_fast(
                        industry, date_str, turnover_cache, stock_keys)
            except:
                pass
            return None

        # 获取各关键日期的行业热度
        stock_keys = list(stock_data_dict.keys())
        industry_heat_buy = _get_heat(industry, actual_buy_date, industry_heats_cache, preloaded_data, stock_keys)
        industry_heat_10pct = _get_heat(industry, trigger_10pct_date, industry_heats_cache, preloaded_data, stock_keys)
        industry_heat_5pct = _get_heat(industry, trigger_5pct_date, industry_heats_cache, preloaded_data, stock_keys)
        industry_heat_neg2pct = _get_heat(industry, trigger_neg2pct_date, industry_heats_cache, preloaded_data, stock_keys)
        industry_heat_neg4pct = _get_heat(industry, trigger_neg4pct_date, industry_heats_cache, preloaded_data, stock_keys)
        industry_heat_sell = _get_heat(industry, actual_sell_date, industry_heats_cache, preloaded_data, stock_keys)

        day_results.append({
            'selection_date': str(sel_date),
            'code': code,
            'name': name,
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
            'max_drop_pct': max_drop_pct,
            'max_drop_date': str(max_drop_date) if max_drop_date else '',
            'max_drop_day': max_drop_day,
            'sell_date': str(actual_sell_date) if actual_sell_date else '',
            'hold_days': hold_days,
            'trigger_10pct_date': str(trigger_10pct_date) if trigger_10pct_date else '',
            'trigger_5pct_date': str(trigger_5pct_date) if trigger_5pct_date else '',  # 新增
            'trigger_neg2pct_date': str(trigger_neg2pct_date) if trigger_neg2pct_date else '',
            'trigger_neg4pct_date': str(trigger_neg4pct_date) if trigger_neg4pct_date else '',
            'trigger_10pct_day': trigger_10pct_day if trigger_10pct_day is not None else '',
            'trigger_5pct_day': trigger_5pct_day if trigger_5pct_day is not None else '',  # 新增
            'first_reach_order': str(first_reach_order) if first_reach_order else '',
            'ever_below_zero': ever_below_zero,  # 是否曾跌破成本价
            # 行业热度字段
            'industry': industry if industry else '未知',
            'industry_heat_buy': round(industry_heat_buy, 2) if industry_heat_buy is not None else '',
            'industry_heat_10pct': round(industry_heat_10pct, 2) if industry_heat_10pct is not None else '',
            'industry_heat_5pct': round(industry_heat_5pct, 2) if industry_heat_5pct is not None else '',  # 新增
            'industry_heat_neg2pct': round(industry_heat_neg2pct, 2) if industry_heat_neg2pct is not None else '',
            'industry_heat_neg4pct': round(industry_heat_neg4pct, 2) if industry_heat_neg4pct is not None else '',
            'industry_heat_sell': round(industry_heat_sell, 2) if industry_heat_sell is not None else ''
        })

    if not day_results:
        return ([], None)

    day_return = sum(r['return_pct'] for r in day_results) / len(day_results)
    win_count = sum(1 for r in day_results if r['return_pct'] > 0)
    win_rate = win_count / len(day_results) * 100

    day_summary = {
        'date': str(sel_date),
        'count': len(day_results),
        'win_count': win_count,
        'win_rate': win_rate,
        'avg_return': day_return
    }

    return (day_results, day_summary)


class FastDailyBacktestConfig:
    """快速每日回测配置"""
    def __init__(self, start_date, end_date, top_n=10, hold_days=30, cap_threshold=4e9, lookback_days=45, filter_zge_risk=False):
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.hold_days = hold_days
        self.cap_threshold = cap_threshold
        self.lookback_days = lookback_days
        self.filter_zge_risk = filter_zge_risk


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

        # 使用进程 ID 控制打印，避免多进程重复输出
        import os
        worker_id = os.environ.get('WORKER_ID', os.getpid())
        if not hasattr(FastDailyBacktestStrategy, '_case_lib_printed'):
            FastDailyBacktestStrategy._case_lib_printed = set()
        if worker_id not in FastDailyBacktestStrategy._case_lib_printed:
            FastDailyBacktestStrategy._case_lib_printed.add(worker_id)
            if len(FastDailyBacktestStrategy._case_lib_printed) == 1:
                print(f"  ✓ 案例库加载完成：{len(self.cases)} 个案例")

    def rank_stocks_single_date(self, stock_data_dict, target_date, precomputed_indicators=None, lookback_days=45, filter_zge_risk=False):
        """
        对单日进行选股排名

        :param stock_data_dict: 股票数据字典 {code: (name, df)}
        :param target_date: 选股日期
        :param precomputed_indicators: 预计算的指标字典（可选）
        :param lookback_days: 回看天数
        :param filter_zge_risk: 是否过滤Z哥风控规则触发的股票
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

            # 获取最近 lookback_days 天数据（倒序，最新在前）
            candidate_df = historical_df.head(lookback_days).reset_index(drop=True)

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

        # ============== Z哥过滤 ==============
        zge_filter = ZgeFilter()
        filtered_candidates = []
        for cand in candidates:
            code = cand['code']
            # 获取股票数据
            if code not in stock_data_dict:
                continue
            name, full_df = stock_data_dict[code]
            # 截断到target_date，只使用选股日之前的数据，避免未来数据
            historical_df = full_df[full_df['date'] <= target_pd].copy().reset_index(drop=True)
            if len(historical_df) < 21: # 风控需要至少21天数据
                cand['has_risk'] = True
                cand['zge_hints'] = ["日线数据不足21天"]
                cand['zge_triggered_rules'] = ["日线数据不足21天"]
                if not filter_zge_risk:
                    filtered_candidates.append(cand)
                continue
            # 执行过滤
            _, zge_result = zge_filter.filter_stock(code, historical_df)
            cand.update(zge_result)
            # 过滤规则：开启过滤时剔除has_risk=True的股票，关闭时全部保留
            if not filter_zge_risk or not zge_result['has_risk']:
                filtered_candidates.append(cand)
        # 替换结果
        candidates = filtered_candidates
        # ============== 过滤结束 ==============

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
        """
        计算持有期收益率和期间最大涨幅，以及止盈止损触发日期
        :return: (buy_price, sell_price, return_pct, actual_sell_date, max_gain_pct, max_gain_day,
                  trigger_10pct_date, trigger_neg2pct_date, trigger_neg4pct_date, trigger_10pct_day,
                  first_reach_order)
        """
        df = cached_df if cached_df is not None else self.csv_manager.read_stock(code)
        if df is None or df.empty:
            return None, None, None, None, None, None, None, None, None, None, None

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 找到买入日数据
        buy_mask = df['date'].dt.date == buy_date
        buy_rows = df[buy_mask]

        if buy_rows.empty:
            buy_row = df[df['date'] <= pd.to_datetime(buy_date)].head(1)
            if buy_row.empty:
                return None, None, None, None, None, None, None, None, None, None
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
                return None, None, None, None, None, None, None, None, None, None
            sell_price = sell_df['close'].iloc[-1]  # tail(1) = 最小的大于等于 sell_date 的日期
            actual_sell_date = sell_df['date'].iloc[-1].date()
        else:
            sell_price = sell_rows['close'].iloc[0]
            actual_sell_date = sell_date

        return_pct = (sell_price - buy_price) / buy_price * 100

        # 计算最大涨幅和止盈止损触发日期（修复：重置索引确保连续，使用 iloc 避免索引跳跃问题）
        df_sorted = df.sort_values('date').reset_index(drop=True)
        buy_mask = df_sorted['date'].dt.date == actual_buy_date
        sell_mask = df_sorted['date'].dt.date == actual_sell_date

        if buy_mask.any() and sell_mask.any():
            start_pos = buy_mask.argmax()  # 获取买入位置
            end_pos = sell_mask.argmax()   # 获取卖出位置
            period_df = df_sorted.iloc[start_pos:end_pos+1]  # 包含两端

            # 最大涨幅计算
            max_high = period_df['high'].max()
            max_gain_pct = (max_high - buy_price) / buy_price * 100

            max_high_row = period_df[period_df['high'] == max_high]
            if len(max_high_row) > 0:
                max_high_date = max_high_row['date'].iloc[0].date()
                max_gain_day = (max_high_date - actual_buy_date).days
            else:
                max_gain_day = 0

            # 止盈止损触发日期计算
            trigger_10pct_date = None
            trigger_neg2pct_date = None
            trigger_neg4pct_date = None

            # 计算阈值
            price_10pct = buy_price * 1.10  # 涨 10%
            price_neg2pct = buy_price * 0.98  # 跌 2%
            price_neg4pct = buy_price * 0.96  # 跌 4%

            # 遍历持有期内的每日数据，查找首次触发日期
            for idx, row in period_df.iterrows():
                row_date = row['date'].date()
                days_from_buy = (row_date - actual_buy_date).days

                # 检查是否达到 +10%（用最高价）
                if trigger_10pct_date is None and row['high'] >= price_10pct:
                    trigger_10pct_date = row_date
                    trigger_10pct_day = days_from_buy

                # 检查是否达到 -2%（用最低价）
                if trigger_neg2pct_date is None and row['low'] <= price_neg2pct:
                    trigger_neg2pct_date = row_date
                    trigger_neg2pct_day = days_from_buy

                # 检查是否达到 -4%（用最低价）
                if trigger_neg4pct_date is None and row['low'] <= price_neg4pct:
                    trigger_neg4pct_date = row_date
                    trigger_neg4pct_day = days_from_buy

                # 如果三个触发日期都找到了，可以提前退出
                if trigger_10pct_date and trigger_neg2pct_date and trigger_neg4pct_date:
                    break

            # 计算触发顺序（用于路径分析）
            first_reach_order = None
            triggers = []
            if trigger_10pct_date:
                triggers.append(('10pct', trigger_10pct_date))
            if trigger_neg2pct_date:
                triggers.append(('neg2pct', trigger_neg2pct_date))
            if trigger_neg4pct_date:
                triggers.append(('neg4pct', trigger_neg4pct_date))

            if len(triggers) >= 2:
                # 按日期排序
                triggers_sorted = sorted(triggers, key=lambda x: x[1])
                first_reach_order = tuple([t[0] for t in triggers_sorted])
        else:
            max_gain_pct = return_pct
            max_gain_day = hold_days
            trigger_10pct_date = None
            trigger_neg2pct_date = None
            trigger_neg4pct_date = None
            trigger_10pct_day = None
            first_reach_order = None

        return (buy_price, sell_price, return_pct, actual_sell_date, max_gain_pct, max_gain_day,
                trigger_10pct_date, trigger_neg2pct_date, trigger_neg4pct_date, trigger_10pct_day,
                first_reach_order)

    def run_backtest(self, max_workers=None):
        """
        执行每日回测（多进程并行优化版）

        :param max_workers: 最大工作进程数，默认使用 CPU 核心数
        """
        print("=" * 95)
        print("📊 每日回测报告 - B1 完美图形匹配版 (多进程并行优化版)")
        print("=" * 95)
        print(f"回测参数:")
        print(f"  - 日期范围：{self.config.start_date} 至 {self.config.end_date}")
        print(f"  - 每次选股：Top {self.config.top_n}")
        print(f"  - 持有期：{self.config.hold_days}天")
        print(f"  - 市值门槛：{self.config.cap_threshold / 1e8:.0f}亿")
        print(f"  - 并行进程数：{max_workers if max_workers else cpu_count()}")
        print("=" * 95)

        # 获取交易日
        trading_dates = self._get_trading_dates()
        print(f"\n交易日数量：{len(trading_dates)}天")

        # 预加载股票数据
        print("\n预加载所有股票数据到缓存...")
        stock_codes = self._get_all_stock_codes()
        stock_data_dict = {}
        invalid_count = 0
        from utils.stock_filter import is_valid_stock
        for i, code in enumerate(stock_codes):
            df = self.csv_manager.read_stock(code)
            if df is None or df.empty:
                invalid_count += 1
                continue
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            name = self.stock_names.get(code, '未知')

            # 过滤退市/异常股票
            if not is_valid_stock(name, df):
                invalid_count += 1
                continue

            stock_data_dict[code] = (name, df)
            if (i + 1) % 2000 == 0:
                print(f"  已缓存 {i + 1}/{len(stock_codes)}...")
        print(f"✓ 缓存完成，共 {len(stock_data_dict)} 只股票，过滤 {invalid_count} 只无效股票\n")

        # 在主进程中预加载案例库和行业映射（避免子进程重复加载）
        print("主进程预加载案例库和行业映射...")
        self.strategy._build_case_library(self.csv_manager)
        cases_data = self.strategy.cases

        # 初始化行业数据获取器并加载行业映射
        industry_cache_dir = self.csv_manager.data_dir / 'industry_cache'
        industry_fetcher = IndustryFetcher(industry_cache_dir)
        industry_fetcher.load_industry_mapping(force_refresh=False)
        # 合并内置映射，扩展行业覆盖
        industry_fetcher._merge_builtin_industry_map()
        print(f"  ✓ 行业映射加载完成：{len(industry_fetcher.stock_industry_map)} 只股票")

        # 准备要传递给子进程的预加载数据
        preloaded_data = {
            'cases': cases_data,
            'industry_map': industry_fetcher.stock_industry_map,
            'industry_stocks_map': industry_fetcher.industry_stocks_map,
            'industry_heats': {},  # 预计算的行业热度缓存
        }

        # 准备并行处理数据 - 将 DataFrame 转换为可序列化的格式
        # 使用 pickle 序列化 DataFrame 以减少内存占用
        import io
        stock_data_list = []
        for code, (name, df) in stock_data_dict.items():
            buffer = io.BytesIO()
            df.to_pickle(buffer)
            stock_data_list.append((code, name, buffer.getvalue()))

        config_dict = {
            'data_path': str(self.csv_manager.data_dir),
            'top_n': self.config.top_n,
            'hold_days': self.config.hold_days,
            'lookback_days': self.config.lookback_days,
            'filter_zge_risk': self.config.filter_zge_risk
        }

        # 预计算所有行业在所有交易日的热度（在主进程完成，避免子进程重复计算）
        print("主进程预计算行业热度...")

        # 首先预计算所有股票在所有交易日的成交额（加速行业热度计算）
        trading_dates_str = [d.strftime('%Y-%m-%d') for d in trading_dates]
        all_stocks = list(stock_data_dict.keys())

        # 使用持久化成交额缓存（历史数据不变，一次计算永久使用）
        print("  加载成交额缓存...")
        turnover_cache_manager = TurnoverCache()
        loaded_from_cache = turnover_cache_manager.load()

        # 增量更新：只计算缓存中没有的日期
        missing_dates = [d for d in trading_dates_str if not turnover_cache_manager.has_date(d)]
        if missing_dates:
            print(f"  需要补充 {len(missing_dates)} 个交易日的成交额数据...")
            for date_str in missing_dates:
                turnover_cache_manager.update_date(date_str, stock_data_dict, all_stocks)
            # 保存更新后的缓存
            turnover_cache_manager.save()
        else:
            print(f"  ✓ 所有 {len(trading_dates_str)} 个交易日的成交额数据已在缓存中")

        # 获取完整的成交额缓存供行业热度计算使用
        turnover_cache = turnover_cache_manager.get_cache() or {}

        # 将turnover_cache添加到preloaded_data中，供子进程使用
        preloaded_data['turnover_cache'] = turnover_cache

        # 为每个行业预计算热度（使用成交额缓存加速）
        industry_cache = IndustryHeatCalculator(industry_fetcher)
        industries_to_compute = list(industry_fetcher.industry_stocks_map.keys())[:50]
        for ind_name in industries_to_compute:
            preloaded_data['industry_heats'][ind_name] = {}
            for date_str in trading_dates_str:
                heat = industry_cache.calculate_industry_heat_fast(
                    ind_name, date_str, turnover_cache, all_stocks)
                if heat is not None:
                    preloaded_data['industry_heats'][ind_name][date_str] = heat
        print(f"  ✓ 行业热度预计算完成：{len(preloaded_data['industry_heats'])} 个行业")

        # 准备任务列表（添加预加载数据）
        tasks = [
            (sel_date, stock_data_list, config_dict, self.config.top_n, self.config.hold_days, preloaded_data)
            for sel_date in trading_dates
        ]

        # 使用进程池并行处理
        num_workers = max_workers if max_workers else cpu_count()
        print(f"开始并行处理 {len(trading_dates)} 个交易日，使用 {num_workers} 个进程...\n")

        all_results = []
        daily_summary = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_date = {
                executor.submit(_process_single_day, *task): task[0]
                for task in tasks
            }

            # 收集结果
            completed = 0
            # 每个交易日都打印进度（改为更频繁的进度提示）
            for future in as_completed(future_to_date):
                sel_date = future_to_date[future]
                completed += 1

                try:
                    day_results, day_summary = future.result()

                    if day_results:
                        all_results.extend(day_results)
                        if day_summary:
                            daily_summary.append(day_summary)

                    # 打印进度（每个任务完成后都打印）
                    print(f"[{completed}/{len(trading_dates)}] 已完成 {sel_date}")

                except Exception as e:
                    print(f"  ⚠️ 处理 {sel_date} 时出错：{e}")

        # 按日期排序结果
        daily_summary.sort(key=lambda x: x['date'])

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

        # 在文件名中包含回看天数参数和Z哥风控过滤标识
        lookback_days = getattr(self.config, 'lookback_days', 45)  # 默认值为45
        zge_suffix = '_zge_filtered' if self.config.filter_zge_risk else ''
        json_file = output_dir / f'daily_backtest_fast_lb{lookback_days}{zge_suffix}_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'start_date': str(self.config.start_date),
                    'end_date': str(self.config.end_date),
                    'top_n': self.config.top_n,
                    'hold_days': self.config.hold_days,
                    'lookback_days': self.config.lookback_days,
                    'filter_zge_risk': self.config.filter_zge_risk
                },
                'summary': {
                    'total_trades': len(all_results),
                    'win_rate': sum(1 for r in all_results if r['return_pct'] > 0) / len(all_results) * 100 if all_results else 0,
                    'avg_return': sum(r['return_pct'] for r in all_results) / len(all_results) if all_results else 0
                },
                'daily_summary': daily_summary,
                'all_results': all_results
            }, f, ensure_ascii=False, indent=2)

        csv_file = output_dir / f'daily_backtest_fast_lb{lookback_days}{zge_suffix}_{timestamp}.csv'
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
                '最大跌幅': f"{r['max_drop_pct']:+.2f}%" if r.get('max_drop_pct') is not None else 'N/A',
                '最大跌幅日期': r.get('max_drop_date', '') or '-',
                '最大跌幅天数': f"第{r['max_drop_day']}天" if r.get('max_drop_day') is not None else 'N/A',
                '卖出日期': r['sell_date'],
                '持有天数': r['hold_days'],
                '触发 +10% 日期': r.get('trigger_10pct_date', '') or '-',
                '触发 +5% 日期': r.get('trigger_5pct_date', '') or '-',  # 新增
                '触发 -2% 日期': r.get('trigger_neg2pct_date', '') or '-',
                '触发 -4% 日期': r.get('trigger_neg4pct_date', '') or '-',
                '达到 +10% 天数': f"第{r['trigger_10pct_day']}天" if r.get('trigger_10pct_day') not in [None, ''] else '-',
                '达到 +5% 天数': f"第{r['trigger_5pct_day']}天" if r.get('trigger_5pct_day') not in [None, ''] else '-',  # 新增
                '触发顺序': r.get('first_reach_order', '') or '-',
                '是否曾跌破成本价': '是' if r.get('ever_below_zero') else '否',
                # 行业热度字段
                '行业': r.get('industry', '未知'),
                '行业热度_买入日': r.get('industry_heat_buy', ''),
                '行业热度_10pct 日': r.get('industry_heat_10pct', ''),
                '行业热度_5pct 日': r.get('industry_heat_5pct', ''),  # 新增
                '行业热度_neg2pct 日': r.get('industry_heat_neg2pct', ''),
                '行业热度_neg4pct 日': r.get('industry_heat_neg4pct', ''),
                '行业热度_卖出日': r.get('industry_heat_sell', '')
            })

        simple_csv = output_dir / f'daily_backtest_fast_readable_lb{lookback_days}{zge_suffix}_{timestamp}.csv'
        df_simple = pd.DataFrame(simple_rows)
        df_simple.to_csv(simple_csv, index=False, encoding='utf-8-sig')

        print(f"\n💾 结果已保存:")
        print(f"  - JSON: {json_file}")
        print(f"  - CSV:  {csv_file}")
        print(f"  - 可读 CSV: {simple_csv}")

        # 更新汇总文件，便于不同回看天数的横向对比
        self._update_summary_file(output_dir, lookback_days, timestamp, json_file, csv_file, simple_csv)

        # 自动生成分析报告
        print("\n📊 正在生成分析报告...")
        try:
            from analyze_daily_backtest_fast import analyze_and_generate_report
            report_path = analyze_and_generate_report(
                csv_file=simple_csv,
                json_file=json_file
            )
            if report_path:
                print(f"  - 分析报告：{report_path}")
                # 更新汇总文件，包含分析报告路径
                self._update_summary_file_with_report(output_dir, report_path)
        except Exception as e:
            print(f"  ⚠️ 生成分析报告失败：{e}")

    def _update_summary_file(self, output_dir, lookback_days, timestamp, json_file, csv_file, simple_csv):
        """更新汇总文件，记录所有回测结果用于横向对比"""
        import os
        from pathlib import Path

        summary_file = output_dir / 'backtest_summary.csv'

        # 如果summary文件不存在，则写入表头
        if not summary_file.exists():
            with open(summary_file, 'w', encoding='utf-8-sig') as f:
                f.write('时间戳,回看天数,开始日期,结束日期,TopN,持有天数,市值门槛(亿),JSON文件路径,CSV文件路径,可读CSV文件路径,报告文件路径\n')

        # 计算市值门槛的亿数值
        cap_threshold_billion = self.config.cap_threshold / 1e8

        # 写入本次回测的信息
        with open(summary_file, 'a', encoding='utf-8-sig') as f:
            f.write(f'{timestamp},{lookback_days},{self.config.start_date},{self.config.end_date},{self.config.top_n},{self.config.hold_days},{cap_threshold_billion:.1f},"{json_file.name}","{csv_file.name}","{simple_csv.name}",""\n')

    def _update_summary_file_with_report(self, output_dir, report_path):
        """更新汇总文件，添加分析报告路径"""
        import os
        from pathlib import Path
        import pandas as pd

        summary_file = output_dir / 'backtest_summary.csv'

        # 读取现有CSV并更新最新的记录
        if summary_file.exists():
            df = pd.read_csv(summary_file, encoding='utf-8-sig')

            # 找到最新的记录（最后一行）并更新其报告路径
            if not df.empty:
                # 确保 ReportPath 列存在（以防旧的汇总文件没有此列）
                if '报告文件路径' not in df.columns:
                    df['报告文件路径'] = ''

                df.iloc[-1, df.columns.get_loc('报告文件路径')] = str(report_path.name)

                # 重新保存文件
                df.to_csv(summary_file, index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description='每日快速回测 - B1 完美图形匹配版（多进程并行）')
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=10, help='每次选股数量 (默认：10)')
    parser.add_argument('--hold-days', type=int, default=30, help='持有天数 (默认：30)')
    parser.add_argument('--cap', type=float, default=40, help='市值门槛 (单位：亿，默认：40)')
    parser.add_argument('--workers', type=int, default=None, help='并行工作进程数 (默认：CPU 核心数)')
    parser.add_argument('--lookback-days', type=int, default=40, help='B1完美图形匹配的回看天数 (默认：40)')
    parser.add_argument(
        '--filter-zge-risk',
        action='store_true',
        default=False,
        help='开启后回测过程中剔除所有触发Z哥风控规则的股票，不参与回测计算，默认关闭所有股票都参与回测'
    )

    args = parser.parse_args()

    config = FastDailyBacktestConfig(
        start_date=datetime.strptime(args.start, '%Y-%m-%d').date(),
        end_date=datetime.strptime(args.end, '%Y-%m-%d').date(),
        top_n=args.top_n,
        hold_days=args.hold_days,
        cap_threshold=args.cap * 1e8,
        lookback_days=args.lookback_days,
        filter_zge_risk=args.filter_zge_risk
    )

    backtester = FastDailyBacktester(config)
    backtester._load_stock_names()
    all_results, daily_summary = backtester.run_backtest(max_workers=args.workers)
    backtester.print_report(all_results, daily_summary)


if __name__ == '__main__':
    main()
