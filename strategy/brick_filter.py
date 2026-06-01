"""
砖型图过滤器 - 在现有选股结果上叠加砖型图形态过滤

三层筛选逻辑:
  1. 砖型图形态: 红砖转绿砖的力度判断（下跌后首次强力反弹拐点）
  2. 知行线位置: 收盘价 < 短期趋势线 × ratio，且短期趋势线 > 多空线
  3. 周线多头排列: 周线级别 MA 短 > 中 > 长

默认关闭，通过 --brick-filter 命令行参数开启。
"""
import numpy as np
import pandas as pd
from utils.brick_chart import (
    compute_brick_chart,
    check_brick_pattern,
    compute_weekly_ma_bull,
)
from utils.technical import calculate_zhixing_trend


class BrickFilter:
    """砖型图三层过滤器"""

    DEFAULT_PARAMS = {
        # 砖型图计算参数
        'brick_n': 4,
        'brick_m1': 4,
        'brick_m2': 6,
        'brick_m3': 6,
        'brick_t': 4.0,
        # 砖型图形态参数
        'daily_return_threshold': 0.05,
        'brick_growth_ratio': 1.0,
        'min_prior_green_bars': 1,
        # 知行线参数
        'zxdq_ratio': 1.0,
        'require_zxdq_gt_zxdkx': True,
        # 周线参数
        'require_weekly_ma_bull': True,
        'wma_short': 20,
        'wma_mid': 60,
        'wma_long': 120,
    }

    def __init__(self, params=None):
        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)

    def filter_stock(self, df):
        """
        对单只股票进行砖型图三层过滤

        :param df: 带 high/low/close/date 列的 DataFrame（降序，最新在前）
                   如果已有 short_term_trend / bull_bear_line 列会直接使用
        :return: (passed: bool, info: dict)
        """
        if df.empty or len(df) < 30:
            return False, {'reason': '数据不足'}

        info = {}

        # ── 第一层：砖型图形态 ──
        brick = compute_brick_chart(
            df,
            n=self.params['brick_n'],
            m1=self.params['brick_m1'],
            m2=self.params['brick_m2'],
            m3=self.params['brick_m3'],
            t=self.params['brick_t'],
        )

        passed, pattern_info = check_brick_pattern(
            df, brick,
            daily_return_threshold=self.params['daily_return_threshold'],
            brick_growth_ratio=self.params['brick_growth_ratio'],
            min_prior_green_bars=self.params['min_prior_green_bars'],
        )
        info.update(pattern_info)
        if not passed:
            info['fail_layer'] = '形态'
            return False, info

        # ── 第二层：知行线位置 ──
        if 'short_term_trend' in df.columns and 'bull_bear_line' in df.columns:
            zxdq = df['short_term_trend']
            zxdkx = df['bull_bear_line']
        else:
            trend_df = calculate_zhixing_trend(df)
            zxdq = trend_df['short_term_trend']
            zxdkx = trend_df['bull_bear_line']

        latest_close = float(df['close'].iloc[0])
        latest_zxdq = float(zxdq.iloc[0])
        latest_zxdkx = float(zxdkx.iloc[0])

        if not np.isfinite(latest_zxdq) or not np.isfinite(latest_zxdkx):
            return False, {'reason': '知行线数据无效', 'fail_layer': '知行线'}

        # 收盘价 < zxdq × ratio
        if self.params['zxdq_ratio'] is not None:
            if latest_close >= latest_zxdq * self.params['zxdq_ratio']:
                info['fail_layer'] = '知行线'
                info['fail_reason'] = (
                    f'收盘({latest_close:.2f}) >= zxdq×{self.params["zxdq_ratio"]}'
                    f'({latest_zxdq * self.params["zxdq_ratio"]:.2f})'
                )
                return False, info

        # zxdq > zxdkx
        if self.params['require_zxdq_gt_zxdkx']:
            if latest_zxdq <= latest_zxdkx:
                info['fail_layer'] = '知行线'
                info['fail_reason'] = f'zxdq({latest_zxdq:.2f}) <= zxdkx({latest_zxdkx:.2f})'
                return False, info

        info['zxdq'] = round(latest_zxdq, 2)
        info['zxdkx'] = round(latest_zxdkx, 2)

        # ── 第三层：周线多头排列 ──
        if self.params['require_weekly_ma_bull']:
            wma_bull = compute_weekly_ma_bull(
                df,
                ma_periods=(self.params['wma_short'],
                            self.params['wma_mid'],
                            self.params['wma_long']),
            )
            if not bool(wma_bull.iloc[0]):
                info['fail_layer'] = '周线'
                info['fail_reason'] = '周线非多头排列'
                return False, info
            info['weekly_ma_bull'] = True

        info['passed'] = True
        return True, info
