#!/usr/bin/env python3
"""
行业数据获取模块

功能：
1. 从 akshare 获取股票所属行业的完整映射关系
2. 计算行业热度指标（复合版：成交额占比 + 成交额环比）

热度计算公式：
行业热度 = 0.6 × 成交额占比分位数 + 0.4 × 成交额环比
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class IndustryFetcher:
    """
    行业数据获取器

    从 akshare 获取东财/同花顺行业分类数据
    """

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'data' / 'industry_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 股票 -> 行业 映射
        self.stock_industry_map = None
        # 行业 -> 股票列表 映射
        self.industry_stocks_map = None

    def load_industry_mapping(self, force_refresh=False):
        """
        加载行业映射关系（股票 <-> 行业）
        优先从缓存读取，缓存不存在或 force_refresh=True 则从接口获取
        """
        cache_file = self.cache_dir / 'industry_mapping.json'

        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stock_industry_map = data.get('stock_industry', {})
                    self.industry_stocks_map = data.get('industry_stocks', {})
                print(f"  ✓ 从缓存加载行业映射：{len(self.stock_industry_map)} 只股票")
                return True
            except Exception as e:
                print(f"  ⚠️ 读取缓存失败：{e}")

        # 从 akshare 获取
        if self._fetch_industry_mapping_from_akshare():
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'stock_industry': self.stock_industry_map,
                    'industry_stocks': self.industry_stocks_map
                }, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 获取并缓存行业映射：{len(self.stock_industry_map)} 只股票")
            return True
        return False

    def _fetch_industry_mapping_from_akshare(self):
        """
        从 akshare 获取东财行业分类数据
        """
        try:
            import akshare as ak

            print("  正在从 akshare 获取行业分类数据...")

            # 获取东财行业板块数据
            df = ak.stock_board_industry_name_em()

            if df.empty:
                print("  ⚠️ 获取行业板块数据失败")
                return False

            # 获取每个行业的成分股
            self.stock_industry_map = {}
            self.industry_stocks_map = {}

            for _, row in df.iterrows():
                industry_name = row['板块名称']

                try:
                    # 获取该行业的成分股
                    cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)

                    if not cons_df.empty:
                        stock_codes = cons_df['代码'].astype(str).tolist()
                        self.industry_stocks_map[industry_name] = stock_codes

                        for code in stock_codes:
                            # 只记录第一个行业（避免重复）
                            if code not in self.stock_industry_map:
                                self.stock_industry_map[code] = industry_name

                except Exception as e:
                    print(f"  ⚠️ 获取行业 {industry_name} 成分股失败：{e}")
                    continue

            print(f"  ✓ 获取到 {len(self.industry_stocks_map)} 个行业，{len(self.stock_industry_map)} 只股票")
            return True

        except ImportError:
            print("  ⚠️ akshare 未安装，使用内置行业映射")
            # 返回内置的简化映射
            self._load_builtin_industry_map()
            return True
        except Exception as e:
            print(f"  ⚠️ 获取行业数据失败：{e}")
            # 返回内置的简化映射
            self._load_builtin_industry_map()
            return True

    def _load_builtin_industry_map(self):
        """
        加载内置的行业映射（当 akshare 不可用时）
        覆盖主要行业的龙头股
        """
        builtin_map = {
            '600519': '白酒', '000858': '白酒', '000568': '白酒', '000799': '白酒',
            '600030': '证券', '601688': '证券', '000776': '证券', '000686': '证券',
            '601398': '银行', '601288': '银行', '601939': '银行', '000001': '银行',
            '300750': '新能源', '002594': '新能源', '300014': '新能源', '002460': '新能源',
            '600276': '医药', '000538': '医药', '300760': '医药', '603259': '医药',
            '002415': '电子', '000725': '电子', '600745': '电子', '000100': '电子',
            '002230': '计算机', '600570': '计算机', '300496': '计算机', '002268': '计算机',
            '000063': '通信', '600487': '通信', '002583': '通信', '300628': '通信',
            '000625': '汽车', '000338': '汽车', '600104': '汽车', '002594': '汽车',
            '600309': '化工', '002648': '化工', '600346': '化工', '002450': '化工',
            '600031': '机械', '000425': '机械', '601100': '机械', '002459': '机械',
            '600765': '军工', '000768': '军工', '600893': '军工', '002179': '军工',
            '000002': '房地产', '600048': '房地产', '601155': '房地产', '001979': '房地产',
            '000786': '建材', '002271': '建材', '600585': '建材', '002392': '建材',
            '000898': '钢铁', '600019': '钢铁', '600231': '钢铁', '000709': '钢铁',
            '601088': '煤炭', '600188': '煤炭', '601699': '煤炭', '000983': '煤炭',
            '601857': '石油', '600028': '石油', '000852': '石油', '002738': '石油',
            '600900': '电力', '600886': '电力', '000539': '电力', '000027': '电力',
            '688981': '半导体', '002371': '半导体', '603986': '半导体', '300782': '半导体',
            '000895': '消费', '002714': '消费', '002027': '消费', '002511': '消费',
        }
        self.stock_industry_map = builtin_map

        # 反向构建 industry_stocks_map
        self.industry_stocks_map = {}
        for code, industry in builtin_map.items():
            if industry not in self.industry_stocks_map:
                self.industry_stocks_map[industry] = []
            self.industry_stocks_map[industry].append(code)

        print(f"  ✓ 使用内置映射：{len(self.industry_stocks_map)} 个行业，{len(self.stock_industry_map)} 只股票")

    def get_industry_for_stock(self, stock_code):
        """
        获取股票所属行业
        :param stock_code: 股票代码 (如 '600519')
        :return: 行业名称，找不到返回 None
        """
        if self.stock_industry_map is None:
            if not self.load_industry_mapping():
                return None

        return self.stock_industry_map.get(stock_code)

    def get_stocks_in_industry(self, industry_name):
        """
        获取行业内的所有股票代码
        :return: 股票代码列表
        """
        if self.industry_stocks_map is None:
            self.load_industry_mapping()

        return self.industry_stocks_map.get(industry_name, [])

    def get_all_industries(self):
        """
        获取所有行业名称
        :return: 行业名称列表
        """
        if self.industry_stocks_map is None:
            self.load_industry_mapping()

        return list(self.industry_stocks_map.keys())


class IndustryHeatCalculator:
    """
    行业热度计算器

    热度计算公式：
    行业热度 = 0.6 × 成交额占比 + 0.4 × 成交额环比

    其中：
    - 成交额占比 = 行业成交额 / 全市场成交额
    - 成交额环比 = (今日成交额 - 昨日成交额) / 昨日成交额
    """

    def __init__(self, industry_fetcher=None):
        if industry_fetcher is None:
            self.fetcher = IndustryFetcher()
        else:
            self.fetcher = industry_fetcher

    def _get_stock_turnover(self, code, date, stock_data_dict):
        """
        获取单只股票在指定日期的成交额（元）
        """
        if code not in stock_data_dict:
            return None

        name, df = stock_data_dict[code]
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 找到最接近指定日期的数据（小于等于 date 的最大日期）
        mask = df['date'] <= pd.to_datetime(date)
        day_df = df[mask].head(1)

        if day_df.empty:
            return None

        # 尝试多个可能的列名
        for col in ['amount', '成交额', 'turnover', 'amt']:
            if col in day_df.columns and pd.notna(day_df.iloc[0][col]):
                return float(day_df.iloc[0][col])

        # 如果没有成交额列，用成交量 × 收盘价估算
        if 'volume' in day_df.columns and 'close' in day_df.columns:
            vol = day_df.iloc[0]['volume']
            price = day_df.iloc[0]['close']
            if pd.notna(vol) and pd.notna(price):
                return float(vol * price * 100)

        return None

    def _get_yesterday_turnover(self, code, date, stock_data_dict):
        """
        获取前一个交易日的成交额
        """
        if code not in stock_data_dict:
            return None

        name, df = stock_data_dict[code]
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 找到小于 date 的最大日期
        mask = df['date'] < pd.to_datetime(date)
        day_df = df[mask].head(1)

        if day_df.empty:
            return None

        for col in ['amount', '成交额', 'turnover', 'amt']:
            if col in day_df.columns and pd.notna(day_df.iloc[0][col]):
                return float(day_df.iloc[0][col])

        if 'volume' in day_df.columns and 'close' in day_df.columns:
            vol = day_df.iloc[0]['volume']
            price = day_df.iloc[0]['close']
            if pd.notna(vol) and pd.notna(price):
                return float(vol * price * 100)

        return None

    def calculate_industry_heat(self, industry_name, date, stock_data_dict, all_market_stocks=None):
        """
        计算行业在指定日期的热度（复合版）

        :param industry_name: 行业名称
        :param date: 日期 (str 或 datetime.date)
        :param stock_data_dict: 股票数据字典 {code: (name, df)}
        :param all_market_stocks: 全市场股票代码列表（可选，用于计算分母）
        :return: 热度分数 (0-100)，计算失败返回 None
        """
        # 获取行业成分股
        industry_stocks = self.fetcher.get_stocks_in_industry(industry_name)

        if not industry_stocks:
            return None

        # 1. 计算行业今日总成交额
        industry_turnover = 0
        industry_yesterday_turnover = 0
        valid_count = 0
        valid_yesterday_count = 0

        for code in industry_stocks:
            today = self._get_stock_turnover(code, date, stock_data_dict)
            if today is not None:
                industry_turnover += today
                valid_count += 1

            yesterday = self._get_yesterday_turnover(code, date, stock_data_dict)
            if yesterday is not None:
                industry_yesterday_turnover += yesterday
                valid_yesterday_count += 1

        if valid_count == 0 or industry_turnover == 0:
            return None

        # 2. 计算全市场今日总成交额
        if all_market_stocks is None:
            all_market_stocks = list(stock_data_dict.keys())

        market_turnover = 0
        market_yesterday_turnover = 0
        market_valid_count = 0

        for code in all_market_stocks:
            today = self._get_stock_turnover(code, date, stock_data_dict)
            if today is not None:
                market_turnover += today
                market_valid_count += 1

            yesterday = self._get_yesterday_turnover(code, date, stock_data_dict)
            if yesterday is not None:
                market_yesterday_turnover += yesterday

        if market_valid_count == 0 or market_turnover == 0:
            return None

        # 3. 计算成交额占比
        turnover_ratio = industry_turnover / market_turnover

        # 4. 计算成交额环比
        if valid_yesterday_count > 0 and industry_yesterday_turnover > 0:
            turnover_change = (industry_turnover - industry_yesterday_turnover) / industry_yesterday_turnover
        else:
            turnover_change = 0

        # 5. 标准化处理
        # 成交额占比通常 0.01-0.15，转换为 0-100 分数
        # 占比 10% = 100 分，占比 5% = 50 分
        ratio_score = turnover_ratio * 1000
        ratio_score = min(100, max(0, ratio_score))

        # 成交额环比范围 -1.0 到 +2.0，转换为 0-100 分数
        # 0% = 50 分，+50% = 100 分，-50% = 0 分
        change_score = 50 + turnover_change * 50
        change_score = min(100, max(0, change_score))

        # 6. 复合热度分数
        heat_score = 0.6 * ratio_score + 0.4 * change_score

        return round(heat_score, 2)

    def get_industry_heat_series(self, industry_name, start_date, end_date, stock_data_dict):
        """
        获取行业在一段时间内的热度序列
        :return: DataFrame(date, heat)
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        heats = []

        for date in dates:
            heat = self.calculate_industry_heat(industry_name, date, stock_data_dict)
            heats.append(heat if heat is not None else np.nan)

        return pd.DataFrame({
            'date': dates,
            'industry': industry_name,
            'heat': heats
        })


# 便捷函数
def get_industry_heat(industry_name, date, stock_data_dict):
    """
    便捷函数：计算行业热度
    """
    fetcher = IndustryFetcher()
    fetcher.load_industry_mapping()
    calc = IndustryHeatCalculator(fetcher)
    return calc.calculate_industry_heat(industry_name, date, stock_data_dict)


def get_industry_for_stock(stock_code):
    """
    便捷函数：获取股票所属行业
    """
    fetcher = IndustryFetcher()
    fetcher.load_industry_mapping()
    return fetcher.get_industry_for_stock(stock_code)


if __name__ == '__main__':
    print("行业数据获取模块测试")
    print("=" * 50)

    fetcher = IndustryFetcher()
    if fetcher.load_industry_mapping():
        print(f"\n覆盖行业数量：{len(fetcher.get_all_industries())}")
        print(f"覆盖股票数量：{len(fetcher.stock_industry_map)}")

        # 测试获取某股票的行业
        test_code = '600519'  # 贵州茅台
        industry = fetcher.get_industry_for_stock(test_code)
        print(f"\n{test_code} 所属行业：{industry}")

        # 测试获取行业成分股
        if industry:
            stocks = fetcher.get_stocks_in_industry(industry)
            print(f"{industry} 行业内股票数量：{len(stocks)}")
