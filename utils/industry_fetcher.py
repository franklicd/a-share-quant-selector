#!/usr/bin/env python3
"""
行业数据获取模块

功能：
1. 从多源获取股票所属行业的完整映射关系（东方财富 HTTP 接口优先，akshare 备选）
2. 计算行业热度指标（复合版：成交额占比 + 成交额环比）

热度计算公式：
行业热度 = 0.6 × 成交额占比分位数 + 0.4 × 成交额环比

数据源优先级：
1. 本地缓存（7 天内有效）
2. 东方财富 HTTP 接口（稳定，不依赖 akshare）
3. akshare（备选）
4. 内置映射（降级，覆盖约 300 只主要股票）
"""
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入内置行业映射（约 300 只股票）
from utils.industry_builtin_map import BUILTIN_INDUSTRY_MAP


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

    def load_industry_mapping(self, force_refresh=False, cache_days=180):
        """
        加载行业映射关系（股票 <-> 行业）
        优先从缓存读取，缓存过期或 force_refresh=True 则重新获取

        Args:
            force_refresh: 强制刷新
            cache_days: 缓存有效期（天），默认 180 天（行业分类变化不频繁）
        """
        cache_file = self.cache_dir / 'industry_mapping.json'
        cache_meta_file = self.cache_dir / 'industry_mapping_meta.json'

        # 检查缓存是否有效
        cache_valid = False
        if cache_file.exists() and not force_refresh:
            # 检查缓存时间
            cache_age_days = None
            if cache_meta_file.exists():
                try:
                    with open(cache_meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        cache_date = meta.get('cache_date')
                        if cache_date:
                            cache_dt = datetime.fromisoformat(cache_date)
                            cache_age_days = (datetime.now() - cache_dt).days
                except:
                    pass

            # 缓存未过期，直接加载
            if cache_age_days is not None and cache_age_days < cache_days:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.stock_industry_map = data.get('stock_industry', {})
                        self.industry_stocks_map = data.get('industry_stocks', {})
                    # 合并内置映射中独有的股票（扩展覆盖范围）
                    self._merge_builtin_industry_map()
                    # 使用进程 ID 控制打印，避免多进程重复输出
                    import os
                    worker_id = os.environ.get('WORKER_ID', os.getpid())
                    if not hasattr(IndustryFetcher, '_cache_load_printed'):
                        IndustryFetcher._cache_load_printed = set()
                    if worker_id not in IndustryFetcher._cache_load_printed:
                        IndustryFetcher._cache_load_printed.add(worker_id)
                        if len(IndustryFetcher._cache_load_printed) == 1:
                            print(f"  ✓ 从缓存加载行业映射：{len(self.stock_industry_map)} 只股票 (缓存剩余{cache_days - cache_age_days}天有效)")
                    return True
                except Exception as e:
                    print(f"  ⚠️ 读取缓存失败：{e}")
            elif cache_age_days is not None:
                print(f"  ℹ️  缓存已过期 ({cache_age_days}天前)，将刷新行业数据")

        # 从 HTTP 接口获取（优先）
        if self._fetch_industry_mapping_from_http():
            self._save_cache(cache_file, cache_meta_file)
            return True

        # HTTP 失败，尝试 akshare
        if self._fetch_industry_mapping_from_akshare():
            self._save_cache(cache_file, cache_meta_file)
            return True

        # 都失败，使用内置映射
        self._load_builtin_industry_map()
        return True

    def _save_cache(self, cache_file, cache_meta_file):
        """保存行业映射到缓存"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'stock_industry': self.stock_industry_map,
                    'industry_stocks': self.industry_stocks_map
                }, f, ensure_ascii=False, indent=2)

            # 保存元数据（缓存时间）
            with open(cache_meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'cache_date': datetime.now().isoformat(),
                    'stock_count': len(self.stock_industry_map),
                    'industry_count': len(self.industry_stocks_map)
                }, f, indent=2)

            print(f"  ✓ 获取并缓存行业映射：{len(self.stock_industry_map)} 只股票，{len(self.industry_stocks_map)} 个行业")
        except Exception as e:
            print(f"  ⚠️ 保存缓存失败：{e}")

    def _fetch_industry_mapping_from_http(self):
        """
        从东方财富 HTTP 接口获取行业分类数据

        接口说明：
        - 使用东方财富证券行情 API
        - 获取东财行业板块成分股数据
        - 不依赖 akshare，直接 HTTP 请求

        数据源优先级：
        1. 东财行业板块接口（行业概念）
        2. 东财概念板块接口（概念主题）
        """
        try:
            print("  正在从东方财富接口获取行业分类数据...")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Referer': 'https://quote.eastmoney.com/',
            }

            self.stock_industry_map = {}
            self.industry_stocks_map = {}

            # 方法 1：获取东财行业数据（通过 centerapi 接口）
            # 这个接口返回行业板块列表
            industries = self._fetch_dfcf_industries(headers)

            if industries:
                print(f"  ✓ 获取到 {len(industries)} 个行业板块")

                # 遍历每个行业，获取成分股
                for idx, industry_info in enumerate(industries):
                    industry_name = industry_info.get('name', '')
                    industry_id = industry_info.get('id', '')

                    if not industry_name or not industry_id:
                        continue

                    try:
                        # 获取该行业的成分股
                        stocks = self._fetch_industry_stocks(industry_id, industry_name, headers)

                        if stocks:
                            self.industry_stocks_map[industry_name] = stocks
                            for code in stocks:
                                if code not in self.stock_industry_map:
                                    self.stock_industry_map[code] = industry_name

                    except Exception as e:
                        print(f"  ⚠️ 获取行业 {industry_name} 成分股失败：{e}")
                        continue

                    # 限速，每 20 个请求暂停一下
                    if (idx + 1) % 20 == 0:
                        time.sleep(0.2)

                if self.stock_industry_map:
                    print(f"  ✓ 获取到 {len(self.industry_stocks_map)} 只股票，{len(self.industry_stocks_map)} 个行业")
                    return True

            # 方法 1 失败，尝试简化的批量接口
            print("  尝试批量获取行业数据...")
            return self._fetch_industries_batch(headers)

        except requests.RequestException as e:
            print(f"  ⚠️ HTTP 请求失败：{e}")
            return False
        except Exception as e:
            print(f"  ⚠️ 获取行业数据异常：{e}")
            return False

    def _fetch_dfcf_industries(self, headers, max_retries=3):
        """
        获取东财行业板块列表

        返回：[{'name': '行业名', 'id': '板块 ID'}, ...]
        """
        for attempt in range(max_retries):
            try:
                # 东方财富行业板块列表接口
                url = "https://push2.eastmoney.com/api/qt/clist/get"
                params = {
                    'pn': '1',
                    'pz': '500',  # 每页 500 条
                    'po': '1',
                    'np': '1',
                    'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                    'fltt': '2',
                    'invt': '2',
                    'fid': 'f3',
                    'fs': 'm:90 t:3',  # 行业板块
                    'fields': 'f12,f13,f14,f62',  # 代码，市场，名称，市值等
                    '_': str(int(time.time() * 1000))
                }

                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()

                data = resp.json()
                if data.get('data') and data['data'].get('diff'):
                    industries = []
                    for item in data['data']['diff']:
                        industries.append({
                            'name': item.get('f14', ''),  # 行业名称
                            'id': item.get('f12', ''),    # 行业代码
                            'market': item.get('f13', '') # 市场代码
                        })
                    return industries
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))  # 递增延迟
                    continue
                print(f"    ⚠️ 获取行业列表失败（尝试{attempt+1}/{max_retries}）: {e}")
        return []

    def _fetch_industry_stocks(self, industry_id, industry_name, headers):
        """
        获取特定行业的成分股列表

        Args:
            industry_id: 行业 ID（如 BK0473）
            industry_name: 行业名称
            headers: HTTP 请求头

        Returns:
            股票代码列表
        """
        try:
            # 东方财富行业成分股接口
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                'pn': '1',
                'pz': '500',  # 每页 500 条，大部分行业成分股不超过 500
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                # fs 参数：板块 ID，格式为 b+板块代码
                'fs': f'b:{industry_id}',
                'fields': 'f12,f14',  # 代码，名称
                '_': str(int(time.time() * 1000))
            }

            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            if data.get('data') and data['data'].get('diff'):
                stocks = []
                for item in data['data']['diff']:
                    code = item.get('f12', '')
                    if code and len(code) == 6 and code.isdigit():
                        stocks.append(code)
                return stocks
        except Exception as e:
            print(f"    ⚠️ 获取 {industry_name} 成分股失败：{e}")
        return []

    def _fetch_industries_batch(self, headers):
        """
        批量获取行业数据（简化版本，使用概念板块数据）
        """
        try:
            # 获取概念板块数据作为行业分类的补充
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                'pn': '1',
                'pz': '200',
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                'fs': 'm:90 t:1',  # 概念板块
                'fields': 'f12,f13,f14',
                '_': str(int(time.time() * 1000))
            }

            resp = requests.get(url, params=params, headers=headers, timeout=30)
            data = resp.json()

            if data.get('data') and data['data'].get('diff'):
                concepts = data['data']['diff']
                # 取前 50 个热门概念
                for concept in concepts[:50]:
                    concept_name = concept.get('f14', '')
                    concept_id = concept.get('f12', '')

                    if concept_name and concept_id:
                        stocks = self._fetch_industry_stocks(concept_id, concept_name, headers)
                        if stocks:
                            self.industry_stocks_map[concept_name] = stocks
                            for code in stocks:
                                if code not in self.stock_industry_map:
                                    self.stock_industry_map[code] = concept_name

                if self.stock_industry_map:
                    print(f"  ✓ 获取到 {len(self.stock_industry_map)} 只股票，{len(self.industry_stocks_map)} 个概念板块")
                    return True

        except Exception as e:
            print(f"  ⚠️ 批量获取失败：{e}")

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
        加载内置的行业映射（当 akshare 和 HTTP 接口都不可用时）
        覆盖约 300 只主要 A 股股票
        """
        self.stock_industry_map = BUILTIN_INDUSTRY_MAP.copy()

        # 反向构建 industry_stocks_map
        self.industry_stocks_map = {}
        for code, industry in self.stock_industry_map.items():
            if industry not in self.industry_stocks_map:
                self.industry_stocks_map[industry] = []
            self.industry_stocks_map[industry].append(code)

        print(f"  ✓ 使用内置映射：{len(self.industry_stocks_map)} 个行业，{len(self.stock_industry_map)} 只股票")

    def _merge_builtin_industry_map(self):
        """
        合并内置行业映射到当前映射中
        用于扩展缓存未覆盖的股票
        """
        initial_count = len(self.stock_industry_map)
        merged_count = 0

        # 合并股票 - 行业映射
        for code, industry in BUILTIN_INDUSTRY_MAP.items():
            if code not in self.stock_industry_map:
                self.stock_industry_map[code] = industry
                merged_count += 1

        # 重新构建 industry_stocks_map
        self.industry_stocks_map = {}
        for code, industry in self.stock_industry_map.items():
            if industry not in self.industry_stocks_map:
                self.industry_stocks_map[industry] = []
            self.industry_stocks_map[industry].append(code)

        if merged_count > 0:
            # 使用进程 ID 控制打印，避免多进程重复输出
            import os
            worker_id = os.environ.get('WORKER_ID', os.getpid())
            if not hasattr(IndustryFetcher, '_merged_printed'):
                IndustryFetcher._merged_printed = set()
            if worker_id not in IndustryFetcher._merged_printed:
                IndustryFetcher._merged_printed.add(worker_id)
                # 只在第一个进程中打印一次
                if len(IndustryFetcher._merged_printed) == 1:
                    print(f"  + 合并内置映射：新增 {merged_count} 只股票，总计 {len(self.stock_industry_map)} 只股票")

    def get_industry_for_stock(self, stock_code, refresh_if_missing=False):
        """
        获取股票所属行业
        :param stock_code: 股票代码 (如 '600519' 或 600519 或 2356)
        :param refresh_if_missing: 如果股票不在映射中，是否尝试从 HTTP 接口获取
        :return: 行业名称，找不到返回 None
        """
        if self.stock_industry_map is None:
            if not self.load_industry_mapping():
                return None

        # 统一转换为 6 位字符串格式
        if isinstance(stock_code, int):
            stock_code = str(stock_code).zfill(6)
        elif isinstance(stock_code, str) and len(stock_code) < 6:
            stock_code = stock_code.zfill(6)

        # 优先从缓存查找
        industry = self.stock_industry_map.get(stock_code)

        # 如果没找到且要求刷新，尝试从 HTTP 接口获取该股票的行业
        if industry is None and refresh_if_missing:
            industry = self._fetch_industry_for_single_stock(stock_code)
            if industry:
                # 更新缓存
                self.stock_industry_map[stock_code] = industry
                if industry not in self.industry_stocks_map:
                    self.industry_stocks_map[industry] = []
                self.industry_stocks_map[industry].append(stock_code)
                # 保存到缓存文件
                self._save_cache(
                    self.cache_dir / 'industry_mapping.json',
                    self.cache_dir / 'industry_mapping_meta.json'
                )

        return industry

    def _fetch_industry_for_single_stock(self, stock_code):
        """
        从东方财富接口获取单只股票的行业分类
        :param stock_code: 股票代码（6 位字符串）
        :return: 行业名称，获取失败返回 None
        """
        try:
            # 使用东方财富接口获取股票行业信息
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }

            # 确定市场代码
            if stock_code.startswith('6'):
                market = '1'  # 沪市
            else:
                market = '0'  # 深市

            secid = f'{market}.{stock_code}'

            # 获取股票基本信息
            url = 'https://push2.eastmoney.com/api/qt/stock/get'
            params = {
                'secid': secid,
                'fields': 'f58,f59,f124,f128'  # f124=行业，f128=概念
            }

            resp = requests.get(url, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    # f124 是行业名称
                    industry = data['data'].get('f124', '')
                    if industry:
                        # 清理行业名称，去掉"行业"等后缀
                        industry = industry.replace('行业', '').strip()
                        return industry if industry else None
        except Exception as e:
            print(f"  ⚠️ 获取 {stock_code} 行业失败：{e}")

        return None

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

        # 尝试多个可能的列名（要求值大于 0）
        for col in ['amount', '成交额', 'turnover', 'amt']:
            if col in day_df.columns:
                val = day_df.iloc[0][col]
                if pd.notna(val) and val > 0:
                    return float(val)

        # 如果没有成交额列或值为 0，用成交量 × 收盘价估算
        if 'volume' in day_df.columns and 'close' in day_df.columns:
            vol = day_df.iloc[0]['volume']
            price = day_df.iloc[0]['close']
            if pd.notna(vol) and pd.notna(price) and vol > 0:
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

        # 尝试多个可能的列名（要求值大于 0）
        for col in ['amount', '成交额', 'turnover', 'amt']:
            if col in day_df.columns:
                val = day_df.iloc[0][col]
                if pd.notna(val) and val > 0:
                    return float(val)

        # 如果没有成交额列或值为 0，用成交量 × 收盘价估算
        if 'volume' in day_df.columns and 'close' in day_df.columns:
            vol = day_df.iloc[0]['volume']
            price = day_df.iloc[0]['close']
            if pd.notna(vol) and pd.notna(price) and vol > 0:
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

    def calculate_industry_heat_fast(self, industry_name, date, turnover_cache, all_market_stocks=None):
        """
        使用预计算的成交额缓存计算行业热度（快速版本）

        :param industry_name: 行业名称
        :param date: 日期字符串 (str)
        :param turnover_cache: {date_str: {code: turnover}} 成交额缓存
        :param all_market_stocks: 全市场股票代码列表
        :return: 热度分数 (0-100)，计算失败返回 None
        """
        # 获取行业成分股
        industry_stocks = self.fetcher.get_stocks_in_industry(industry_name)

        if not industry_stocks:
            return None

        # 获取该日期的成交额数据
        date_turnover = turnover_cache.get(date, {})

        # 1. 计算行业今日总成交额
        industry_turnover = 0
        valid_count = 0

        for code in industry_stocks:
            if code in date_turnover:
                industry_turnover += date_turnover[code]
                valid_count += 1

        if valid_count == 0 or industry_turnover == 0:
            return None

        # 2. 计算全市场今日总成交额
        if all_market_stocks is None:
            all_market_stocks = list(date_turnover.keys())

        market_turnover = 0
        market_valid_count = 0

        for code in all_market_stocks:
            if code in date_turnover:
                market_turnover += date_turnover[code]
                market_valid_count += 1

        if market_valid_count == 0 or market_turnover == 0:
            return None

        # 3. 计算成交额占比
        turnover_ratio = industry_turnover / market_turnover

        # 4. 标准化处理
        # 成交额占比通常 0.01-0.15，转换为 0-100 分数
        ratio_score = turnover_ratio * 1000
        ratio_score = min(100, max(0, ratio_score))

        # 简单版本：不使用环比数据（因为需要前一天的缓存）
        # 假设环比为中性 50 分
        change_score = 50

        # 5. 复合热度分数
        heat_score = 0.6 * ratio_score + 0.4 * change_score

        return round(heat_score, 2)


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
