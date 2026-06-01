#!/usr/bin/env python3
"""
A股量化选股系统 - 主程序

使用方法:
    python main.py init      # 首次全量抓取
    python main.py update    # 每日增量更新（内部使用）
    python main.py select    # 执行选股
    python main.py run       # 完整流程（更新+选股+通知）
    python main.py schedule  # 启动定时调度
"""
import sys
import os
import argparse
import platform
import logging
from pathlib import Path
from datetime import datetime, time as dt_time
import time
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 版本信息
__version__ = "1.0.0"

# VPN 路由修复（必须在网络请求之前执行）
from utils import patch_requests_for_vpn
patch_requests_for_vpn()

# 全局模块导入（只在主进程导入一次，避免子进程重复加载）
from utils.akshare_fetcher import AKShareFetcher
from utils.csv_manager import CSVManager
from utils.feishu_notifier import FeishuNotifier
from strategy.strategy_registry import get_registry

# 可导入K线图模块，没有的话不影响核心选股逻辑
try:
    from utils.kline_chart import generate_kline_chart
    KLINE_CHART_AVAILABLE = True
except ImportError:
    KLINE_CHART_AVAILABLE = False
    generate_kline_chart = None
    logger.warning("K线图模块未安装，仅运行核心选股逻辑，图片功能不可用")

# 并行处理函数（全局函数，支持多进程序列化）
def _process_stock_parallel(args):
    code, df, strategy_class, strategy_params, name, category = args
    try:
        # 重新实例化策略（避免多进程共享状态问题）
        strategy = strategy_class(strategy_params)
        # 计算指标
        df_with_indicators = strategy.calculate_indicators(df)
        # 选股
        signal_list = strategy.select_stocks(df_with_indicators, name)
        return code, name, signal_list, df_with_indicators if signal_list else None
    except Exception as e:
        logger.error(f"处理 {code} 异常: {str(e)}", exc_info=False)
        return code, name, [], None

# 横盘整理模式的并行处理函数
def _process_consolidation_parallel(args):
    code, df, strategy_class, strategy_params, name, after_high_only = args
    try:
        strategy = strategy_class(strategy_params)
        df_with_indicators = strategy.calculate_indicators(df)
        signal_list = strategy.select_consolidation(df_with_indicators, name, after_high_only=after_high_only)
        return code, name, signal_list, df_with_indicators if signal_list else None
    except Exception as e:
        logger.error(f"处理 {code} 异常: {str(e)}", exc_info=False)
        return code, name, [], None


class QuantSystem:
    """量化系统主类"""
    
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        self.data_dir = self.config.get('data_dir', 'data')
        self.csv_manager = CSVManager(self.data_dir)
        self.fetcher = AKShareFetcher(self.data_dir)
        self.notifier = self._init_notifier()
        self.registry = get_registry("config/strategy_params.yaml")
    
    def _load_config(self, config_file):
        """加载配置文件"""
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
            return {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"配置文件 {config_file} 格式错误: {str(e)}，使用默认配置")
            return {}
        except Exception as e:
            logger.error(f"加载配置文件 {config_file} 失败: {str(e)}，使用默认配置")
            return {}
    
    def _init_notifier(self):
        """初始化通知器（使用OpenClaw内置推送，不需要配置webhook）"""
        return FeishuNotifier()  # 无需任何参数，直接发送到当前飞书对话
    
    def _fetch_date_turnover_from_csv(self, date_str, stock_codes):
        """
        从 CSV 文件获取指定日期的成交额数据
        :param date_str: 日期字符串 (YYYY-MM-DD)
        :param stock_codes: 股票代码列表
        :return: {code: turnover} 成交额字典
        """
        import pandas as pd
        date_turnover = {}
        valid_count = 0
        processed = 0

        # 从 CSV 读取股票数据
        for code in stock_codes:
            df = self.csv_manager.read_stock(code)
            if df is not None and not df.empty:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                # 检查是否有指定日期的数据
                latest_date = df['date'].max()
                if latest_date.strftime('%Y-%m-%d') >= date_str:
                    # 先尝试精确匹配指定日期
                    row = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]

                    # 如果不是交易日，使用最后一个实际交易日的数据
                    if row.empty:
                        # 找到小于等于指定日期的最大日期（最后一个交易日）
                        df_before = df[df['date'] <= pd.to_datetime(date_str)]
                        if not df_before.empty:
                            row = df_before.iloc[0:1]  # 取最新的一个交易日（倒序数据第一行）

                    if not row.empty:
                        val = row['amount'].iloc[0] if 'amount' in row.columns else None
                        if pd.notna(val) and val > 0:
                            date_turnover[code] = val
                            valid_count += 1
                        else:
                            vol = row['volume'].iloc[0] if 'volume' in row.columns else 0
                            price = row['close'].iloc[0] if 'close' in row.columns else 0
                            if vol > 0 and price > 0:
                                date_turnover[code] = vol * price * 100
                                valid_count += 1
            processed += 1
            if processed % 1000 == 0:
                print(f"    已处理 {processed}/{len(stock_codes)}...")

        print(f"  ✓ 从 CSV 获取到 {valid_count} 只股票的成交额")
        return date_turnover

    def _load_stock_names(self, stock_data):
        """加载股票名称（优先从CSV文件）"""
        names_file = Path(self.data_dir) / 'stock_names.json'
        
        # 尝试从网络获取
        try:
            stock_names = self.fetcher.get_all_market_stocks()
            if stock_names:
                # 保存到本地
                import json
                with open(names_file, 'w', encoding='utf-8') as f:
                    json.dump(stock_names, f, ensure_ascii=False)
                return stock_names
        except:
            pass
        
        # 从本地缓存读取
        if names_file.exists():
            import json
            with open(names_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 使用默认名称
        return {code: f"股票{code}" for code in stock_data.keys()}
    
    def init_data(self, max_stocks=None):
        """首次全量抓取"""
        print("=" * 60)
        print("🚀 首次全量数据抓取")
        print("=" * 60)
        self.fetcher.init_full_data(max_stocks=max_stocks)
        print("\n✓ 数据初始化完成")

    def _get_latest_trading_date(self):
        """从本地股票数据推断最近的交易日，不依赖外部 API"""
        import pandas as pd
        try:
            stock_codes = self.csv_manager.list_all_stocks()
            if not stock_codes:
                return None
            # 取第一只股票（000001 平安银行）的最新日期
            df = self.csv_manager.read_stock(stock_codes[0])
            if df is None or df.empty:
                return None
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            return df['date'].max().date()
        except Exception:
            return None

    def _smart_update(self, max_stocks=None, check_latest=True, force_update=False):
        """智能更新：3 点前不更新，检查每只股票是否有当天数据"""
        from datetime import datetime
        import pandas as pd

        today = datetime.now().date()
        today_str = today.strftime('%Y-%m-%d')
        current_time = datetime.now().time()
        market_close_time = datetime.strptime("15:00", "%H:%M").time()

        # 3 点前：不更新，使用旧数据
        if current_time < market_close_time:
            print("\n⏰ 当前时间尚未收盘 (15:00)")
            print("  使用本地已有数据，跳过网络更新")
            return

        # 检查缓存：如果今天已更新过，直接跳过
        update_cache_file = Path(self.data_dir) / '.update_cache.json'
        update_cache = {}
        if update_cache_file.exists():
            try:
                import json
                with open(update_cache_file, 'r', encoding='utf-8') as f:
                    update_cache = json.load(f)
            except:
                update_cache = {}

        cache_date = update_cache.get('last_update_date')
        if cache_date == today_str and not max_stocks and not force_update:
            # 交叉验证：抽样检查多只股票，确保大多数都有今天的数据
            import random
            all_codes = self.csv_manager.list_all_stocks()
            sample = random.sample(all_codes, min(50, len(all_codes)))
            has_today = sum(
                1 for code in sample
                if (df := self.csv_manager.read_stock(code)) is not None
                and not df.empty
                and pd.to_datetime(df.iloc[0]['date']).date() == today
            )
            if has_today >= len(sample) * 0.8:
                print(f"✓ 数据已于 {cache_date} 收盘后更新过，无需重复更新")
                return
            else:
                print(f"⚠️ 缓存记录今天已更新，但抽样 {len(sample)} 只中仅 {has_today} 只有今天数据，重新执行更新")
                update_cache.pop('last_update_date', None)
                import json
                with open(update_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(update_cache, f)

        # 检查每只股票是否有当天数据
        if check_latest:
            print("\n🔍 检查数据更新状态...")
            import random
            stock_codes = self.csv_manager.list_all_stocks()
            if max_stocks:
                stock_codes = stock_codes[:max_stocks]

            total = len(stock_codes)
            check_limit = min(100, total)  # 抽样检查 100 只
            # 随机抽样，避免只抽到 000xxx 沪市股票导致误判
            sample_codes = random.sample(stock_codes, check_limit) if total > check_limit else stock_codes
            has_today = 0
            no_today = 0

            for code in sample_codes:
                df = self.csv_manager.read_stock(code)
                if not df.empty:
                    latest_date = pd.to_datetime(df.iloc[0]['date']).date()
                    if latest_date == today:
                        has_today += 1
                    else:
                        no_today += 1

            # 如果 100% 股票都有今天数据，跳过更新（force_update 时除外）
            if check_limit > 0 and has_today == check_limit and not force_update:
                print(f"  ✓ 已检查 {check_limit} 只股票，全部已有今天数据")
                print("  数据已是最新，跳过网络更新")
                return
            elif force_update:
                print(f"  --force 参数已设置，强制更新数据")
            else:
                print(f"  已检查 {check_limit} 只，{has_today} 只有今天数据，{no_today} 只需要更新")

        # 执行更新
        print("\n🔄 执行数据更新...")
        self.fetcher.daily_update(max_stocks=max_stocks, force_update=force_update)
        print("\n✓ 数据更新完成")

    def update_data(self, max_stocks=None, force_update=False):
        """每日增量更新"""
        print("=" * 60)
        print("🔄 每日增量更新")
        print("=" * 60)
        self.fetcher.daily_update(max_stocks=max_stocks, force_update=force_update)
        print("\n✓ 数据更新完成")

    def select_stocks(self, category='all', max_stocks=None, return_data=False, M_days=None, pick_date=None, skip_kdj=False):
        """执行选股
        :param category: 股票分类筛选，'all'表示全部，其他值按分类筛选
        :param max_stocks: 限制处理的股票数量（用于快速测试）
        :param return_data: 是否返回股票数据字典（用于K线图生成）
        :param M_days: 碗口反弹策略的回溯天数 M，None 则使用配置文件值
        :param skip_kdj: 是否跳过KDJ过滤（B1Plus模式）
        :return: (results, stock_names) 或 (results, stock_names, stock_data_dict, turnover_dict)
                 turnover_dict: {date_str: {code: amount}} 从内存数据顺手提取，避免二次读盘
        """
        print("=" * 60)
        print("🎯 执行选股策略")
        if max_stocks:
            print(f"   快速测试模式：只处理前 {max_stocks} 只股票")
        print("=" * 60)
        
        # 加载策略
        print("\n加载策略...")
        self.registry.auto_register_from_directory("strategy")
        
        if not self.registry.list_strategies():
            print("✗ 没有找到可用策略")
            return {}, {}
        
        print(f"已加载 {len(self.registry.list_strategies())} 个策略")
        # 如果启用了 skip_kdj，设置到碗口反弹策略
        if skip_kdj:
            for strategy_name, strategy_obj in self.registry.strategies.items():
                if hasattr(strategy_obj, 'skip_kdj'):
                    strategy_obj.skip_kdj = True
                    print(f"  ⚠️  {strategy_name}: 跳过KDJ过滤（B1Plus模式）")
        # 如果传入了 M_days，覆盖所有策略的 M 参数
        if M_days is not None:
            print(f"⚠️  使用命令行覆盖策略参数 M = {M_days}")
            for strategy_name, strategy_obj in self.registry.strategies.items():
                if 'M' in strategy_obj.params:
                    strategy_obj.params['M'] = M_days
        
        
        # 输出当前策略参数
        print("\n当前策略参数:")
        for strategy_name, strategy_obj in self.registry.strategies.items():
            print(f"\n  🎯 {strategy_name}:")
            for param_name, param_value in strategy_obj.params.items():
                # 对特定参数添加说明
                note = ""
                if param_name == 'N':
                    note = " (成交量倍数)"
                elif param_name == 'M':
                    if M_days:
                        note = f" (回溯天数={M_days}, 命令行覆盖)"
                    else:
                        note = " (回溯天数)"
                elif param_name == 'CAP':
                    note = f" ({param_value/1e8:.0f}亿市值门槛)"
                elif param_name == 'J_VAL':
                    note = " (J值上限)"
                elif param_name in ['M1', 'M2', 'M3', 'M4']:
                    note = " (MA周期)"
                print(f"      {param_name}: {param_value}{note}")
        
        # 加载股票数据（一次性读取缓存，避免每个策略重复读取）
        print("\n执行选股（已优化IO复用，速度提升2~3倍）...")
        stock_codes = self.csv_manager.list_all_stocks()
        
        if not stock_codes:
            print("✗ 没有股票数据，请先执行 init 或 update")
            if return_data:
                return {}, {}, {}
            return {}, {}
        
        print(f"共 {len(stock_codes)} 只股票")
        
        # 先获取股票名称
        stock_names = self._load_stock_names({})
        
        # 限制处理数量
        process_codes = stock_codes[:max_stocks] if max_stocks else stock_codes
        
        # 第一步：一次性读取所有需要处理的股票数据（缓存复用）
        print("\n[1/2] 加载股票数据...")
        import gc
        from datetime import datetime
        import pandas as pd

        # 如果指定了选股日期，解析为 datetime 对象
        target_date = None
        if pick_date:
            if isinstance(pick_date, str):
                target_date = datetime.strptime(pick_date, '%Y-%m-%d')
            else:
                target_date = pick_date
            print(f"  选股日期：{target_date.strftime('%Y-%m-%d')} (非交易日则使用前一交易日)")

        stock_data_cache = {}
        valid_codes = []
        invalid_count = 0
        _turnover_dict = {}  # {date_str: {code: amount}}，顺手提取，避免后续二次读盘
        
        for i, code in enumerate(process_codes, 1):
            df = self.csv_manager.read_stock(code)
            name = stock_names.get(code, '未知')

            # 如果指定了日期，截断到该日期（或前一交易日）
            if target_date and df is not None and not df.empty:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                # 找到小于等于目标日期的最大日期（最后一个交易日）
                df_before = df[df['date'] <= target_date]
                if not df_before.empty:
                    df = df_before.reset_index(drop=True)
                else:
                    # 如果所有数据都晚于目标日期，跳过该股票
                    continue

            # 顺手提取最新两天的成交额（在过滤前提取，确保全市场覆盖）
            if df is not None and not df.empty:
                for row_idx in range(min(2, len(df))):
                    row = df.iloc[row_idx]
                    d = row['date']
                    row_date = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
                    amt = row.get('amount', None)
                    if pd.notna(amt) and amt > 0:
                        _turnover_dict.setdefault(row_date, {})[code] = float(amt)
                    else:
                        vol = row.get('volume', 0)
                        price = row.get('close', 0)
                        if pd.notna(vol) and pd.notna(price) and vol > 0 and price > 0:
                            _turnover_dict.setdefault(row_date, {})[code] = float(vol * price * 100)

            # 过滤无效股票（使用统一过滤规则，与选股逻辑完全一致）
            from utils.stock_filter import is_valid_stock
            if not is_valid_stock(name, df):
                invalid_count += 1
                continue

            # 缓存有效股票数据
            stock_data_cache[code] = df
            valid_codes.append(code)

            # 每500只显示进度
            if i % 500 == 0 or i == len(process_codes):
                gc.collect()
                print(f"  进度: [{i}/{len(process_codes)}] 有效 {len(valid_codes)} 只，过滤 {invalid_count} 只...")
        
        print(f"\n✓ 数据加载完成：共 {len(valid_codes)} 只有效股票")
        
        # 第二步：所有策略复用缓存数据，不需要重复读文件
        print("\n[2/2] 执行选股策略（已开启并行计算优化，速度提升3~4倍）...")
        results = {}
        indicators_dict = {}  # 只保存入选股票的数据
        category_count = {'bowl_center': 0, 'near_duokong': 0, 'near_short_trend': 0}
        
        # 并行处理单只股票的选股逻辑
        def process_stock(args):
            code, df, strategy, name, category = args
            try:
                # 计算指标
                df_with_indicators = strategy.calculate_indicators(df)
                # 选股
                signal_list = strategy.select_stocks(df_with_indicators, name)
                return code, name, signal_list, df_with_indicators if signal_list else None
            except Exception as e:
                print(f"  处理 {code} 异常: {e}")
                return code, name, [], None
        
        for strategy_name, strategy in self.registry.strategies.items():
            print(f"\n执行策略: {strategy_name}")
            signals = []
            
            # 准备并行参数
            process_args = []
            for code in valid_codes:
                df = stock_data_cache[code]
                name = stock_names.get(code, '未知')
                # 传递策略类和参数，避免序列化实例
                process_args.append((code, df, strategy.__class__, strategy.params, name, category))
            
            # 多进程并行处理
            import multiprocessing as mp
            # 使用CPU核心数-1的进程数，避免系统满载
            num_workers = max(1, mp.cpu_count() - 1)
            print(f"  启动 {num_workers} 个进程并行计算...")
            
            with mp.Pool(num_workers) as pool:
                results_iter = pool.imap(_process_stock_parallel, process_args, chunksize=20)
                
                for i, (code, name, signal_list, df_with_indicators) in enumerate(results_iter, 1):
                    if signal_list:
                        for s in signal_list:
                            cat = s.get('category', 'unknown')
                            category_count[cat] = category_count.get(cat, 0) + 1
                            
                            if category == 'all' or cat == category:
                                signals.append({
                                    'code': code,
                                    'name': name,
                                    'signals': [s]
                                })
                                # 只保存入选股票的数据
                                if return_data:
                                    indicators_dict[code] = df_with_indicators
                    
                    # 每200只显示进度
                    if i % 200 == 0 or i == len(valid_codes):
                        gc.collect()
                        print(f"  进度: [{i}/{len(valid_codes)}] 选出 {len(signals)} 只...")
            
            results[strategy_name] = signals
            print(f"  ✓ 选股完成: 共 {len(signals)} 只")
        

        # 按20日量价趋势一致度降序排列（一致度越高越靠前）
        for strategy_name in results:
            results[strategy_name].sort(
                key=lambda x: x['signals'][0].get('price_volume_alignment', 0) if x.get('signals') else 0,
                reverse=True
            )
        # 显示结果汇总
        print("\n" + "=" * 60)
        print("📊 选股结果汇总（已按20日量价一致度排序）")
        print("=" * 60)
        
        for strategy_name, signals in results.items():
            print(f"\n{strategy_name}: {len(signals)} 只")
            for signal in signals:
                code = signal['code']
                name = signal.get('name', stock_names.get(code, '未知'))
                for s in signal['signals']:
                    cat_emoji = {'bowl_center': '🥣', 'near_duokong': '📊', 'near_short_trend': '📈'}.get(s.get('category'), '❓')
                    yang_tag = " | 🔺阳量>阴量" if s.get('yang_gt_yin') else ""
                    pv_align = s.get('price_volume_alignment', 0)
                    pv_tag = f" | 量价一致性={pv_align}%" if pv_align > 0 else ""
                    print(f"  {cat_emoji} {code} {name}: 价格={s['close']}, J={s['J']}, 理由={s['reasons']}{yang_tag}{pv_tag}")
        
        # 显示分类统计
        print("\n" + "-" * 60)
        print("分类统计:")
        print(f"  🥣 回落碗中: {category_count.get('bowl_center', 0)} 只")
        print(f"  📊 靠近多空线: {category_count.get('near_duokong', 0)} 只")
        print(f"  📈 靠近短期趋势线: {category_count.get('near_short_trend', 0)} 只")
        print("-" * 60)
        
        # 如果需要返回数据字典（用于K线图生成）
        if return_data:
            # 返回计算了指标的数据（包含趋势线）+ 顺手提取的成交额
            return results, stock_names, indicators_dict, _turnover_dict
        
        return results, stock_names

    def run_consolidation(self, max_stocks=None, notify=False, chart=False, pick_date=None, force_update=False, after_high_only=False):
        """横盘整理选股模式：高点后整理，多空线上升，价格在线上
        :param after_high_only: True=条件仅检查高点之后（宽松），False=检查整个60天窗口（严格）
        """
        from datetime import datetime
        import json
        from pathlib import Path

        print("=" * 60)
        print("📊 执行横盘整理选股模式")
        if max_stocks:
            print(f"   快速测试模式：只处理前 {max_stocks} 只股票")
        print("=" * 60)

        # 1. 更新数据
        self._smart_update(max_stocks=max_stocks, force_update=force_update)

        # 2. 加载股票数据
        all_stocks = self.csv_manager.list_all_stocks()
        stock_names = self._load_stock_names({})

        if max_stocks:
            process_codes = all_stocks[:max_stocks]
        else:
            process_codes = all_stocks

        from utils.stock_filter import is_valid_stock
        import gc

        valid_codes = []
        stock_data_cache = {}
        invalid_count = 0

        print(f"\n加载 {len(process_codes)} 只股票数据...")
        for i, code in enumerate(process_codes, 1):
            name = stock_names.get(code, '未知')
            df = self.csv_manager.read_stock(code)
            if df is None or df.empty:
                invalid_count += 1
                continue

            # 日期截断（回测支持）
            if pick_date:
                if isinstance(pick_date, str):
                    pick_date_dt = datetime.strptime(pick_date, '%Y-%m-%d')
                else:
                    pick_date_dt = pick_date
                df = df[df['date'] <= pick_date_dt]

            if not is_valid_stock(name, df):
                invalid_count += 1
                continue

            stock_data_cache[code] = df
            valid_codes.append(code)

            if i % 500 == 0 or i == len(process_codes):
                gc.collect()
                print(f"  进度: [{i}/{len(process_codes)}] 有效 {len(valid_codes)} 只，过滤 {invalid_count} 只...")

        print(f"\n✓ 数据加载完成：共 {len(valid_codes)} 只有效股票")

        # 3. 并行执行横盘整理选股
        print("\n执行横盘整理选股...")
        self.registry.auto_register_from_directory("strategy")
        strategy = self.registry.strategies.get('BowlReboundStrategy')
        if not strategy:
            print("✗ 未找到BowlReboundStrategy")
            return

        process_args = [
            (code, stock_data_cache[code], strategy.__class__, strategy.params, stock_names.get(code, '未知'), after_high_only)
            for code in valid_codes
        ]

        import multiprocessing as mp
        num_workers = max(1, mp.cpu_count() - 1)
        print(f"  启动 {num_workers} 个进程并行计算...")

        results = {}
        signals = []
        indicators_dict = {}

        with mp.Pool(num_workers) as pool:
            results_iter = pool.imap(_process_consolidation_parallel, process_args, chunksize=20)

            for i, (code, name, signal_list, df_with_indicators) in enumerate(results_iter, 1):
                if signal_list:
                    # 补算阳量>阴量标记
                    for s in signal_list:
                        recent_10 = stock_data_cache[code].head(10)
                        yang = recent_10.loc[recent_10['close'] > recent_10['open'], 'amount'].sum()
                        yin = recent_10.loc[recent_10['close'] < recent_10['open'], 'amount'].sum()
                        s['yang_gt_yin'] = yang > yin

                    signals.append({
                        'code': code,
                        'name': name,
                        'signals': signal_list
                    })
                    if df_with_indicators is not None:
                        indicators_dict[code] = df_with_indicators

                if i % 200 == 0 or i == len(valid_codes):
                    gc.collect()
                    print(f"  进度: [{i}/{len(valid_codes)}] 选出 {len(signals)} 只...")

        results['BowlReboundStrategy'] = signals
        print(f"\n✓ 横盘整理选股完成: 共 {len(signals)} 只")

        # 4. 输出结果
        print("\n" + "=" * 60)
        print("📊 横盘整理选股结果")
        print("=" * 60)
        for signal in signals:
            code = signal['code']
            name = signal.get('name', stock_names.get(code, '未知'))
            for s in signal['signals']:
                high_price = s.get('high_price', '-')
                days = s.get('reasons', [''])[0]
                yang_tag = " | 🔺阳量>阴量" if s.get('yang_gt_yin') else ""
                print(f"  📊 {code} {name}: 价格={s['close']}, 高点={high_price}, {days}{yang_tag}")
        print("-" * 60)

        # 5. 保存结果到文件
        import json
        from pathlib import Path
        results_dir = Path(self.data_dir) / 'results'
        results_dir.mkdir(exist_ok=True)
        save_date = pick_date or datetime.now().strftime('%Y-%m-%d')
        mode_tag = 'consolidation_宽松' if after_high_only else 'consolidation_严格'
        output_file = results_dir / f'{save_date}_results_{mode_tag}.json'

        save_data = {
            'date': save_date,
            'mode': mode_tag,
            'total': len(signals),
            'stocks': []
        }
        for signal in signals:
            code = signal['code']
            name = signal.get('name', stock_names.get(code, '未知'))
            for s in signal['signals']:
                save_data['stocks'].append({
                    'code': code,
                    'name': name,
                    'close': float(s.get('close', 0)),
                    'high_price': float(s.get('high_price', 0)),
                    'reasons': s.get('reasons', []),
                    'yang_gt_yin': bool(s.get('yang_gt_yin', False)),
                    'market_cap': float(s.get('market_cap', 0)),
                })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {output_file}")

        return results, stock_names

    def run_full(self, category='all', max_stocks=None, notify=False, chart=False, M_days=None, pick_date=None, force_update=False, filter_zge_risk=False, brick_filter=False):
        """完整流程：更新 + 选股 + 通知（可选K线图）
        :param max_stocks: 限制处理的股票数量（用于快速测试）
        :param notify: 是否发送飞书通知（默认不发送）
        :param chart: 是否生成K线图并发送（默认不生成，需配合notify使用）
        :param filter_zge_risk: 是否剔除触发Z哥风控规则的股票
        :param brick_filter: 是否启用砖型图三层过滤
        """
        from datetime import datetime
        import json
        from pathlib import Path

        print("=" * 60)
        print("🚀 执行完整流程")
        if max_stocks:
            print(f"   快速测试模式：只处理前 {max_stocks} 只股票")
        print("=" * 60)

        # 1. 更新数据（内置逻辑：3点前不更新，检查每只股票是否有当天数据）
        self._smart_update(max_stocks=max_stocks, force_update=force_update)

        # 2. 选股（返回数据和结果）
        results, stock_names, stock_data_dict, _ = self.select_stocks(category=category, max_stocks=max_stocks, return_data=True, M_days=M_days, pick_date=pick_date)
        
        # ============== 强制Z哥交易规则过滤 ==============
        print("\n[Z哥过滤] 开始执行Z哥交易规则筛选...")
        from strategy.zge_filter import ZgeFilter
        zge_filter = ZgeFilter(self.data_dir)
        filtered_total = 0
        original_total = 0
        
        # 处理每个策略的信号
        for strategy_name, signals in results.items():
            original_total += len(signals)
            for signal in signals:
                code = signal['code']
                if code not in stock_data_dict or stock_data_dict[code].empty:
                    # 没有数据，标记为剔除
                    signal['zge_passed'] = False
                    signal['zge_reason'] = "无有效日线数据"
                    continue
                df_daily = stock_data_dict[code]
                
                # 执行Z哥过滤
                _, zge_info = zge_filter.filter_stock(code, df_daily)
                # filter_stock总是返回True，使用has_risk字段判断是否通过
                signal.update(zge_info)
                signal['zge_passed'] = not zge_info['has_risk']
                if not signal['zge_passed']:
                    signal['zge_reason'] = ",".join(zge_info.get('zge_triggered_rules', []))
            
            # 所有股票全部保留
            results[strategy_name] = signals
            filtered_total = len(signals)
        
        print(f"✓ Z哥规则标注完成: 共 {filtered_total} 只股票")
        # ============== Z哥过滤结束 ==============
        
        # 如果开启了过滤Z哥风控，移除未通过的股票
        if filter_zge_risk:
            print(f"\n[Z哥过滤] 开启剔除模式，正在移除触发风控的股票...")
            new_total = 0
            removed_count = 0
            for strategy_name, signals in results.items():
                # 保留通过Z哥风控的股票
                passed_signals = [s for s in signals if s.get('zge_passed', False)]
                removed = len(signals) - len(passed_signals)
                removed_count += removed
                results[strategy_name] = passed_signals
                new_total += len(passed_signals)
            print(f"✓ 剔除完成: 原有 {filtered_total} 只，保留 {new_total} 只，移除 {removed_count} 只触发风控的股票")
            filtered_total = new_total

        # ============== 砖型图三层过滤 ==============
        if brick_filter:
            print("\n[砖型图过滤] 开始执行砖型图三层筛选...")
            from strategy.brick_filter import BrickFilter
            bf = BrickFilter()
            brick_pass = 0
            brick_fail = 0
            for strategy_name, signals in results.items():
                passed = []
                for signal in signals:
                    code = signal['code']
                    if code in stock_data_dict and not stock_data_dict[code].empty:
                        ok, bi = bf.filter_stock(stock_data_dict[code])
                        signal['brick_passed'] = ok
                        if not ok:
                            brick_fail += 1
                            continue
                        brick_pass += 1
                        passed.append(signal)
                    else:
                        brick_fail += 1
                results[strategy_name] = passed
            print(f"✓ 砖型图过滤完成: 通过 {brick_pass} 只，剔除 {brick_fail} 只")
        # ============== 砖型图过滤结束 ==============

        # 4. 最终结果汇总（过滤后）
        has_filter = filter_zge_risk or brick_filter
        if has_filter:
            total_final = sum(len(s) for s in results.values())
            if total_final > 0:
                print("\n" + "=" * 60)
                print(f"📊 最终推荐榜（共 {total_final} 只）")
                if filter_zge_risk:
                    print("   已过滤: Z哥风控")
                if brick_filter:
                    print("   已过滤: 砖型图")
                print("=" * 60)

                for strategy_name, signals in results.items():
                    if not signals:
                        continue
                    print(f"\n{strategy_name}: {len(signals)} 只")
                    for signal in signals:
                        code = signal['code']
                        name = signal.get('name', stock_names.get(code, '未知'))
                        for s in signal['signals']:
                            cat_emoji = {'bowl_center': '🥣', 'near_duokong': '📊', 'near_short_trend': '📈'}.get(s.get('category'), '❓')
                            yang_tag = " | 🔺阳量>阴量" if s.get('yang_gt_yin') else ""
                            pv_align = s.get('price_volume_alignment', 0)
                            pv_tag = f" | 量价一致性={pv_align}%" if pv_align > 0 else ""
                            brick_tag = " | 🧱砖型图" if s.get('brick_passed') else ""
                            zge_tag = ""
                            if signal.get('zge_passed') is not None:
                                zge_tag = " | ✅Z哥通过" if signal.get('zge_passed') else " | ⚠️Z哥风控"
                            print(f"  {cat_emoji} {code} {name}: 价格={s['close']}, J={s['J']}, 理由={s['reasons']}{yang_tag}{pv_tag}{brick_tag}{zge_tag}")
            else:
                print("\n⚠️ 过滤后无股票满足条件")

        # 3. 发送通知（可选K线图）
        if results and notify:
            if chart:
                # 带K线图发送
                self.notifier.send_stock_selection_with_charts(
                    results,
                    stock_names,
                    category_filter=category,
                    stock_data_dict=stock_data_dict,
                    params=self.registry.strategies.get('BowlReboundStrategy', {}).params if self.registry.strategies else {},
                    send_text_first=True
                )
            else:
                # 只发送文本
                self.notifier.send_stock_selection(
                    results,
                    stock_names,
                    category_filter=category
                )
        elif results:
            print("\n💡 提示: 使用 --notify 发送飞书通知，--chart 附带K线图")

        return results
    
    def select_with_b1_match(self, category='all', max_stocks=None, min_similarity=None, M_days=None, pick_date=None, use_cache=False, force_update=False, filter_zge_risk=False, skip_kdj=False, cases_list=None, weights=None, cache_file=None, label='B1', rank_by_vol=False, brick_filter=False):
        """
        执行选股 + 完美图形匹配排序（支持 B1 / B1Plus）

        Args:
            category: 股票分类筛选，'all'表示全部
            max_stocks: 限制处理的股票数量
            min_similarity: 最小相似度阈值，低于此值不显示
            skip_kdj: 是否跳过KDJ过滤（B1Plus模式）
            cases_list: 案例配置列表（B1Plus模式时传入）
            weights: 相似度权重（B1Plus模式时传入）
            cache_file: 缓存文件路径（B1Plus模式时传入）
            label: 流程标签（B1 / B1Plus）

        Returns:
            dict: 包含选股结果和匹配结果
        """
        # 从配置读取默认值
        from strategy.pattern_config import MIN_SIMILARITY_SCORE
        if min_similarity is None:
            min_similarity = MIN_SIMILARITY_SCORE
        
        print("=" * 60)
        print(f"🎯 执行选股 + {label}完美图形匹配")
        if max_stocks:
            print(f"   快速测试模式：只处理前 {max_stocks} 只股票")
        print(f"   相似度阈值: {min_similarity}%")
        if skip_kdj:
            print(f"   模式: {label}（跳过KDJ过滤）")
        print("=" * 60)
        
        # 1. 先执行原有选股逻辑
        print("\n[1/3] 执行策略选股...")
        results, stock_names, stock_data_dict, inline_turnover = self.select_stocks(
            category=category,
            max_stocks=max_stocks,
            return_data=True,
            M_days=M_days,
            pick_date=pick_date,
            skip_kdj=skip_kdj
        )
        
        # 统计选股总数
        total_selected = sum(len(signals) for signals in results.values())
        if total_selected == 0:
            print("\n✗ 策略未选出任何股票，跳过匹配")
            return {'results': results, 'stock_names': stock_names, 'matched': []}
        
        print(f"\n✓ 策略选出 {total_selected} 只股票")
        
        # 2. 初始化完美图形库
        print(f"\n[2/3] 初始化{label}完美图形库...")
        try:
            from strategy.pattern_library import B1PatternLibrary
            from strategy.pattern_config import MIN_SIMILARITY_SCORE

            library = B1PatternLibrary(
                self.csv_manager,
                cases_list=cases_list,
                weights=weights,
                cache_file=cache_file,
            )
            
            if not library.cases:
                print("⚠️ 警告: 案例库为空，可能数据不足")
                return {'results': results, 'stock_names': stock_names, 'matched': []}
            
            print(f"✓ 案例库加载完成: {len(library.cases)} 个案例")
            
        except Exception as e:
            print(f"✗ 初始化案例库失败: {e}")
            import traceback
            traceback.print_exc()
            return {'results': results, 'stock_names': stock_names, 'matched': []}
        
        # 3. 对每只候选股进行匹配
        matched_results = []

        # 按成交量倍数排名模式：跳过B1图形匹配
        if rank_by_vol:
            print("\n[3/3] 跳过B1图形匹配，按成交量倍数排名...")
            for strategy_name, signals in results.items():
                for signal in signals:
                    code = signal['code']
                    name = signal.get('name', stock_names.get(code, '未知'))
                    s = signal['signals'][0] if signal.get('signals') else {}
                    matched_results.append({
                        'stock_code': code,
                        'stock_name': name,
                        'strategy': strategy_name,
                        'category': s.get('category', 'unknown'),
                        'close': s.get('close', '-'),
                        'J': s.get('J', '-'),
                        'volume_ratio': float(s.get('volume_ratio', 1.0)),
                        'key_vol_ratio': float(s.get('key_vol_ratio', 1.0)),
                        'similarity_score': float(s.get('key_vol_ratio', 1.0)) * 10,
                        'matched_case': '',
                        'matched_date': '',
                        'matched_code': '',
                        'breakdown': {},
                        'tags': [],
                        'description': f"突破量比{s.get('key_vol_ratio', 1.0):.1f}倍",
                        'all_matches': [],
                        'industry': '未知',
                        'industry_heat': 'N/A',
                        'industry_heat_change': None,
                    })
            matched_results.sort(key=lambda x: x['key_vol_ratio'], reverse=True)
            print(f"✓ 按成交量倍数排名完成: {len(matched_results)} 只")
        else:
            print(f"\n[3/3] 执行{label}完美图形匹配...")

        # 初始化行业数据获取器（用于获取行业和行业热度）
        import pandas as pd
        from utils.industry_fetcher import IndustryFetcher, IndustryHeatCalculator, TurnoverCache
        industry_fetcher = IndustryFetcher(Path(self.data_dir) / 'industry_cache')
        industry_fetcher.load_industry_mapping(force_refresh=not use_cache)

        # 获取选股日期（优先使用传入的 pick_date 参数）
        pick_date_str = None
        if pick_date:
            from datetime import datetime
            if isinstance(pick_date, datetime):
                pick_date_str = pick_date.strftime('%Y-%m-%d')
            elif isinstance(pick_date, str):
                pick_date_str = pick_date
            print(f"  使用传入的选股日期：{pick_date_str}")
        else:
            # 从信号中获取日期
            from datetime import datetime
            for strategy_name, signals in results.items():
                if signals:
                    for sig in signals:
                        s = sig['signals'][0] if sig.get('signals') else {}
                        # 尝试多个可能的日期字段
                        signal_date = s.get('actual_date') or s.get('date')
                        if signal_date:
                            if isinstance(signal_date, datetime):
                                pick_date_str = signal_date.strftime('%Y-%m-%d')
                            elif isinstance(signal_date, str):
                                pick_date_str = signal_date
                            break
                    if pick_date_str:
                        break

        # 确定用于行业热度计算的日期：优先从最新股票数据获取，其次是传入的pick_date，最后是信号中的日期
        heat_calc_date = None

        # 1. 优先从全市场第一只股票获取最新交易日（最准确，避免选股过滤导致取到旧数据）
        all_codes = self.csv_manager.list_all_stocks()
        if all_codes:
            # 取全市场第一只股票（000001 平安银行），它的数据更新最及时
            first_stock_code = all_codes[0]
            df = self.csv_manager.read_stock(first_stock_code)
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                latest_date = df_copy['date'].max()
                heat_calc_date = latest_date.strftime('%Y-%m-%d')
                print(f"  ℹ️  使用最新交易日计算行业热度：{heat_calc_date}")

        # 2. 如果股票数据中没有日期，使用传入的pick_date
        if not heat_calc_date and pick_date_str:
            heat_calc_date = pick_date_str
            print(f"  ℹ️  使用传入的选股日期：{heat_calc_date}")

        # 3. 最后尝试从信号中获取日期
        if not heat_calc_date:
            # 从信号中获取日期
            from datetime import datetime
            for strategy_name, signals in results.items():
                if signals:
                    for sig in signals:
                        s = sig['signals'][0] if sig.get('signals') else {}
                        # 尝试多个可能的日期字段
                        signal_date = s.get('actual_date') or s.get('date')
                        if signal_date:
                            if isinstance(signal_date, datetime):
                                heat_calc_date = signal_date.strftime('%Y-%m-%d')
                            elif isinstance(signal_date, str):
                                heat_calc_date = signal_date
                            break
                    if heat_calc_date:
                        break

        if not heat_calc_date:
            print("  ℹ️  未指定选股日期，行业热度将显示 N/A")
        else:
            # 只在 pick_date_str 为空时才用 heat_calc_date 补充（不覆盖用户传入的日期）
            if not pick_date_str:
                pick_date_str = heat_calc_date

        # 加载成交额缓存（用于行业热度计算）
        turnover_cache_manager = TurnoverCache()
        if turnover_cache_manager.load():
            turnover_cache = turnover_cache_manager.get_cache()
        else:
            turnover_cache = {}

        # 预先获取全市场股票代码（用于行业热度计算）
        all_market_stocks = self.csv_manager.list_all_stocks()
        print(f"  全市场股票数（用于行业热度计算）：{len(all_market_stocks)}")

        # 用 select_stocks 顺手提取的内存成交额填充缓存，完全跳过二次读盘
        if heat_calc_date:
            MIN_STOCKS = 500
            # 将 inline_turnover 合并进持久化缓存（只补充缺失日期）
            for d, d_data in inline_turnover.items():
                if d not in turnover_cache_manager.cache or len(turnover_cache_manager.cache[d]) < len(d_data):
                    turnover_cache_manager.cache[d] = d_data

            existing_today = turnover_cache_manager.cache.get(heat_calc_date, {})
            if len(existing_today) >= MIN_STOCKS:
                print(f"  ✓ 成交额缓存命中 {heat_calc_date}：{len(existing_today)}只股票")
            else:
                # 内存数据不足时才回退到磁盘读取
                print(f"  [行业热度] 内存数据不足（{len(existing_today)}只），从 CSV 补充...")
                date_turnover = self._fetch_date_turnover_from_csv(heat_calc_date, all_market_stocks)
                if date_turnover and len(date_turnover) >= MIN_STOCKS:
                    turnover_cache_manager.cache[heat_calc_date] = date_turnover
                    print(f"  ✓ 已补充 {heat_calc_date} 的成交额数据：{len(date_turnover)}只股票")
                else:
                    # 找缓存中数据最多的日期作为替代
                    best_date = max(
                        ((d, len(v)) for d, v in turnover_cache_manager.cache.items() if len(v) >= MIN_STOCKS),
                        key=lambda x: x[1], default=(None, 0)
                    )[0]
                    if best_date:
                        heat_calc_date = best_date
                        print(f"  ✓ 使用缓存中 {heat_calc_date} 的数据替代")
                    else:
                        print(f"  ✗ 缓存中没有足够的数据，无法计算行业热度")
                        heat_calc_date = None

            # 统一保存
            turnover_cache_manager.save()
            turnover_cache = turnover_cache_manager.get_cache()
            print(f"  ✓ 行业热度使用日期：{heat_calc_date} (缓存中共有 {len(turnover_cache)} 个交易日)")
        else:
            print("  ℹ️  未指定选股日期，行业热度将显示 N/A")

        # 行业热度计算器（循环外创建一次，避免重复实例化）
        heat_calc = IndustryHeatCalculator(industry_fetcher) if heat_calc_date else None
        # 行业热度 memo 缓存，key=(industry, date)，避免同行业多只股票重复计算
        industry_heat_memo = {}

        if not rank_by_vol:
            for strategy_name, signals in results.items():
                for signal in signals:
                    code = signal['code']
                    name = signal.get('name', stock_names.get(code, '未知'))

                    # 获取该股票的完整数据
                    if code not in stock_data_dict:
                        continue

                    df = stock_data_dict[code]
                    if df.empty:
                        continue

                    try:
                        # 匹配最佳案例（每个案例使用各自的lookback_days）
                        match_result = library.find_best_match(code, df)

                        if match_result.get('best_match'):
                            best = match_result['best_match']
                            score = best.get('similarity_score', 0)

                            # 只保留超过阈值的股票
                            if score >= min_similarity:
                                # 获取第一个信号的信息
                                s = signal['signals'][0] if signal.get('signals') else {}

                                # 获取行业信息（如果缓存中没有，尝试从 API 获取）
                                industry = industry_fetcher.get_industry_for_stock(code, refresh_if_missing=True)

                                # 获取行业热度（使用 memo 缓存，同行业只计算一次）
                                industry_heat = None
                                industry_heat_change = None
                                if industry and heat_calc and heat_calc_date:
                                    memo_key = (industry, heat_calc_date)
                                    if memo_key not in industry_heat_memo:
                                        industry_heat_memo[memo_key] = heat_calc.calculate_industry_heat_fast(
                                            industry, heat_calc_date, turnover_cache, all_market_stocks)
                                    industry_heat, industry_heat_change = industry_heat_memo[memo_key]

                                matched_results.append({
                                    'stock_code': code,
                                    'stock_name': name,
                                    'strategy': strategy_name,
                                    'category': s.get('category', 'unknown'),
                                    'close': s.get('close', '-'),
                                    'J': s.get('J', '-'),
                                    'similarity_score': score,
                                    'matched_case': best.get('case_name', ''),
                                    'matched_date': best.get('case_date', ''),
                                    'matched_code': best.get('case_code', ''),
                                    'breakdown': best.get('breakdown', {}),
                                    'tags': best.get('tags', []),
                                    'description': best.get('description', ''),
                                    'all_matches': best.get('all_matches', []),
                                    'industry': industry if industry else '未知',
                                    'industry_heat': round(industry_heat, 1) if industry_heat is not None else 'N/A',
                                    'industry_heat_change': round(industry_heat_change, 1) if industry_heat_change is not None else None,
                                })

                    except Exception as e:
                        print(f"  ⚠️ 匹配 {code} 失败: {e}")
                        continue

            # 按相似度排序
            matched_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # ============== Z哥交易规则处理 ==============
        print("\n[Z哥过滤] 开始执行Z哥交易规则标注...")
        from strategy.zge_filter import ZgeFilter
        zge_filter = ZgeFilter(self.data_dir)
        processed_results = []
        original_matched_count = len(matched_results)
        
        for result in matched_results:
            code = result['stock_code']
            if code not in stock_data_dict:
                continue
            df_daily = stock_data_dict[code]
            
            # 构造行业热度数据
            industry_heat_data = {}
            heat_val = result.get('industry_heat')
            if heat_val not in ('N/A', None):
                # heat_score 是分位数(0-100)，转换为 heat_rank_pct（0=最热，1=最冷）
                industry_heat_data[code] = {
                    'heat_rank_pct': (100 - heat_val) / 100
                }
            
            # 执行Z哥过滤
            _, zge_info = zge_filter.filter_stock(code, df_daily, industry_heat_data)
            # 合并Z哥规则信息到结果
            result.update(zge_info)
            result['zge_passed'] = not zge_info['has_risk']

            # 根据filter_zge_risk参数决定是否过滤
            if not filter_zge_risk or result['zge_passed']:
                processed_results.append(result)
        
        # 替换为处理后的结果
        matched_results = processed_results
        if filter_zge_risk:
            removed_count = original_matched_count - len(matched_results)
            print(f"✓ Z哥规则过滤完成: 原有 {original_matched_count} 只，保留 {len(matched_results)} 只，移除 {removed_count} 只触发风控的股票")
        else:
            print(f"✓ Z哥规则标注完成: 共 {len(matched_results)} 只股票")
        # ============== Z哥处理结束 ==============

        # ============== 砖型图三层过滤 ==============
        if brick_filter:
            print("\n[砖型图过滤] 开始执行砖型图三层筛选...")
            from strategy.brick_filter import BrickFilter
            bf = BrickFilter()
            brick_before = len(matched_results)
            brick_passed = []
            for result in matched_results:
                code = result['stock_code']
                if code in stock_data_dict and not stock_data_dict[code].empty:
                    ok, bi = bf.filter_stock(stock_data_dict[code])
                    result['brick_passed'] = ok
                    if ok:
                        brick_passed.append(result)
                else:
                    result['brick_passed'] = False
            matched_results = brick_passed
            brick_removed = brick_before - len(matched_results)
            print(f"✓ 砖型图过滤完成: 通过 {len(matched_results)} 只，剔除 {brick_removed} 只")
        # ============== 砖型图过滤结束 ==============
        
        print(f"\n✓ 匹配完成: {len(matched_results)} 只股票超过阈值")
        
        # 显示Top N结果（使用配置）
        from strategy.pattern_config import TOP_N_RESULTS
        if matched_results:
            print("\n" + "=" * 60)
            mode_label = "成交量倍数排名" if rank_by_vol else f"B1完美图形匹配"
            filter_tags = []
            if brick_filter:
                filter_tags.append("🧱砖型图")
            if filter_zge_risk:
                filter_tags.append("Z哥风控")
            filter_label = f"（已过滤: {'+'.join(filter_tags)}）" if filter_tags else ""
            print(f"📊 Top {TOP_N_RESULTS} {mode_label}结果{filter_label}")
            print("=" * 60)
            for i, r in enumerate(matched_results[:TOP_N_RESULTS], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                close_price = r.get('close', '-')
                print(f"{emoji} {r['stock_code']} {r['stock_name']} 现价:{close_price}")
                heat_change = r.get('industry_heat_change')
                heat_change_str = f' | 较昨日：{heat_change:+.1f}%' if heat_change is not None else ''
                print(f"   行业：{r['industry']} | 行业热度：{r['industry_heat']}分{heat_change_str}")
                if rank_by_vol:
                    print(f"   突破量比: {r['key_vol_ratio']}倍")
                else:
                    print(f"   相似度: {r['similarity_score']}% | 匹配: {r['matched_case']}")
                    # 显示匹配原因/趋势描述
                    if r.get('description'):
                        print(f"   趋势：{r['description']}")
                    bd = r.get('breakdown', {})
                    print(f"   趋势:{bd.get('trend_structure', 0)}% "\
                          f"KDJ:{bd.get('kdj_state', 0)}% "\
                          f"量能:{bd.get('volume_pattern', 0)}% "\
                          f"形态:{bd.get('price_shape', 0)}%")
                # 显示Z哥规则标注
                if r.get('zge_triggered_rules'):
                    rules_str = "|".join(r['zge_triggered_rules'])
                    hints_str = "，".join(r['zge_hints'])
                    position_pct = int(r['zge_position_ratio'] * 100)
                    
                    # 按照规则设置emoji
                    if "触发Z哥风控规则" in hints_str:
                        zge_emoji = "⚠️"
                        position_pct = 0
                    elif r.get('zge_signal'):
                        zge_emoji = "✅"
                    else:
                        zge_emoji = "ℹ️"
                        position_pct = 0
                    
                    print(f"   {zge_emoji} Z哥规则：{rules_str} | 建议仓位：{position_pct}% | 提示：{hints_str}")
        
        # 保存结果到文件（持久化存储），传递执行参数用于文件名标识
        self._save_results_to_file(
            matched_results,
            results,
            stock_names,
            pick_date_str,
            category=category,
            max_stocks=max_stocks,
            min_similarity=min_similarity,
            M_days=M_days,
            use_cache=use_cache,
            filter_zge_risk=filter_zge_risk,
            rank_by_vol=rank_by_vol,
            brick_filter=brick_filter,
        )
        
        return {
            'results': results,
            'stock_names': stock_names,
            'matched': matched_results,
            'total_selected': total_selected,
        }
    
    def _save_results_to_file(self, matched_results, results, stock_names, pick_date, **kwargs):
        """保存选股结果到 JSON 文件
        :param kwargs: 包含执行参数，用于生成带参数标识的文件名
        """
        import json
        from pathlib import Path
        
        # 创建结果目录
        results_dir = Path(self.data_dir) / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # 确定日期
        if not pick_date:
            pick_date = datetime.now().strftime('%Y-%m-%d')
        
        # 构建包含参数标识的文件名
        param_parts = []
        
        # 添加类别参数
        category = kwargs.get('category', 'all')
        if category != 'all':
            param_parts.append(f'cat_{category}')
        
        # 添加最大股票数量参数
        max_stocks = kwargs.get('max_stocks')
        if max_stocks:
            param_parts.append(f'max_{max_stocks}')
        
        # 添加最小相似度参数
        min_similarity = kwargs.get('min_similarity')
        if min_similarity is not None and min_similarity != 60.0:
            param_parts.append(f'sim_{int(min_similarity)}')

        # 添加M天数参数
        M_days = kwargs.get('M_days')
        if M_days:
            param_parts.append(f'M_{M_days}')
        
        # 添加选股日期参数
        if pick_date and pick_date != datetime.now().strftime('%Y-%m-%d'):
            param_parts.append(f'date_{pick_date}')
        
        # 添加是否使用缓存参数
        use_cache = kwargs.get('use_cache')
        if use_cache is False:  # 只在显式禁用缓存时添加标识
            param_parts.append('nocache')
            
        # 添加Z哥过滤标识
        filter_zge_risk = kwargs.get('filter_zge_risk', False)
        if filter_zge_risk:
            param_parts.append('zge_filtered')

        # 添加砖型图过滤标识
        brick_filter = kwargs.get('brick_filter', False)
        if brick_filter:
            param_parts.append('brick')

        # 组合参数标识
        param_suffix = '_' + '_'.join(param_parts) if param_parts else ''
        
        # 生成输出文件名
        output_file = results_dir / f'{pick_date}_results{param_suffix}.json'
        
        # 准备保存数据
        save_data = {
            'pick_date': pick_date,
            'generated_at': datetime.now().isoformat(),
            'total_selected': sum(len(signals) for signals in results.values()),
            'matched_count': len(matched_results),
            'execution_params': {
                'category': kwargs.get('category', 'all'),
                'max_stocks': kwargs.get('max_stocks'),
                'min_similarity': kwargs.get('min_similarity'),
                'M_days': kwargs.get('M_days'),
                'pick_date': kwargs.get('pick_date'),
                'use_cache': kwargs.get('use_cache', True)
            },
            'top_results': [],
            'all_results': []
        }
        
        # Top 结果（带行业热度）
        rank_by_vol = kwargs.get('rank_by_vol', False)
        for i, r in enumerate(matched_results[:10], 1):
            item = {
                'rank': i,
                'code': r['stock_code'],
                'name': r['stock_name'],
                'industry': r['industry'],
                'industry_heat': r['industry_heat'],
                'industry_heat_change': r.get('industry_heat_change'),
            }
            if rank_by_vol:
                item['key_vol_ratio'] = r.get('key_vol_ratio', 0)
            else:
                item['similarity_score'] = r['similarity_score']
                item['matched_case'] = r['matched_case']
                item['description'] = r.get('description', '')
                item['breakdown'] = r.get('breakdown', {})
            save_data['top_results'].append(item)

        # 所有匹配结果（只保留前10名）
        for r in matched_results[:10]:
            item = {
                'code': r['stock_code'],
                'name': r['stock_name'],
                'industry': r['industry'],
                'industry_heat': r['industry_heat'],
                'industry_heat_change': r.get('industry_heat_change'),
                'signal_type': r.get('signal_type', ''),
                'price': r.get('close', 0),
                'j_value': r.get('J', 0)
            }
            if rank_by_vol:
                item['key_vol_ratio'] = r.get('key_vol_ratio', 0)
            else:
                item['similarity_score'] = r['similarity_score']
                item['matched_case'] = r['matched_case']
            save_data['all_results'].append(item)
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 结果已保存到: {output_file.resolve()}")
    
    def run_with_b1_match(self, category='all', max_stocks=None, min_similarity=60.0, M_days=None, pick_date=None, use_cache=False, force_update=False, filter_zge_risk=False, b1plus=False, rank_by_vol=False, brick_filter=False):
        """
        完整流程：更新 + 选股 + B1/B1Plus完美图形匹配 + 通知

        Args:
            category: 股票分类筛选
            max_stocks: 限制处理的股票数量
            min_similarity: 最小相似度阈值
            b1plus: 是否同时执行 B1Plus 流程
        """
        from datetime import datetime

        print("=" * 60)
        print("🚀 执行完整流程（含B1完美图形匹配）")
        if max_stocks:
            print(f"   快速测试模式：只处理前 {max_stocks} 只股票")
        if b1plus:
            print("   B1Plus: 启用（上涨中震荡平台整理型）")
        print("=" * 60)

        # 1. 更新数据（_smart_update 内部会检查本地数据是否已最新）
        self._smart_update(max_stocks=max_stocks, force_update=force_update)

        # 2. 选股 + B1完美图形匹配
        match_result = self.select_with_b1_match(
            category=category,
            max_stocks=max_stocks,
            use_cache=use_cache, force_update=force_update,
            min_similarity=min_similarity,
            M_days=M_days,
            pick_date=pick_date,
            filter_zge_risk=filter_zge_risk,
            rank_by_vol=rank_by_vol,
            brick_filter=brick_filter,
        )

        # 3. B1Plus 选股 + 匹配（如果启用）
        b1plus_result = None
        if b1plus:
            from strategy.pattern_config import B1PLUS_PERFECT_CASES, B1PLUS_SIMILARITY_WEIGHTS
            from pathlib import Path as _Path

            b1plus_cache = _Path(__file__).parent / 'data' / 'cache' / 'b1plus_pattern_library_cache.json'
            b1plus_result = self.select_with_b1_match(
                category=category,
                max_stocks=max_stocks,
                use_cache=use_cache, force_update=force_update,
                min_similarity=min_similarity,
                M_days=M_days,
                pick_date=pick_date,
                filter_zge_risk=filter_zge_risk,
                skip_kdj=True,
                cases_list=B1PLUS_PERFECT_CASES,
                weights=B1PLUS_SIMILARITY_WEIGHTS,
                cache_file=b1plus_cache,
                label='B1Plus',
                brick_filter=brick_filter,
            )
        
        # 3. 发送通知（暂时禁用）
        # if match_result.get('matched'):
        #     print("\n📤 发送通知...")
        #     self.notifier.send_b1_match_results(
        #         match_result['matched'],
        #         match_result.get('total_selected', 0)
        #     )
        #     print("✓ 通知发送完成")
        # else:
        #     print("\n⚠️ 没有匹配结果，跳过通知")

        result = {'b1': match_result}
        if b1plus_result is not None:
            result['b1plus'] = b1plus_result
        return result
    
    def run_schedule(self):
        """启动定时调度"""
        try:
            import schedule
        except ImportError:
            print("✗ 请安装 schedule: pip install schedule")
            return
        
        schedule_time = self.config.get('schedule', {}).get('time', '15:05')
        
        print("=" * 60)
        print(f"⏰ 启动定时调度")
        print(f"   每日 {schedule_time} 执行选股任务")
        print("=" * 60)
        
        # 设置定时任务
        schedule.every().day.at(schedule_time).do(self.run_full)
        
        print("\n按 Ctrl+C 停止")
        
        while True:
            schedule.run_pending()
            time.sleep(60)


def print_version():
    """打印版本信息"""
    import akshare
    import pandas
    
    print(f"A-Share Quant v{__version__}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"akshare: {akshare.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"System: {platform.system()}")
    print(f"B1 Pattern Match: 支持（基于双线+量比+形态三维匹配，10个历史案例，需 --b1-match 开启）")
    print(f"B1Plus Pattern Match: 支持（上涨中震荡平台整理型，跳过KDJ，2个历史案例，需 --b1-match --b1plus-match 开启）")


def main():
    parser = argparse.ArgumentParser(
        description='A股量化选股系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
命令:
  init             - 首次全量抓取历史数据（默认 6 年）
  run              - 完整流程（数据更新 + 选股 + 通知）
  web              - 启动 Web 服务器

参数:
  --max-stocks N   - 限制处理的股票数量（用于快速测试）
  --config PATH    - 配置文件路径 (默认：config/config.yaml)
  --category CAT   - 股票分类筛选：all(全部), bowl_center(回落碗中), near_duokong(靠近多空线), near_short_trend(靠近短期趋势线)
  --pick-date DATE - 指定选股日期，格式：YYYY-MM-DD（默认使用最新数据日期）
  --M-days N       - 碗口反弹策略的回溯天数 M

B1 完美图形匹配参数:
  --b1-match       - 启用 B1 完美图形匹配排序（默认禁用）
  --b1plus-match   - 启用 B1Plus 完美图形匹配（上涨中震荡平台整理型，跳过KDJ过滤）
  --min-similarity N - B1 匹配的最小相似度阈值 (默认：60，范围 0-100)

数据缓存参数:
  --force          - 强制从网络拉取最新数据，忽略缓存（等同于 --no-cache）
  --use-cache      - 使用缓存数据（行业数据/成交额数据），不从 API 重新拉取

通知与图表:
  --no-notify      - 跳过飞书通知发送，仅输出结果到控制台
  --no-chart       - 跳过 K 线图生成和发送，仅输出文本结果

Web 服务器参数:
  --host ADDR      - Web 服务器监听地址 (默认：0.0.0.0)
  --port N         - Web 服务器端口 (默认：5000)

其他:
  --version        - 显示版本信息并退出

示例:
  python main.py init                          # 首次抓取 6 年历史数据
  python main.py run                           # 完整流程（更新 + 选股 + 通知，不含B1匹配）
  python main.py run --force                   # 强制更新数据并执行选股（获取当日数据）
  python main.py run --b1-match                # 完整流程（启用 B1 完美图形匹配排序）
  python main.py run --max-stocks 100          # 只处理前 100 只股票（快速测试）
  python main.py run --category bowl_center    # 只筛选"回落碗中"分类的股票
  python main.py run --b1-match --min-similarity 70  # 启用B1匹配，相似度阈值70%
  python main.py run --M-days 20               # 碗口反弹策略使用 20 天回溯期
  python main.py run --pick-date 2026-03-25    # 回测指定日期的选股结果
  python main.py run --no-notify --no-chart    # 不发送通知和图表，仅控制台输出
  python main.py run --use-cache               # 使用缓存的行业/成交额数据，加速执行
  python main.py web --port 8080               # 启动 Web 服务器，监听 8080 端口
  python main.py --version                     # 显示版本信息

分类说明:
  all              - 全部（回落碗中 + 靠近多空线 + 靠近短期趋势线）
  bowl_center      - 回落碗中（优先级最高）
  near_duokong     - 靠近多空线（±duokong_pct%，默认 3%）
  near_short_trend - 靠近短期趋势线（±short_pct%，默认 2%）

B1 完美图形匹配:
  基于 10 个历史成功案例（双线 + 量比 + 形态三维相似度匹配）
  **默认不启用**，使用 --b1-match 参数开启
  --min-similarity 调整匹配阈值（默认 60%，范围 0-100）
  --b1plus-match 启用 B1Plus 匹配（上涨中震荡平台整理型，跳过KDJ过滤，2个历史案例）
  --rank-by-vol 跳过图形匹配，直接按突破当天成交量倍数排名

砖型图过滤:
  --brick-filter   启用砖型图三层过滤（形态+知行线+周线），在现有选股基础上叠加，默认关闭

横盘整理选股:
  --consolidation 启用横盘整理选股模式（高点后整理不破，多空线上升，收盘价在线上）
  --after-high-only 配合 --consolidation 使用，条件仅检查高点之后（默认检查整个60天窗口）

数据更新说明:
  默认行为：检查.update_cache.json 缓存，避免重复更新
  --force 参数：强制从 tushare/akshare 拉取最新数据，忽略缓存
  --use-cache 参数：使用缓存的行业热度/成交额数据，不从 API 重新拉取
        """
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='显示版本信息并退出'
    )

    parser.add_argument(
        'command',
        choices=['init', 'run', 'web'],
        nargs='?',
        help='要执行的命令: init(初始化数据), run(执行选股), web(启动Web服务器)'
    )

    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='限制处理的股票数量（用于快速测试）'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Web服务器监听地址 (默认: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web服务器端口 (默认: 5000)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=['all', 'bowl_center', 'near_duokong', 'near_short_trend'],
        default='all',
        help='筛选股票分类: all(全部), bowl_center(回落碗中), near_duokong(靠近多空线), near_short_trend(靠近短期趋势线)'
    )
    
    parser.add_argument(
        '--notify',
        action='store_true',
        default=False,
        help='发送飞书通知（默认不发送，仅输出结果到控制台）'
    )

    parser.add_argument(
        '--chart',
        action='store_true',
        default=False,
        help='生成K线图并发送飞书（需配合 --notify 使用，默认不生成）'
    )

    parser.add_argument(
        '--use-cache',
        action='store_true',
        default=True,
        help='使用缓存数据（行业数据/成交额数据），不从 API 重新拉取。默认行为是使用缓存，网络不可用时会自动保留缓存数据'
    )
    
    parser.add_argument(
        '--force',
        '--no-cache',
        action='store_true',
        default=False,
        dest='force_update',
        help='强制从网络拉取最新数据，忽略 .update_cache.json 缓存（用于数据更新）'
    )
    
    # 从配置读取B1PatternMatch默认值
    try:
        from strategy.pattern_config import MIN_SIMILARITY_SCORE
        default_min_similarity = MIN_SIMILARITY_SCORE
    except:
        default_min_similarity = 60.0
    
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=None,
        help=f'B1完美图形匹配的最小相似度阈值 (默认: {default_min_similarity})'
    )
    
    parser.add_argument(
        '--b1-match',
        action='store_true',
        default=False,
        dest='b1_match',
        help='启用B1完美图形匹配排序（默认禁用）'
    )

    parser.add_argument(
        '--b1plus-match',
        action='store_true',
        default=False,
        dest='b1plus_match',
        help='启用B1Plus完美图形匹配（上涨中震荡平台整理型，跳过KDJ过滤）'
    )

    parser.add_argument(
        '--rank-by-vol',
        action='store_true',
        default=False,
        help='跳过B1图形匹配，按突破当天成交量倍数排名（量比越高排越前）'
    )
    
    parser.add_argument(
        '--pick-date',
        type=str,
        default=None,
        help='指定选股日期，格式：YYYY-MM-DD（默认使用最新数据日期）'
    )
    # 从配置读取 BowlReboundStrategy 的 M 默认值
    try:
        from utils.config_loader import get_strategy_params
        bowl_params = get_strategy_params('BowlReboundStrategy')
        default_M_days = bowl_params.get('M', 30)
    except:
        default_M_days = 30

    parser.add_argument(
        '--M-days',
        type=int,
        default=None,
        dest='M_days',
        help=f'碗口反弹策略的回溯天数 M (默认：{default_M_days})'
    )

    parser.add_argument(
        '--filter-zge-risk',
        action='store_true',
        default=False,
        help='开启后从最终选股结果中剔除所有触发Z哥风控规则的股票，默认关闭仅做标注'
    )

    parser.add_argument(
        '--brick-filter',
        action='store_true',
        default=False,
        dest='brick_filter',
        help='启用砖型图三层过滤（形态+知行线+周线），在现有选股基础上叠加，默认关闭'
    )

    parser.add_argument(
        '--consolidation',
        action='store_true',
        default=False,
        help='横盘整理选股模式：高点后整理不破，多空线上升，收盘价在线上（跳过B1匹配和Z哥过滤）'
    )

    parser.add_argument(
        '--after-high-only',
        action='store_true',
        default=False,
        help='配合--consolidation使用：条件2/3/4仅检查高点之后的整理期（默认检查整个60天窗口）'
    )

    args = parser.parse_args()

    # 处理 --version 参数
    if args.version:
        print_version()
        sys.exit(0)

    # 检查命令是否提供
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 参数有效性校验
    if args.min_similarity is not None and (args.min_similarity < 0 or args.min_similarity > 100):
        logger.error(f"--min-similarity 参数无效: {args.min_similarity}，必须在0-100范围内")
        sys.exit(1)

    if args.max_stocks is not None and args.max_stocks <= 0:
        logger.error(f"--max-stocks 参数无效: {args.max_stocks}，必须大于0")
        sys.exit(1)
    
    if args.M_days is not None and args.M_days <= 0:
        logger.error(f"--M-days 参数无效: {args.M_days}，必须大于0")
        sys.exit(1)
    
    if args.pick_date is not None:
        # 校验日期格式是否正确
        try:
            datetime.strptime(args.pick_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"--pick-date 参数格式无效: {args.pick_date}，必须是YYYY-MM-DD格式")
            sys.exit(1)
    
    # 切换工作目录
    os.chdir(project_root)
    
    # 创建系统实例
    quant = QuantSystem(args.config)
    
    # 执行命令
    if args.command == 'init':
        quant.init_data(max_stocks=args.max_stocks)
    
    elif args.command == 'run':
        if args.consolidation:
            # 横盘整理选股模式
            pick_date = args.pick_date
            quant.run_consolidation(
                max_stocks=args.max_stocks,
                notify=args.notify,
                chart=args.chart,
                pick_date=pick_date,
                force_update=args.force_update,
                after_high_only=args.after_high_only,
            )
        elif args.b1_match:
            # 启用B1完美图形匹配

            min_sim = args.min_similarity if args.min_similarity is not None else default_min_similarity
            M_days = args.M_days if args.M_days is not None else None
            pick_date = args.pick_date or datetime.now().strftime('%Y-%m-%d')
            quant.run_with_b1_match(
                category=args.category,
                max_stocks=args.max_stocks,
                use_cache=args.use_cache, force_update=args.force_update,
                min_similarity=min_sim,
                M_days=M_days,
                pick_date=pick_date,
                filter_zge_risk=args.filter_zge_risk,
                b1plus=args.b1plus_match,
                rank_by_vol=args.rank_by_vol,
                brick_filter=args.brick_filter,
            )
        else:
            # 原有选股流程（不带B1匹配）
            M_days = args.M_days if args.M_days is not None else None
            quant.run_full(
                category=args.category,
                max_stocks=args.max_stocks,
                notify=args.notify,
                chart=args.chart,
                M_days=M_days, force_update=args.force_update,
                filter_zge_risk=args.filter_zge_risk,
                brick_filter=args.brick_filter,
            )
    
    elif args.command == 'web':
        # 启动Web服务器
        from web_server import run_web_server
        run_web_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()