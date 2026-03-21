"""
飞书群通知模块（完全兼容原有DingTalkNotifier接口，无需修改其他代码）
"""
import os
import requests
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime
from pathlib import Path
import pandas as pd

# 导入K线图模块
try:
    # 优先使用快速版
    from utils.kline_chart_fast import generate_kline_chart_fast as generate_kline_chart
    KLINE_CHART_AVAILABLE = True
    print("✓ 使用快速K线图生成")
except ImportError:
    try:
        from utils.kline_chart import generate_kline_chart
        KLINE_CHART_AVAILABLE = True
    except ImportError:
        KLINE_CHART_AVAILABLE = False
        print("警告: K线图模块未安装，图片功能不可用")


class RateLimiter:
    """限流器 - 控制每分钟发送数量"""
    
    def __init__(self, max_per_minute=20, min_interval=1.0):
        """
        Args:
            max_per_minute: 每分钟最大发送次数（飞书默认限制约20条/分钟）
            min_interval: 每次发送最小间隔（秒）
        """
        self.max_per_minute = max_per_minute
        self.min_interval = min_interval
        self.send_times = []  # 记录每次发送的时间戳
        self._lock_time = 0   # 锁定时间（遇到限速错误时延长）"
    
    def acquire(self):
        """
        获取发送许可，必要时阻塞等待
        Returns: 实际等待的秒数
        """
        now = time.time()
        
        # 清理1分钟前的记录
        self.send_times = [t for t in self.send_times if now - t < 60]
        
        # 检查是否处于锁定状态（遇到过限速错误）
        if now < self._lock_time:
            wait = self._lock_time - now
            time.sleep(wait)
            now = time.time()
        
        # 检查每分钟限制
        if len(self.send_times) >= self.max_per_minute:
            # 需要等到最早一条记录超过1分钟
            oldest = self.send_times[0]
            wait = 60 - (now - oldest) + 0.1  # 多等0.1秒确保
            if wait > 0:
                print(f"    ⏱️ 限流: 已达到每分钟{self.max_per_minute}条限制，等待{wait:.1f}秒...")
                time.sleep(wait)
                now = time.time()
                # 重新清理
                self.send_times = [t for t in self.send_times if now - t < 60]
        
        # 检查最小间隔
        if self.send_times:
            last_send = self.send_times[-1]
            elapsed = now - last_send
            if elapsed < self.min_interval:
                wait = self.min_interval - elapsed
                time.sleep(wait)
                now = time.time()
        
        # 记录本次发送时间
        self.send_times.append(now)
        return now
    
    def on_rate_limit_error(self, retry_count=0):
        """
        遇到限速错误时的处理 - 指数退避
        Args:
            retry_count: 当前重试次数
        """
        backoff = min(2 ** retry_count, 30)  # 最大等待30秒
        self._lock_time = time.time() + backoff
        print(f"    ⏱️ 遇到限速，退避等待{backoff}秒...")
        time.sleep(backoff)


class FeishuNotifier:
    """飞书通知器（接口完全兼容DingTalkNotifier）"""
    
    def __init__(self, webhook_url=None, secret=None):
        self.webhook_url = webhook_url
        self.secret = secret
        self._last_send_time = 0
        self._min_interval = 1.0  # 最小发送间隔1秒
        self._rate_limiter = RateLimiter(max_per_minute=20, min_interval=1.0)  # 限流器
    
    def _generate_sign(self):
        """生成飞书签名"""
        if not self.secret:
            return "", ""

        timestamp = str(round(time.time()))
        secret_enc = self.secret.encode('utf-8')
        string_to_sign = f'{timestamp}\n{self.secret}'
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = base64.b64encode(hmac_code).decode('utf-8')
        return timestamp, sign

    def _send_request(self, data: dict, max_retries=3) -> bool:
        """使用OpenClaw内置消息能力发送到当前飞书会话（不需要配置webhook）
        
        Args:
            data: 要发送的数据
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries + 1):
            # 使用限流器获取发送许可
            self._rate_limiter.acquire()
            
            try:
                # 解析消息内容
                msg_type = data.get('msg_type', 'text')
                content = ''
                
                if msg_type == 'text':
                    content = data.get('content', {}).get('text', '')
                elif msg_type == 'markdown':
                    content = data.get('content', {}).get('text', '')
                elif msg_type == 'image':
                    # 图片消息暂不支持直接发送，转为文字提示
                    content = "📈 已生成K线图（图片发送功能待升级）"
                
                if not content:
                    print("⚠️ 空消息，跳过发送")
                    return False
                
                # 使用OpenClaw内置消息工具发送到当前飞书会话
                import subprocess
                cmd = [
                    'openclaw', 'message', 'send',
                    '--channel', 'feishu',
                    '--message', content
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                # 更新最后发送时间
                self._last_send_time = time.time()
                
                if result.returncode == 0:
                    return True
                else:
                    print(f"    ✗ 消息发送失败: {result.stderr}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return False

            except Exception as e:
                print(f"    ✗ 发送异常: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return False
        
        return False
    
    def send_text(self, content):
        """
        发送纯文本消息（兼容原有接口）
        """
        if not self.webhook_url:
            print("警告: 未配置飞书 webhook")
            return False
        
        data = {
            "msg_type": "text",
            "content": {
                "text": content
            }
        }
        
        if self._send_request(data):
            print("✓ 飞书通知发送成功")
            return True
        return False
    
    def send_markdown(self, title, content):
        """
        发送 Markdown 格式消息（兼容原有接口）
        """
        if not self.webhook_url:
            print("警告: 未配置飞书 webhook")
            return False
        
        # 飞书markdown不需要title参数，整合到内容里
        full_content = f"# {title}\n\n{content}"
        
        data = {
            "msg_type": "markdown",
            "content": {
                "title": title,
                "text": full_content
            }
        }
        
        if self._send_request(data):
            print("✓ 飞书Markdown通知发送成功")
            return True
        return False

    def send_image(self, image_path: str, title: str = "K线图") -> bool:
        """
        发送图片到飞书
        飞书支持base64图片直接上传
        """
        if not self.webhook_url:
            print("警告: 未配置飞书 webhook")
            return False

        if not Path(image_path).exists():
            print(f"✗ 图片文件不存在: {image_path}")
            return False

        try:
            # 读取图片
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # 转为base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # 检查大小（飞书限制约2MB）
            if len(image_data) > 2 * 1024 * 1024:
                print(f"⚠️ 图片超过2MB，可能发送失败")

            # 发送图片消息
            data = {
                "msg_type": "image",
                "content": {
                    "image_key": image_base64
                }
            }

            # 先发送标题
            if title:
                self.send_text(f"📈 {title}")

            # 发送图片
            success = self._send_request(data)

            # 发送成功后删除本地图片
            if success:
                import os
                os.remove(image_path)
                print(f"✓ 已删除本地图片: {image_path}")

            return success

        except Exception as e:
            print(f"✗ 图片发送失败: {e}")
            return False

    def format_stock_results(self, results, stock_names=None, category_filter='all'):
        """
        格式化选股结果为 Markdown (适配飞书手机端)
        :param results: {strategy_name: [signals]} 格式的结果
        :param stock_names: {code: name} 股票名称字典
        :param category_filter: 分类筛选，'all'表示全部
        """
        if stock_names is None:
            stock_names = {}
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 分类名称映射
        category_names = {
            'bowl_center': '🥣 回落碗中',
            'near_duokong': '📊 靠近多空线',
            'near_short_trend': '📈 靠近短期趋势线'
        }
        
        # 筛选标签
        filter_label = "全部" if category_filter == 'all' else category_names.get(category_filter, category_filter)
        
        content = f"📊 A股量化选股结果\n\n"
        content += f"⏰ 时间: {now}\n"
        content += f"🔍 筛选: {filter_label}\n"
        content += "━" * 30 + "\n\n"
        
        total_signals = 0
        # 按分类统计
        category_count = {'bowl_center': 0, 'near_duokong': 0, 'near_short_trend': 0}
        
        for strategy_name, signals in results.items():
            content += f"🎯 {strategy_name}\n\n"
            
            if not signals:
                content += "暂无选股信号\n\n"
                continue
            
            # 按分类分组，并根据 category_filter 过滤
            category_groups = {}
            for signal in signals:
                for s in signal['signals']:
                    cat = s.get('category', 'unknown')
                    # 如果指定了分类筛选，只保留对应分类
                    if category_filter != 'all' and cat != category_filter:
                        continue
                    if cat not in category_groups:
                        category_groups[cat] = []
                    category_groups[cat].append((signal, s))
                    category_count[cat] = category_count.get(cat, 0) + 1
            
            total_signals += sum(len(group) for group in category_groups.values())
            
            # 按优先级顺序显示分类
            display_order = ['bowl_center', 'near_duokong', 'near_short_trend']
            # 如果指定了分类，只显示该分类
            if category_filter != 'all' and category_filter in display_order:
                display_order = [category_filter]
            
            for cat in display_order:
                if cat in category_groups:
                    group_signals = category_groups[cat]
                    cat_name = category_names.get(cat, cat)
                    content += f"{cat_name} ({len(group_signals)}只)\n"
                    content += "-" * 20 + "\n"
                    
                    for i, (signal, s) in enumerate(group_signals, 1):
                        code = signal['code']
                        name = signal.get('name', stock_names.get(code, '未知'))
                        close = s.get('close', '-')
                        j_val = s.get('J', '-')
                        key_date = s.get('key_candle_date', '-')
                        if isinstance(key_date, pd.Timestamp):
                            key_date = key_date.strftime("%m-%d")
                        reasons = ' '.join(s.get('reasons', []))
                        
                        # 手机端友好的格式
                        content += f"{i}. {code} {name}\n"
                        content += f"   💰 价格: {close}  |  J值: {j_val}\n"
                        content += f"   📅 关键K线: {key_date}\n"
                        content += f"   📝 {reasons}\n\n"
                    
                    content += "\n"
            
            content += "━" * 30 + "\n\n"
        
        # 显示分类统计
        content += "📊 分类统计:\n"
        content += f"   🥣 回落碗中: {category_count.get('bowl_center', 0)} 只\n"
        content += f"   📊 靠近多空线: {category_count.get('near_duokong', 0)} 只\n"
        content += f"   📈 靠近短期趋势线: {category_count.get('near_short_trend', 0)} 只\n"
        content += f"   📈 共选出: {total_signals} 只\n\n"
        content += "⚠️ 提示: 以上结果仅供参考，不构成投资建议"
        
        return content
    
    def send_stock_selection(self, results, stock_names=None, category_filter='all'):
        """
        发送选股结果到飞书（兼容原有接口）
        """
        content = self.format_stock_results(results, stock_names, category_filter)
        return self.send_markdown("A股选股结果", content)

    def _format_stock_info_message(self, stock_code, stock_name, category, params, signal):
        """
        格式化股票信息文字消息
        """
        category_names = {
            'bowl_center': '🥣 回落碗中',
            'near_duokong': '📊 靠近多空线',
            'near_short_trend': '📈 靠近短期趋势线'
        }
        category_name = category_names.get(category, category)
        
        # 格式化参数
        cap = params.get('CAP', 4000000000)
        cap_display = f"{cap/1e8:.0f}亿" if cap >= 1e8 else f"{cap/1e4:.0f}万"
        
        # 获取信号数据
        close = signal.get('close', '-')
        j_val = signal.get('J', '-')
        key_date = signal.get('key_candle_date', '-')
        if hasattr(key_date, 'strftime'):
            key_date = key_date.strftime("%m-%d")
        reasons = ' '.join(signal.get('reasons', []))
        
        message = f"""### 📊 {stock_code} {stock_name}

**分类**: {category_name}
**价格**: {close} | **J值**: {j_val}
**关键K线日期**: {key_date}
**入选理由**: {reasons}
"""
        return message

    def send_stock_selection_with_charts(
        self, 
        results, 
        stock_names=None, 
        category_filter='all',
        stock_data_dict=None,
        params=None,
        send_text_first: bool = True
    ):
        """
        发送选股结果（带K线图）到飞书（兼容原有接口）
        """
        if not KLINE_CHART_AVAILABLE:
            print("⚠️ K线图模块不可用，发送普通文本消息")
            return self.send_stock_selection(results, stock_names, category_filter)
        
        if stock_names is None:
            stock_names = {}
        
        if params is None:
            params = {
                'N': 4,
                'M': 15,
                'CAP': 4000000000,
                'J_VAL': 30,
                'duokong_pct': 3,
                'short_pct': 2
            }
        
        # 先发送汇总消息
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        category_names = {
            'bowl_center': '🥣 回落碗中',
            'near_duokong': '📊 靠近多空线',
            'near_short_trend': '📈 靠近短期趋势线'
        }
        
        total_sent = 0
        total_failed = 0
        chart_count = 0

        # 统计各分类数量
        category_count = {'bowl_center': 0, 'near_duokong': 0, 'near_short_trend': 0}
        for strategy_name, signals in results.items():
            for signal in signals:
                for s in signal['signals']:
                    cat = s.get('category', 'unknown')
                    if category_filter == 'all' or cat == category_filter:
                        category_count[cat] = category_count.get(cat, 0) + 1
        
        # 发送汇总消息
        summary = f"🎯 BowlReboundStrategy:\n"
        summary += f"N: {params.get('N', 4)} (成交量倍数)\n"
        summary += f"M: {params.get('M', 15)} (回溯天数)\n"
        summary += f"CAP: {params.get('CAP', 4000000000)} (40亿市值门槛)\n"
        summary += f"J_VAL: {params.get('J_VAL', 30)} (J值上限)\n"
        summary += f"duokong_pct: {params.get('duokong_pct', 3)}\n"
        summary += f"short_pct: {params.get('short_pct', 2)}\n"
        summary += f"M1: {params.get('M1', 14)} (MA周期)\n"
        summary += f"M2: {params.get('M2', 28)} (MA周期)\n"
        summary += f"M3: {params.get('M3', 57)} (MA周期)\n"
        summary += f"M4: {params.get('M4', 114)} (MA周期)\n\n"
        
        summary += f"⏰ {now}\n"
        if category_filter != 'all':
            summary += f"🔍 筛选: {category_names.get(category_filter, category_filter)}\n"
        summary += "━" * 20 + "\n\n"
        summary += f"🥣 回落碗中: {category_count.get('bowl_center', 0)} 只\n"
        summary += f"📊 靠近多空线: {category_count.get('near_duokong', 0)} 只\n"
        summary += f"📈 靠近短期趋势线: {category_count.get('near_short_trend', 0)} 只\n"
        total = sum(category_count.values())
        summary += f"📈 共选出: {total} 只\n\n"
        
        if stock_data_dict:
            summary += "📈 正在为每只股票生成K线图...\n"
            if send_text_first:
                summary += "（文字说明与图片分离发送，节省流量）\n"
        else:
            summary += "详细列表见下方消息 👇"
        
        if self.send_text(summary):
            total_sent += 1
        else:
            total_failed += 1
        
        time.sleep(1)
        
        # 如果提供了股票数据，生成并发送K线图
        if stock_data_dict:
            print(f"📊 准备发送 {len(stock_data_dict)} 只股票的K线图...")
            for strategy_name, signals in results.items():
                print(f"  处理策略: {strategy_name}, {len(signals)} 只信号")
                for signal in signals:
                    code = signal['code']
                    name = signal.get('name', stock_names.get(code, '未知'))
                    
                    for s in signal['signals']:
                        cat = s.get('category', 'unknown')
                        if category_filter != 'all' and cat != category_filter:
                            continue
                        
                        # 获取股票数据
                        if code not in stock_data_dict:
                            print(f"  ⚠️ {code} 不在stock_data_dict中")
                            continue
                        
                        df = stock_data_dict[code]
                        if df.empty:
                            print(f"  ⚠️ {code} 数据为空")
                            continue
                        
                        try:
                            print(f"  📈 处理 {code} {name}...")
                            # 准备关键K线日期
                            key_date = s.get('key_candle_date')
                            key_dates = [key_date] if key_date else []
                            
                            if send_text_first:
                                # 先发送文字说明
                                info_message = self._format_stock_info_message(
                                    code, name, cat, params, s
                                )
                                cat_name = category_names.get(cat, cat)
                                title = f"{code} {name}"
                                print(f"    发送文字...")
                                self.send_markdown(title, info_message)
                                
                                print(f"    生成K线图...")
                                t0 = time.time()
                                # 再生成无文字版本K线图
                                chart_path = generate_kline_chart(
                                    stock_code=code,
                                    stock_name=name,
                                    df=df,
                                    category=cat,
                                    params=params,
                                    key_candle_dates=key_dates,
                                    output_dir='/tmp/kline_charts',
                                    show_text=False,  # 无文字版本
                                    show_legend=True
                                )
                                t1 = time.time()
                                print(f"    生成K线图耗时: {t1-t0:.2f}秒")
                                
                                print(f"    发送图片...")
                                t0 = time.time()
                                # 发送图片（标题简化）
                                if self.send_image(chart_path, f"{code} K线图"):
                                    chart_count += 1
                                t1 = time.time()
                                print(f"    发送图片耗时: {t1-t0:.2f}秒")
                            else:
                                # 旧方式：生成带文字的K线图
                                chart_path = generate_kline_chart(
                                    stock_code=code,
                                    stock_name=name,
                                    df=df,
                                    category=cat,
                                    params=params,
                                    key_candle_dates=key_dates,
                                    output_dir='/tmp/kline_charts',
                                    show_text=True,
                                    show_legend=True
                                )
                                
                                # 发送图片信息
                                cat_name = category_names.get(cat, cat)
                                title = f"{code} {name} - {cat_name}"
                                if self.send_image(chart_path, title):
                                    chart_count += 1

                        except Exception as e:
                            print(f"✗ 生成 {code} 的K线图失败: {e}")
                            continue
        
        # 发送普通文本详情（作为备份）
        text_result = self.send_stock_selection(results, stock_names, category_filter)

        print(f"\n✓ 已发送 {chart_count} 张K线图到飞书")

        return text_result

    def send_b1_match_results(self, results: list, total_selected: int):
        """
        发送带B1完美图形匹配的选股结果（兼容原有接口）
        """
        if not results:
            return
        
        from datetime import datetime
        from strategy.pattern_config import TOP_N_RESULTS
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 分类名称映射
        category_names = {
            'bowl_center': '[回落碗中]',
            'near_duokong': '[靠近多空线]',
            'near_short_trend': '[靠近短期趋势线]'
        }
        
        # 构建Markdown消息
        lines = [
            "## 选股结果（按B1完美图形相似度排序）",
            "",
            f"时间: {now}",
            f"策略筛选: {total_selected} 只 | B1 Top匹配: {len(results)} 只",
            "━" * 30,
            "",
        ]
        
        # 只显示前N个（从配置读取）
        for i, r in enumerate(results[:TOP_N_RESULTS], 1):
            rank = f"{i}."
            if i == 1:
                rank = "🥇"
            elif i == 2:
                rank = "🥈"
            elif i == 3:
                rank = "🥉"
            
            stock_code = r.get('stock_code', '')
            stock_name = r.get('stock_name', '')
            score = r.get('similarity_score', 0)
            matched_case = r.get('matched_case', '')
            matched_date = r.get('matched_date', '')
            category = r.get('category', '')
            close = r.get('close', '-')
            j_val = r.get('J', '-')
            breakdown = r.get('breakdown', {})
            
            lines.append(f"{rank} **{stock_code}** {stock_name}  **相似度: {score}%**")
            lines.append(f"   匹配: {matched_case} ({matched_date})")
            
            trend_score = breakdown.get('trend_structure', 0)
            kdj_score = breakdown.get('kdj_state', 0)
            vol_score = breakdown.get('volume_pattern', 0)
            shape_score = breakdown.get('price_shape', 0)
            lines.append(f"   分项: 趋势{trend_score}% | KDJ{kdj_score}% | 量能{vol_score}% | 形态{shape_score}%")
            
            cat_name = category_names.get(category, category)
            lines.append(f"   策略: {cat_name} | 价格: {close} | J值: {j_val}")
            lines.append("")
        
        lines.append("---")
        lines.append("**B1匹配逻辑**: 基于双线+量比+形态三维相似度")
        lines.append("**案例来源**: 10个历史成功案例")
        
        content = "\n".join(lines)
        
        self.send_markdown("B1完美图形匹配选股结果", content)
