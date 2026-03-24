#!/usr/bin/env python3
"""
选股结果增强模块

功能：
1. 为选股结果添加行业信息
2. 计算行业热度分数
3. 按相似度 + 行业热度综合排序

使用方法:
    from enhance_stock_selection import enhance_with_industry_heat

    # 在 select_with_b1_match 之后调用
    enhanced_results = enhance_with_industry_heat(matched_results, stock_data_dict, data_dir)
"""
from pathlib import Path
from datetime import datetime
from utils.industry_fetcher import IndustryFetcher, IndustryHeatCalculator


def enhance_with_industry_heat(matched_results: list, stock_data_dict: dict, data_dir: str):
    """
    为选股结果添加行业和行业热度信息

    Args:
        matched_results: B1 匹配结果列表
        stock_data_dict: 股票数据字典 {code: (name, df)}
        data_dir: 数据目录路径

    Returns:
        增强后的结果列表（每个元素增加 industry 和 industry_heat 字段）
    """
    if not matched_results:
        return matched_results

    print("\n🔍 正在计算行业热度...")

    # 初始化行业数据获取器和热度计算器
    try:
        industry_fetcher = IndustryFetcher(Path(data_dir) / 'industry_cache')
        industry_fetcher.load_industry_mapping()
        industry_calc = IndustryHeatCalculator(industry_fetcher)
        print(f"✓ 行业数据加载完成：{len(industry_fetcher.get_all_industries())} 个行业")
    except Exception as e:
        print(f"⚠️ 行业数据模块初始化失败：{e}")
        # 返回原结果，不添加行业信息
        for r in matched_results:
            r['industry'] = '未知'
            r['industry_heat'] = None
        return matched_results

    # 获取今日日期
    today = datetime.now().date()
    all_stock_codes = list(stock_data_dict.keys())

    # 为每只股票添加行业信息
    enhanced = []
    for r in matched_results:
        code = r.get('stock_code', '')

        # 获取行业
        industry = industry_fetcher.get_industry_for_stock(code)
        industry_heat = None

        if industry:
            # 计算行业热度
            try:
                industry_heat = industry_calc.calculate_industry_heat(
                    industry, today, stock_data_dict, all_stock_codes)
            except Exception as e:
                print(f"  ⚠️ 计算 {code} 行业热度失败：{e}")

        # 添加字段
        r['industry'] = industry if industry else '未知'
        r['industry_heat'] = round(industry_heat, 1) if industry_heat is not None else None

        enhanced.append(r)

    # 按相似度 + 行业热度综合排序
    # 排序规则：相似度优先，相似度相同时热度高的在前
    enhanced.sort(key=lambda x: (x['similarity_score'], x['industry_heat'] or 0), reverse=True)

    print(f"✓ 行业热度计算完成")

    # 显示行业热度统计
    heat_values = [r['industry_heat'] for r in enhanced if r['industry_heat'] is not None]
    if heat_values:
        avg_heat = sum(heat_values) / len(heat_values)
        max_heat = max(heat_values)
        min_heat = min(heat_values)
        high_heat_count = len([h for h in heat_values if h >= 70])
        print(f"  平均热度：{avg_heat:.1f}分 | 最高：{max_heat:.1f}分 | 最低：{min_heat:.1f}分")
        print(f"  高热度 (≥70 分) 股票：{high_heat_count} 只 ({high_heat_count/len(enhanced)*100:.1f}%)")

    return enhanced


def get_industry_insights(matched_results: list) -> dict:
    """
    分析选股结果的行业分布和热度情况

    Args:
        matched_results: 已包含 industry 和 industry_heat 字段的结果列表

    Returns:
        行业洞察字典
    """
    if not matched_results:
        return {}

    # 按行业分组统计
    industry_groups = {}
    for r in matched_results:
        industry = r.get('industry', '未知')
        if industry not in industry_groups:
            industry_groups[industry] = []
        industry_groups[industry].append(r)

    # 计算每个行业的统计信息
    insights = {
        'total_stocks': len(matched_results),
        'industries': {}
    }

    for industry, stocks in industry_groups.items():
        heat_values = [r['industry_heat'] for r in stocks if r['industry_heat'] is not None]
        avg_heat = sum(heat_values) / len(heat_values) if heat_values else 0
        avg_similarity = sum(r['similarity_score'] for r in stocks) / len(stocks)

        insights['industries'][industry] = {
            'count': len(stocks),
            'avg_heat': round(avg_heat, 1),
            'avg_similarity': round(avg_similarity, 1),
            'stocks': stocks
        }

    # 找出最佳行业（平均热度最高）
    if insights['industries']:
        best_industry = max(insights['industries'].items(),
                           key=lambda x: x[1]['avg_heat'])
        insights['best_industry'] = best_industry[0]
        insights['best_industry_heat'] = best_industry[1]['avg_heat']

    return insights


def print_enhanced_results(enhanced_results: list, top_n: int = 10):
    """
    打印增强后的选股结果（包含行业热度）

    Args:
        enhanced_results: 已包含行业信息的结果列表
        top_n: 显示前 N 只股票
    """
    if not enhanced_results:
        print("无匹配结果")
        return

    from strategy.pattern_config import TOP_N_RESULTS
    top_n = TOP_N_RESULTS if top_n is None else top_n

    print("\n" + "=" * 60)
    print(f"📊 Top {top_n} B1 完美图形匹配结果（含行业热度）")
    print("=" * 60)

    for i, r in enumerate(enhanced_results[:top_n], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

        # 行业热度显示
        heat = r.get('industry_heat')
        if heat is not None:
            if heat >= 70:
                heat_emoji = "🔥"
                heat_label = f"{heat}分 (高热度)"
            elif heat >= 50:
                heat_emoji = "📈"
                heat_label = f"{heat}分 (中等)"
            else:
                heat_emoji = "❄️"
                heat_label = f"{heat}分 (冷门)"
        else:
            heat_emoji = "❓"
            heat_label = "N/A"

        print(f"{emoji} {r['stock_code']} {r['stock_name']}")
        print(f"   相似度：{r['similarity_score']}% | 行业：{heat_emoji} {r['industry']} ({heat_label})")
        print(f"   匹配：{r['matched_case']}")

        bd = r.get('breakdown', {})
        print(f"   趋势:{bd.get('trend_structure', 0)}% "
              f"KDJ:{bd.get('kdj_state', 0)}% "
              f"量能:{bd.get('volume_pattern', 0)}% "
              f"形态:{bd.get('price_shape', 0)}%")
        print()

    # 显示行业洞察
    insights = get_industry_insights(enhanced_results)
    if insights.get('best_industry'):
        print("-" * 60)
        print(f"💡 行业洞察：{insights['best_industry']} 行业平均热度最高 "
              f"({insights['best_industry_heat']}分)")


if __name__ == '__main__':
    # 测试代码
    print("行业热度增强模块测试")
    print("=" * 50)

    # 模拟数据
    test_results = [
        {'stock_code': '600519', 'stock_name': '贵州茅台', 'similarity_score': 95.5},
        {'stock_code': '000858', 'stock_name': '五粮液', 'similarity_score': 92.0},
        {'stock_code': '300750', 'stock_name': '宁德时代', 'similarity_score': 88.5},
    ]

    print("模拟数据：", test_results)
    print("\n提示：此模块需要在选股后调用，需要 stock_data_dict 参数")
