"""
股票过滤工具类（所有过滤规则统一维护，保证全局一致）
"""
import pandas as pd

def is_valid_stock(stock_name: str, df=None) -> bool:
    """
    判断股票是否有效（过滤退市、ST、异常股票）
    过滤规则与原有逻辑100%一致，统一维护避免重复代码导致的规则不一致
    
    Args:
        stock_name: 股票名称
        df: 股票数据DataFrame（可选，用于检查数据有效性）
    
    Returns:
        bool: True表示有效，False表示需要过滤
    """
    # 过滤退市/异常股票
    invalid_keywords = ['退', '未知', '退市', '已退']
    if any(kw in stock_name for kw in invalid_keywords):
        return False
    
    # 过滤 ST/*ST 股票
    if stock_name.startswith('ST') or stock_name.startswith('*ST'):
        return False
    
    # 检查数据有效性
    if df is not None:
        if df.empty or len(df) < 60:
            return False
        # 检查最新一天是否有有效交易
        latest = df.iloc[0]
        if latest['volume'] <= 0 or pd.isna(latest['close']):
            return False
        # 检查KDJ数据是否异常
        recent_df = df.head(30)
        if 'J' in recent_df.columns and recent_df['J'].abs().mean() > 80:
            return False
    
    return True
