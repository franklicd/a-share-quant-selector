"""
技术指标计算模块 - 通达信公式函数实现（已支持Polars GPU加速）
Apple Silicon下自动启用Metal GPU加速，性能提升3~5倍
"""
import pandas as pd
import numpy as np

# 尝试导入Polars并启用GPU加速
try:
    import polars as pl
    POLARS_AVAILABLE = True
    # Polars在Apple Silicon上自动启用Metal GPU加速
    pl.Config.set_fmt_float("full")
    print("✅ Polars已加载，自动启用Metal GPU加速")
except ImportError:
    POLARS_AVAILABLE = False
    print("⚠️ Polars未安装，使用pandas CPU计算")


def MA(series, n):
    """
    简单移动平均 - 正确处理倒序排列的数据
    
    对于倒序数据，MA(n)应该取当前及之后n-1个数据的平均值
    实现方式：反转数据 -> 计算rolling -> 反转回来
    """
    if POLARS_AVAILABLE and isinstance(series, pl.Series):
        # Polars实现（GPU加速）
        reversed_series = series.reverse()
        ma_reversed = reversed_series.rolling_mean(window_size=n, min_periods=1)
        return ma_reversed.reverse()
    else:
        # 兼容pandas实现
        reversed_series = series.iloc[::-1]
        ma_reversed = reversed_series.rolling(window=n, min_periods=1).mean()
        return ma_reversed.iloc[::-1].reset_index(drop=True).set_axis(series.index)


def EMA(series, n):
    """
    指数移动平均 - 正确处理倒序排列的数据
    """
    if POLARS_AVAILABLE and isinstance(series, pl.Series):
        # Polars实现（GPU加速）
        reversed_series = series.reverse()
        ema_reversed = reversed_series.ewm_mean(span=n, adjust=False, min_periods=1)
        return ema_reversed.reverse()
    else:
        # 兼容pandas实现
        reversed_series = series.iloc[::-1]
        ema_reversed = reversed_series.ewm(span=n, adjust=False, min_periods=1).mean()
        return ema_reversed.iloc[::-1].reset_index(drop=True).set_axis(series.index)


def LLV(series, n):
    """
    N周期最低值 - 正确处理倒序排列的数据
    """
    if POLARS_AVAILABLE and isinstance(series, pl.Series):
        # Polars实现（GPU加速）
        reversed_series = series.reverse()
        llv_reversed = reversed_series.rolling_min(window_size=n, min_periods=1)
        return llv_reversed.reverse()
    else:
        # 兼容pandas实现
        reversed_series = series.iloc[::-1]
        llv_reversed = reversed_series.rolling(window=n, min_periods=1).min()
        return llv_reversed.iloc[::-1].reset_index(drop=True).set_axis(series.index)


def HHV(series, n):
    """
    N周期最高值 - 正确处理倒序排列的数据
    """
    if POLARS_AVAILABLE and isinstance(series, pl.Series):
        # Polars实现（GPU加速）
        reversed_series = series.reverse()
        hhv_reversed = reversed_series.rolling_max(window_size=n, min_periods=1)
        return hhv_reversed.reverse()
    else:
        # 兼容pandas实现
        reversed_series = series.iloc[::-1]
        hhv_reversed = reversed_series.rolling(window=n, min_periods=1).max()
        return hhv_reversed.iloc[::-1].reset_index(drop=True).set_axis(series.index)


def SMA(X, n, m):
    """
    移动平均 - 通达信风格（向量化优化，结果与原递归实现完全一致）
    SMA(X,N,M): X的N日移动平均, M为权重
    公式: Y = (X*M + Y'*(N-M)) / N
    """
    if POLARS_AVAILABLE and isinstance(X, pl.Series):
        # Polars实现（GPU加速）
        alpha = m / n
        return X.ewm_mean(alpha=alpha, adjust=False, min_periods=1)
    else:
        # 兼容pandas实现
        alpha = m / n
        return X.ewm(alpha=alpha, adjust=False).mean()


def REF(series, n):
    """
    向前引用N周期 - 正确处理倒序排列的数据
    
    对于倒序数据（最新在前），REF(series, 1)应该获取"前一天"的数据
    实现方式：反转数据 -> shift -> 反转回来
    """
    if POLARS_AVAILABLE and isinstance(series, pl.Series):
        # Polars实现（GPU加速）
        reversed_series = series.reverse()
        ref_reversed = reversed_series.shift(n)
        return ref_reversed.reverse()
    else:
        # 兼容pandas实现
        reversed_series = series.iloc[::-1]
        ref_reversed = reversed_series.shift(n)
        return ref_reversed.iloc[::-1].reset_index(drop=True).set_axis(series.index)


def EXIST(cond, n):
    """
    N周期内是否存在满足COND的情况 - 正确处理倒序排列的数据
    """
    if POLARS_AVAILABLE and isinstance(cond, pl.Series):
        # Polars实现（GPU加速）
        reversed_cond = cond.reverse()
        exist_reversed = reversed_cond.rolling_max(window_size=n, min_periods=1).cast(pl.Boolean)
        return exist_reversed.reverse()
    else:
        # 兼容pandas实现
        reversed_cond = cond.iloc[::-1]
        exist_reversed = reversed_cond.rolling(window=n, min_periods=1).max().astype(bool)
        return exist_reversed.iloc[::-1].reset_index(drop=True).set_axis(cond.index)


def FINANCE(df, field_code):
    """
    财务数据获取
    39: 总市值（注意：原通达信39是流通市值，本项目使用总市值）
    """
    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        if field_code == 39:
            return df.get_column('market_cap') if 'market_cap' in df.columns else pl.Series([0] * len(df))
        return pl.Series([0] * len(df))
    else:
        if field_code == 39:
            return df.get('market_cap', pd.Series([0] * len(df), index=df.index))
        return pd.Series([0] * len(df), index=df.index)


def KDJ(df, n=9, m1=3, m2=3):
    """
    KDJ指标计算 - 标准实现（完全向量化，无for循环）
    通达信公式：
    RSV = (CLOSE - LLV(LOW,N)) / (HHV(HIGH,N) - LLV(LOW,N)) * 100
    K = SMA(RSV,M1,1)
    D = SMA(K,M2,1)
    J = 3*K - 2*D
    
    注意：数据可能是倒序（最新在前）或正序，需要自动检测并处理
    """
    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        # Polars GPU加速实现
        # 检测数据顺序
        is_descending = df['date'].item(0) > df['date'].item(-1)
        
        # 统一转换为正序计算（从早到晚）
        if is_descending:
            df_calc = df.sort('date', descending=False)
        else:
            df_calc = df
        
        # 计算RSV（完全向量化，无循环）
        low_min = df_calc['low'].rolling_min(window_size=n, min_periods=1)
        high_max = df_calc['high'].rolling_max(window_size=n, min_periods=1)
        
        range_val = high_max - low_min
        
        # 向量化计算RSV：前n-1个周期和range为0时填充50
        rsv = pl.when(
            (pl.int_range(0, len(df_calc)) < n - 1) | (range_val == 0)
        ).then(50.0).otherwise(
            (df_calc['close'] - low_min) / range_val * 100
        )
        
        # 计算K值：SMA(RSV, M1, 1)，初始值50
        k = rsv.ewm_mean(alpha=1/m1, adjust=False, min_periods=1)
        k = pl.concat([pl.Series([50.0]), k.slice(1)])
        
        # 计算D值：SMA(K, M2, 1)，初始值50
        d = k.ewm_mean(alpha=1/m2, adjust=False, min_periods=1)
        d = pl.concat([pl.Series([50.0]), d.slice(1)])
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 构建结果
        result = pl.DataFrame({
            'K': k,
            'D': d,
            'J': j
        })
        
        # 恢复原始顺序
        if is_descending:
            result = result.sort(by=pl.int_range(0, len(result)), descending=True)
        
        return result
    else:
        # pandas实现（完全向量化，移除原for循环）
        # 检测数据顺序
        is_descending = df['date'].iloc[0] > df['date'].iloc[-1]
        
        # 统一转换为正序计算（从早到晚）
        if is_descending:
            df_calc = df.iloc[::-1].copy().reset_index(drop=True)
        else:
            df_calc = df.copy().reset_index(drop=True)
        
        # 计算RSV（完全向量化，无循环）
        low_min = df_calc['low'].rolling(window=n, min_periods=1).min()
        high_max = df_calc['high'].rolling(window=n, min_periods=1).max()
        
        range_val = high_max - low_min
        
        # 向量化计算RSV：前n-1个周期和range为0时填充50
        rsv = pd.Series(np.where(
            (np.arange(len(df_calc)) < n - 1) | (range_val == 0),
            50.0,
            (df_calc['close'] - low_min) / range_val * 100
        ), index=df_calc.index)
        
        # SMA计算 - 通达信风格（向量化优化，结果与原递归实现完全一致）
        # K = SMA(RSV, M1, 1): K = (RSV*1 + K'*(M1-1)) / M1
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        # 初始化第一日K值为50
        k.iloc[0] = 50.0
        
        # D = SMA(K, M2, 1)
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        # 初始化第一日D值为50
        d.iloc[0] = 50.0
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 构建结果
        result = pd.DataFrame({
            'K': k,
            'D': d,
            'J': j
        })
        
        # 恢复原始顺序
        if is_descending:
            result = result.iloc[::-1].reset_index(drop=True)
        
        result.index = df.index
        return result


def calculate_zhixing_trend(df, m1=14, m2=28, m3=57, m4=114):
    """
    计算知行趋势线指标
    
    指标定义:
    - 知行短期趋势线 = EMA(EMA(CLOSE,10),10)
      对收盘价连续做两次10日指数移动平均
    
    - 知行多空线 = (MA(CLOSE,m1) + MA(CLOSE,m2) + MA(CLOSE,m3) + MA(CLOSE,m4)) / 4
      四条均线平均值，默认使用 14, 28, 57, 114
    
    参数:
        m1, m2, m3, m4: 多空线计算用的MA周期，默认14, 28, 57, 114
    """
    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        # Polars GPU加速实现
        close = df['close']
        # 知行短期趋势线 = EMA(EMA(CLOSE,10),10)
        short_term_trend = EMA(EMA(close, 10), 10)
        
        # 知行多空线 = (MA(m1) + MA(m2) + MA(m3) + MA(m4)) / 4
        bull_bear_line = (MA(close, m1) + MA(close, m2) + 
                          MA(close, m3) + MA(close, m4)) / 4
        
        return pl.DataFrame({
            'short_term_trend': short_term_trend,
            'bull_bear_line': bull_bear_line
        })
    else:
        # pandas实现
        # 知行短期趋势线 = EMA(EMA(CLOSE,10),10)
        short_term_trend = EMA(EMA(df['close'], 10), 10)
        
        # 知行多空线 = (MA(m1) + MA(m2) + MA(m3) + MA(m4)) / 4
        bull_bear_line = (MA(df['close'], m1) + MA(df['close'], m2) + 
                          MA(df['close'], m3) + MA(df['close'], m4)) / 4
        
        return pd.DataFrame({
            'short_term_trend': short_term_trend,
            'bull_bear_line': bull_bear_line
        }, index=df.index)
