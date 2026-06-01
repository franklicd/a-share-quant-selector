"""
砖型图计算模块 - 通达信砖型图公式 Python 实现

算法说明:
- 砖型图(Renko)只关注价格变动是否超过阈值，忽略时间和小幅波动
- 红砖(>0)=趋势向上超过砖高，绿砖(<0)=趋势向下超过砖高
- 红砖 ≠ 当天阳线：红砖意味着价格趋势突破砖高阈值，非单日涨跌

通达信公式:
1. HHV/LLV = N日滚动最高/最低
2. VAR2A = SMA((HHV-CLOSE)/(HHV-LLV)*100 - shift1, M1) + shift2  （下行指标）
3. VAR4A = SMA((CLOSE-LLV)/(HHV-LLV)*100, M2)                    （上行指标）
4. VAR5A = SMA(VAR4A, M3) + shift2                                （平滑上行指标）
5. RAW  = MAX(VAR5A - VAR2A - T, 0)                               （净上行-T）
6. BRICK = RAW[i] - RAW[i-1]                                      （砖差分）
"""
import numpy as np
import pandas as pd


def _compute_brick(high, low, close, n, m1, m2, m3, t,
                   shift1=90.0, shift2=100.0):
    """
    核心砖型图计算（纯 numpy）

    :param high/low/close: 价格数组，必须时间正序（旧→新）
    :param n: HHV/LLV 周期
    :param m1/m2/m3: SMA 周期
    :param t: 砖高阈值
    :return: brick 数组（正=红砖，负=绿砖）
    """
    length = len(close)

    # 1. HHV/LLV（pandas rolling 比手写循环快）
    hhv = pd.Series(high).rolling(window=n, min_periods=1).max().values.astype(np.float64)
    llv = pd.Series(low).rolling(window=n, min_periods=1).min().values.astype(np.float64)

    # 2. VAR2A = SMA((HHV-CLOSE)/(HHV-LLV)*100 - shift1, M1, 1) + shift2
    alpha1 = 1.0 / m1
    rng = hhv - llv
    rng[rng == 0.0] = 0.01
    v1 = (hhv - close) / rng * 100.0 - shift1
    var2a = np.empty(length, dtype=np.float64)
    var2a[0] = v1[0] + shift2
    for i in range(1, length):
        var2a[i] = alpha1 * v1[i] + (1 - alpha1) * (var2a[i - 1] - shift2) + shift2

    # 3. VAR4A = SMA((CLOSE-LLV)/(HHV-LLV)*100, M2, 1)
    alpha2 = 1.0 / m2
    v3 = (close - llv) / rng * 100.0
    var4a = np.empty(length, dtype=np.float64)
    var4a[0] = v3[0]
    for i in range(1, length):
        var4a[i] = alpha2 * v3[i] + (1 - alpha2) * var4a[i - 1]

    # 4. VAR5A = SMA(VAR4A, M3, 1) + shift2
    alpha3 = 1.0 / m3
    var5a = np.empty(length, dtype=np.float64)
    var5a[0] = v3[0] + shift2
    for i in range(1, length):
        var5a[i] = alpha3 * var4a[i] + (1 - alpha3) * (var5a[i - 1] - shift2) + shift2

    # 5. RAW = MAX(VAR5A - VAR2A - T, 0)
    raw = np.maximum(var5a - var2a - t, 0.0)

    # 6. BRICK = RAW[i] - RAW[i-1]
    brick = np.empty(length, dtype=np.float64)
    brick[0] = 0.0
    brick[1:] = raw[1:] - raw[:-1]

    return brick


def compute_brick_chart(df, n=4, m1=4, m2=6, m3=6, t=4.0,
                        shift1=90.0, shift2=100.0):
    """
    计算砖型图砖值（自动处理降序数据）

    :param df: 含 high/low/close/date 列的 DataFrame
    :return: pd.Series，正=红砖，负=绿砖
    """
    if df.empty or len(df) < 2:
        return pd.Series([0.0] * len(df), index=df.index)

    is_desc = df['date'].iloc[0] > df['date'].iloc[-1]
    df_calc = df.iloc[::-1] if is_desc else df

    brick = _compute_brick(
        df_calc['high'].to_numpy(dtype=np.float64),
        df_calc['low'].to_numpy(dtype=np.float64),
        df_calc['close'].to_numpy(dtype=np.float64),
        n, m1, m2, m3, t, shift1, shift2,
    )

    result = pd.Series(brick, index=df_calc.index, name='brick')
    return result.iloc[::-1] if is_desc else result


def check_brick_pattern(df, brick_values,
                        daily_return_threshold=0.05,
                        brick_growth_ratio=1.0,
                        min_prior_green_bars=1):
    """
    砖型图形态过滤（检查最新一天是否满足 5 个条件）

    5 个条件:
    1. 今日涨幅 < threshold（防追高）
    2. 今日红砖（brick > 0）
    3. 昨日绿砖（brick < 0）
    4. 红砖高度 >= ratio × 昨日绿砖绝对高度
    5. 前面连续绿砖数 >= min_prior

    :param df: 含 close/date 列的 DataFrame（可降序可升序）
    :param brick_values: 与 df 同 index 的砖值 Series
    :return: (passed: bool, info: dict)
    """
    is_desc = df['date'].iloc[0] > df['date'].iloc[-1]

    bv = brick_values.iloc[::-1].values if is_desc else brick_values.values
    cv = df['close'].iloc[::-1].values if is_desc else df['close'].values

    if len(bv) < 3:
        return False, {'reason': '数据不足（<3条）'}

    b0, b1 = float(bv[-1]), float(bv[-2])
    c0, c1 = float(cv[-1]), float(cv[-2])

    info = {
        'brick_today': round(b0, 4),
        'brick_yesterday': round(b1, 4),
        'return_today': round((c0 / c1 - 1.0) * 100, 2) if c1 > 0 else 0,
    }

    # 条件 1: 涨幅 < threshold
    if c1 <= 0 or (c0 / c1 - 1.0) >= daily_return_threshold:
        info['fail_reason'] = f'涨幅>={daily_return_threshold * 100:.0f}%'
        return False, info

    # 条件 2: 今日红砖
    if b0 <= 0:
        info['fail_reason'] = '今天非红砖'
        return False, info

    # 条件 3: 昨日绿砖
    if b1 >= 0:
        info['fail_reason'] = '昨天非绿砖'
        return False, info

    # 条件 4: 红砖力度
    if b0 < brick_growth_ratio * abs(b1):
        info['fail_reason'] = '红砖力度不足'
        return False, info

    # 条件 5: 前面连续绿砖数
    green_count = 0
    for i in range(len(bv) - 3, -1, -1):
        if bv[i] < 0:
            green_count += 1
        else:
            break

    if green_count < min_prior_green_bars:
        info['fail_reason'] = f'连续绿砖{green_count}根，需{min_prior_green_bars}根'
        return False, info

    info['green_bars_before'] = green_count
    info['brick_growth'] = round(b0 / abs(b1), 2) if abs(b1) > 0 else 0
    return True, info


def compute_weekly_ma_bull(df, ma_periods=(20, 60, 120)):
    """
    周线均线多头排列（MA_short > MA_mid > MA_long），forward-fill 到日线

    :param df: 日线 DataFrame（需有 date 列，可降序）
    :return: pd.Series[bool]（与 df 同 index）
    """
    is_desc = df['date'].iloc[0] > df['date'].iloc[-1]
    df_sorted = df.iloc[::-1] if is_desc else df

    dates = pd.to_datetime(df_sorted['date'])
    close = df_sorted['close'].values

    # 按 ISO 周分组取最后交易日收盘价
    iso = dates.dt.isocalendar()
    year_week = iso['year'].astype(str) + '-' + iso['week'].astype(str).str.zfill(2)

    weekly_dates = []
    weekly_closes = []
    last_yw = None
    for yw, d, c in zip(year_week, dates, close):
        if yw != last_yw:
            if last_yw is not None:
                weekly_dates.append(last_date)
                weekly_closes.append(last_close)
            last_yw = yw
        last_date = d
        last_close = c
    if last_yw is not None:
        weekly_dates.append(last_date)
        weekly_closes.append(last_close)

    s, m, l = ma_periods
    if len(weekly_closes) < l:
        return pd.Series([False] * len(df), index=df.index)

    wc = pd.Series(weekly_closes, index=pd.DatetimeIndex(weekly_dates))
    ma_s = wc.rolling(s, min_periods=s).mean()
    ma_m = wc.rolling(m, min_periods=m).mean()
    ma_l = wc.rolling(l, min_periods=l).mean()
    bull = (ma_s > ma_m) & (ma_m > ma_l)

    daily_index = pd.DatetimeIndex(dates)
    bull_daily = (bull.astype(float)
                  .reindex(daily_index)
                  .ffill()
                  .fillna(0.0)
                  .astype(bool))

    result = pd.Series(bull_daily.values, index=df_sorted.index)
    return result.iloc[::-1] if is_desc else result
