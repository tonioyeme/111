"""
Gamma策略增强分析框架
新增维度：
1. Move Ratio - 基于Implied High/Low的波动强度
2. Zone分类 - 基于所有Key Levels的位置分类
3. 距离最近Level - 接近关键位的影响
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings
import yaml
import datetime

warnings.filterwarnings('ignore')

# ============ 配置 ============
CONFIG = {
    'data_dir': 'data/spx',
    'key_levels_dir': 'data/key_levels',
    'exclude_first_n_minutes': 30,
    'min_samples': 10,
}


# ============ 数据加载 ============
def load_key_levels(config):
    """加载所有日期的key levels"""
    levels_dir = Path(config['key_levels_dir'])
    all_levels = {}

    for yaml_file in levels_dir.glob('*.yaml'):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        date_str = data.get('date', yaml_file.stem)
        date = pd.to_datetime(date_str).date()

        # 解析implied_move百分比
        implied_move = data.get('implied_move', '0%')
        if isinstance(implied_move, str):
            implied_move = float(implied_move.replace('%', '')) / 100

        all_levels[date] = {
            'call_wall': data.get('call_wall'),
            'put_wall': data.get('put_wall'),
            'hedge_wall': data.get('hedge_wall'),
            'implied_high': data.get('implied_high'),
            'implied_low': data.get('implied_low'),
            'implied_move': implied_move,
            'resistance': data.get('resistance', []),
            'support': data.get('support', []),
            'pivot': data.get('pivot', []),
        }

    print(f"加载 {len(all_levels)} 天的 key levels 数据")
    return all_levels


def load_data(config, key_levels):
    """加载并预处理SPX数据，合并key levels"""
    data_dir = Path(config['data_dir'])
    all_files = sorted(data_dir.glob('SPX_*.csv'))

    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中找不到SPX数据文件")

    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)

    # 处理时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace(r'-\d{2}:\d{2}$', '', regex=True))
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 提取日期和时间信息
    df['date'] = df['timestamp'].dt.date
    df['minutes_from_open'] = (df['timestamp'].dt.hour - 9) * 60 + df['timestamp'].dt.minute - 30

    # 合并key levels
    for col in ['call_wall', 'put_wall', 'hedge_wall', 'implied_high', 'implied_low', 'implied_move']:
        df[col] = df['date'].map(lambda d: key_levels.get(d, {}).get(col))

    # 存储resistance和support列表（用于后续计算）
    df['resistance_levels'] = df['date'].map(lambda d: key_levels.get(d, {}).get('resistance', []))
    df['support_levels'] = df['date'].map(lambda d: key_levels.get(d, {}).get('support', []))

    # 过滤有key levels的数据
    df = df[df['hedge_wall'].notna()].copy()

    # 计算Gamma环境
    df['gamma'] = np.where(df['close'] > df['hedge_wall'], 'positive', 'negative')

    # 计算每日开盘价（用于Move Ratio）
    daily_open = df.groupby('date')['open'].first()
    df['daily_open'] = df['date'].map(daily_open)

    print(f"加载数据: {len(df)} 行, {df['date'].nunique()} 天")
    print(f"Gamma分布: positive={len(df[df['gamma']=='positive'])}, negative={len(df[df['gamma']=='negative'])}")

    return df


# ============ Move Ratio 计算 ============
def calculate_move_ratio(df):
    """
    计算Move Ratio

    优先使用 implied_high/low:
        涨时: (price - daily_open) / (implied_high - daily_open)
        跌时: (daily_open - price) / (daily_open - implied_low)

    当 implied_high/low 不可用时，使用 implied_move:
        implied_high = daily_open * (1 + implied_move)
        implied_low = daily_open * (1 - implied_move)
    """
    df = df.copy()

    # 当前价格相对开盘的方向
    df['price_vs_open'] = df['close'] - df['daily_open']

    # 计算 effective implied_high 和 implied_low
    # 优先使用 YAML 中的 implied_high/low，否则用 implied_move 计算
    df['eff_implied_high'] = df['implied_high'].where(
        df['implied_high'].notna(),
        df['daily_open'] * (1 + df['implied_move'])
    )
    df['eff_implied_low'] = df['implied_low'].where(
        df['implied_low'].notna(),
        df['daily_open'] * (1 - df['implied_move'])
    )

    # 计算Move Ratio
    # 涨: 向implied_high方向
    up_range = df['eff_implied_high'] - df['daily_open']
    up_move = df['close'] - df['daily_open']

    # 跌: 向implied_low方向
    down_range = df['daily_open'] - df['eff_implied_low']
    down_move = df['daily_open'] - df['close']

    # 根据方向选择
    df['move_ratio'] = np.where(
        df['price_vs_open'] >= 0,
        np.where(up_range > 0, up_move / up_range, 0),
        np.where(down_range > 0, down_move / down_range, 0)
    )

    # 标记方向
    df['move_direction'] = np.where(df['price_vs_open'] >= 0, 'up', 'down')

    # 打印使用情况统计
    has_implied_hl = df['implied_high'].notna().sum()
    used_implied_move = df['implied_high'].isna().sum()
    print(f"  Move Ratio计算: {has_implied_hl}行使用implied_high/low, {used_implied_move}行使用implied_move")

    return df


# ============ Zone 分类 ============
def classify_zone(df):
    """
    基于Key Levels的Zone分类（适应不同日期levels顺序可能不同的情况）

    使用独立的二元分类，而非固定顺序的Zone：
    - gamma: positive/negative (基于hedge_wall)
    - vs_implied: in_range/above/below (相对于implied high/low)
    - vs_walls: normal/above_call/below_put (相对于call/put wall)
    """
    df = df.copy()

    # 1. Gamma环境（已有）
    # df['gamma'] 已在load_data中计算

    # 2. 相对于Implied Range的位置
    # 使用 effective implied_high/low (已在calculate_move_ratio中计算)
    df['vs_implied'] = np.select(
        [
            df['close'] > df['eff_implied_high'],
            df['close'] < df['eff_implied_low'],
        ],
        ['above_implied', 'below_implied'],
        default='in_implied_range'
    )

    # 3. 相对于Walls的位置
    df['vs_walls'] = np.select(
        [
            df['close'] > df['call_wall'],
            df['close'] < df['put_wall'],
        ],
        ['above_call_wall', 'below_put_wall'],
        default='between_walls'
    )

    # 4. 组合Zone（Gamma × Implied位置）
    df['zone'] = df['gamma'] + '_' + df['vs_implied']

    # 简化Zone名称
    zone_map = {
        'positive_above_implied': 'G+_above_implied',
        'positive_in_implied_range': 'G+_in_range',
        'positive_below_implied': 'G+_below_implied',
        'negative_above_implied': 'G-_above_implied',
        'negative_in_implied_range': 'G-_in_range',
        'negative_below_implied': 'G-_below_implied',
    }
    df['zone'] = df['zone'].map(zone_map).fillna('unknown')

    return df


# ============ 距离Level计算 ============
def calculate_all_level_distances(df):
    """计算距离所有关键level的距离"""
    df = df.copy()

    def get_all_distances(row):
        price = row['close']

        # 距离各个固定level的距离（百分比，正值=level在价格上方，负值=level在价格下方）
        dist_hedge_wall = (row['hedge_wall'] - price) / price if row['hedge_wall'] else np.nan
        dist_call_wall = (row['call_wall'] - price) / price if row['call_wall'] else np.nan
        dist_put_wall = (row['put_wall'] - price) / price if row['put_wall'] else np.nan
        dist_implied_high = (row['implied_high'] - price) / price if row['implied_high'] else np.nan
        dist_implied_low = (row['implied_low'] - price) / price if row['implied_low'] else np.nan

        # 收集所有阻力位（价格上方）
        resistances = []
        if row['call_wall'] and row['call_wall'] > price:
            resistances.append(row['call_wall'])
        if row['implied_high'] and row['implied_high'] > price:
            resistances.append(row['implied_high'])
        if row['hedge_wall'] and row['hedge_wall'] > price:
            resistances.append(row['hedge_wall'])
        if isinstance(row['resistance_levels'], list):
            resistances.extend([r for r in row['resistance_levels'] if r and r > price])

        # 收集所有支撑位（价格下方）
        supports = []
        if row['put_wall'] and row['put_wall'] < price:
            supports.append(row['put_wall'])
        if row['implied_low'] and row['implied_low'] < price:
            supports.append(row['implied_low'])
        if row['hedge_wall'] and row['hedge_wall'] < price:
            supports.append(row['hedge_wall'])
        if isinstance(row['support_levels'], list):
            supports.extend([s for s in row['support_levels'] if s and s < price])

        # 最近阻力和支撑
        dist_to_nearest_resistance = min([(r - price) / price for r in resistances]) if resistances else np.nan
        dist_to_nearest_support = min([(price - s) / price for s in supports]) if supports else np.nan

        nearest_resistance = min(resistances) if resistances else np.nan
        nearest_support = max(supports) if supports else np.nan

        return pd.Series({
            'dist_hedge_wall': dist_hedge_wall,
            'dist_call_wall': dist_call_wall,
            'dist_put_wall': dist_put_wall,
            'dist_implied_high': dist_implied_high,
            'dist_implied_low': dist_implied_low,
            'dist_to_resistance': dist_to_nearest_resistance,
            'dist_to_support': dist_to_nearest_support,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
        })

    distances = df.apply(get_all_distances, axis=1)
    df = pd.concat([df, distances], axis=1)

    return df


# ============ 信号识别（增强版）============
def find_signals_enhanced(df, lookback, threshold, gamma_type, direction, config):
    """
    找出符合条件的信号，附带Move Ratio和Zone信息
    """
    df = df.copy()

    # 计算过去收益
    df['past_return'] = df['close'].pct_change(lookback)

    # 基础过滤
    mask = (
        (df['minutes_from_open'] >= config['exclude_first_n_minutes']) &
        (df['gamma'] == gamma_type)
    )

    # 方向过滤
    if direction == 'after_drop':
        mask = mask & (df['past_return'] < -threshold)
    else:
        mask = mask & (df['past_return'] > threshold)

    signals = df[mask].copy()
    signals['signal_idx'] = signals.index

    return signals


# ============ MFE/MAE 分析 ============
def analyze_signals_mfe_mae(df, signals, horizon, is_short=True):
    """分析每个信号的MFE和MAE"""
    results = []

    for idx in signals['signal_idx']:
        if idx + horizon >= len(df):
            continue

        entry_price = df.loc[idx, 'close']
        future_slice = df.loc[idx+1 : idx+horizon]

        if len(future_slice) < horizon:
            continue

        future_highs = future_slice['high'].values
        future_lows = future_slice['low'].values
        future_closes = future_slice['close'].values

        if is_short:
            mfe = (entry_price - future_lows.min()) / entry_price
            mae = (entry_price - future_highs.max()) / entry_price
            final_return = (entry_price - future_closes[-1]) / entry_price
            return_series = (entry_price - future_closes) / entry_price
        else:
            mfe = (future_highs.max() - entry_price) / entry_price
            mae = (future_lows.min() - entry_price) / entry_price
            final_return = (future_closes[-1] - entry_price) / entry_price
            return_series = (future_closes - entry_price) / entry_price

        # 获取信号时的附加信息
        signal_row = signals[signals['signal_idx'] == idx].iloc[0]

        results.append({
            'signal_idx': idx,
            'entry_price': entry_price,
            'final_return': final_return,
            'mfe': mfe,
            'mae': mae,
            'return_series': return_series,
            # 基础字段
            'move_ratio': signal_row.get('move_ratio', np.nan),
            'zone': signal_row.get('zone', 'unknown'),
            'zone_num': signal_row.get('zone_num', -1),
            # 距离Level字段
            'dist_to_support': signal_row.get('dist_to_support', np.nan),
            'dist_to_resistance': signal_row.get('dist_to_resistance', np.nan),
            'dist_hedge_wall': signal_row.get('dist_hedge_wall', np.nan),
            'dist_call_wall': signal_row.get('dist_call_wall', np.nan),
            'dist_put_wall': signal_row.get('dist_put_wall', np.nan),
            'dist_implied_high': signal_row.get('dist_implied_high', np.nan),
            'dist_implied_low': signal_row.get('dist_implied_low', np.nan),
        })

    return pd.DataFrame(results)


# ============ 固定止盈止损回测 ============
def backtest_fixed_tpsl(results, tp, sl):
    """
    固定止盈止损回测

    参数:
        results: analyze_signals_mfe_mae的输出
        tp: 止盈阈值（正值，如0.002表示0.2%）
        sl: 止损阈值（正值，如0.003表示0.3%）

    返回:
        dict with backtest stats
    """
    if len(results) == 0:
        return None

    tp_count = 0
    sl_count = 0
    timeout_returns = []

    for _, row in results.iterrows():
        series = row['return_series']

        hit_tp = False
        hit_sl = False

        for ret in series:
            if ret >= tp and not hit_tp and not hit_sl:
                hit_tp = True
                tp_count += 1
                break
            if ret <= -sl and not hit_sl and not hit_tp:
                hit_sl = True
                sl_count += 1
                break

        if not hit_tp and not hit_sl:
            timeout_returns.append(series[-1] if len(series) > 0 else 0)

    n = len(results)
    p_tp = tp_count / n
    p_sl = sl_count / n
    p_timeout = len(timeout_returns) / n
    avg_timeout = np.mean(timeout_returns) if timeout_returns else 0

    expected = p_tp * tp - p_sl * sl + p_timeout * avg_timeout

    return {
        'tp': tp,
        'sl': sl,
        'p_tp': p_tp,
        'p_sl': p_sl,
        'p_timeout': p_timeout,
        'avg_timeout': avg_timeout,
        'expected': expected,
    }


def find_optimal_tpsl(results, signal_name=""):
    """
    对给定的信号结果进行止盈止损网格测试，找出最优参数

    返回:
        dict with optimal TP/SL and grid results
    """
    if len(results) < 10:
        return None

    # 止盈止损网格
    tp_range = [0.001, 0.0015, 0.002, 0.0025, 0.003]  # 0.1% - 0.3%
    sl_range = [0.002, 0.0025, 0.003, 0.0035, 0.004]  # 0.2% - 0.4%

    best_expected = -999
    best_params = None
    grid_results = []

    for tp in tp_range:
        for sl in sl_range:
            bt = backtest_fixed_tpsl(results, tp, sl)
            if bt:
                grid_results.append(bt)
                if bt['expected'] > best_expected:
                    best_expected = bt['expected']
                    best_params = bt

    # 基于MFE/MAE推荐的参数
    mfe_p50 = results['mfe'].median()
    mae_p10 = results['mae'].quantile(0.1)  # 90%不会触发
    mae_p25 = results['mae'].quantile(0.25)  # 75%不会触发

    return {
        'signal_name': signal_name,
        'n_samples': len(results),
        'best_tp': best_params['tp'] if best_params else None,
        'best_sl': best_params['sl'] if best_params else None,
        'best_expected': best_expected,
        'best_p_tp': best_params['p_tp'] if best_params else None,
        'best_p_sl': best_params['p_sl'] if best_params else None,
        'mfe_p50': mfe_p50,
        'mae_p10': mae_p10,
        'mae_p25': mae_p25,
        'recommended_tp': mfe_p50,  # 基于MFE P50
        'recommended_sl': -mae_p25,  # 基于MAE P25 (75%覆盖)
        'grid_results': grid_results,
    }


def print_tpsl_analysis(optimal_result):
    """打印止盈止损分析结果"""
    if optimal_result is None:
        print("  样本不足，无法进行止盈止损分析")
        return

    print(f"\n【止盈止损分析】 - {optimal_result['signal_name']}")
    print(f"  样本数: {optimal_result['n_samples']}")
    print(f"\n  MFE/MAE基础数据:")
    print(f"    MFE P50: {optimal_result['mfe_p50']*100:.3f}%")
    print(f"    MAE P10 (90%覆盖): {optimal_result['mae_p10']*100:.3f}%")
    print(f"    MAE P25 (75%覆盖): {optimal_result['mae_p25']*100:.3f}%")
    print(f"\n  推荐参数 (基于MFE/MAE分布):")
    print(f"    止盈 (MFE P50): {optimal_result['recommended_tp']*100:.2f}%")
    print(f"    止损 (MAE P75): {optimal_result['recommended_sl']*100:.2f}%")

    if optimal_result['best_tp']:
        print(f"\n  网格测试最优参数:")
        print(f"    止盈: {optimal_result['best_tp']*100:.2f}%")
        print(f"    止损: {optimal_result['best_sl']*100:.2f}%")
        print(f"    期望收益: {optimal_result['best_expected']*100:.4f}%")
        print(f"    触发止盈概率: {optimal_result['best_p_tp']*100:.1f}%")
        print(f"    触发止损概率: {optimal_result['best_p_sl']*100:.1f}%")

    # 打印网格
    if optimal_result['grid_results']:
        print(f"\n  止盈止损网格测试:")
        print(f"  {'TP':>6} | {'SL':>6} | {'触发TP':>7} | {'触发SL':>7} | {'超时':>6} | {'期望收益':>10}")
        print("  " + "-" * 60)
        for bt in optimal_result['grid_results']:
            marker = " *" if bt['tp'] == optimal_result['best_tp'] and bt['sl'] == optimal_result['best_sl'] else ""
            print(f"  {bt['tp']*100:>5.2f}% | {bt['sl']*100:>5.2f}% | {bt['p_tp']*100:>6.1f}% | "
                  f"{bt['p_sl']*100:>6.1f}% | {bt['p_timeout']*100:>5.1f}% | {bt['expected']*100:>9.4f}%{marker}")


# ============ 分层分析函数 ============
def print_detailed_stats(subset, label):
    """打印详细的MFE/MAE统计"""
    if len(subset) < 5:
        return False

    mfe_p25 = subset['mfe'].quantile(0.25)
    mfe_p50 = subset['mfe'].median()
    mfe_p75 = subset['mfe'].quantile(0.75)

    mae_p25 = subset['mae'].quantile(0.25)  # 最差的25%
    mae_p50 = subset['mae'].median()
    mae_p75 = subset['mae'].quantile(0.75)

    # MFE阈值统计
    pct_01 = (subset['mfe'] > 0.001).mean()
    pct_02 = (subset['mfe'] > 0.002).mean()
    pct_03 = (subset['mfe'] > 0.003).mean()

    # MAE阈值统计（多少比例亏损超过X%）
    mae_01 = (subset['mae'] < -0.001).mean()
    mae_02 = (subset['mae'] < -0.002).mean()
    mae_03 = (subset['mae'] < -0.003).mean()

    # MFE/MAE比率
    mfe_mae_ratio = mfe_p50 / abs(mae_p50) if mae_p50 != 0 else np.inf

    print(f"{label}")
    print(f"  样本数: {len(subset)}")
    print(f"  MFE: P25={mfe_p25*100:.3f}%, P50={mfe_p50*100:.3f}%, P75={mfe_p75*100:.3f}%")
    print(f"  MAE: P25={mae_p25*100:.3f}%, P50={mae_p50*100:.3f}%, P75={mae_p75*100:.3f}%")
    print(f"  MFE/MAE比: {mfe_mae_ratio:.2f}")
    print(f"  盈利>0.1%: {pct_01*100:.1f}%, >0.2%: {pct_02*100:.1f}%, >0.3%: {pct_03*100:.1f}%")
    print(f"  亏损>0.1%: {mae_01*100:.1f}%, >0.2%: {mae_02*100:.1f}%, >0.3%: {mae_03*100:.1f}%")
    print()
    return True


def analyze_by_move_ratio(results, thresholds=[0.3, 0.5, 0.7, 1.0]):
    """按Move Ratio分层分析"""
    if len(results) == 0:
        return

    print(f"\n{'='*60}")
    print(f"【Move Ratio 分层分析】")
    print(f"{'='*60}")

    bins = [0] + thresholds + [float('inf')]
    labels = [f'<{thresholds[0]*100:.0f}%'] + \
             [f'{thresholds[i]*100:.0f}-{thresholds[i+1]*100:.0f}%' for i in range(len(thresholds)-1)] + \
             [f'>{thresholds[-1]*100:.0f}%']

    results['move_ratio_bin'] = pd.cut(results['move_ratio'], bins=bins, labels=labels)

    for label in labels:
        subset = results[results['move_ratio_bin'] == label]
        print_detailed_stats(subset, f"Move Ratio {label}")


def analyze_by_zone(results):
    """按Zone分层分析"""
    if len(results) == 0:
        return

    print(f"\n{'='*60}")
    print(f"【Zone 分层分析】")
    print(f"{'='*60}")

    for zone in sorted(results['zone'].unique()):
        subset = results[results['zone'] == zone]
        print_detailed_stats(subset, f"Zone: {zone}")


def analyze_by_distance_to_support(results, thresholds=[0.002, 0.005, 0.01]):
    """按距离Support分层分析"""
    if len(results) == 0:
        return

    # 过滤有效数据
    valid = results[results['dist_to_support'].notna()].copy()
    if len(valid) < 10:
        print("\n【距离Support分析】样本不足")
        return

    print(f"\n{'='*60}")
    print(f"【距离 Support 分层分析】")
    print(f"{'='*60}")

    bins = [0] + thresholds + [float('inf')]
    labels = [f'<{thresholds[0]*100:.1f}%'] + \
             [f'{thresholds[i]*100:.1f}-{thresholds[i+1]*100:.1f}%' for i in range(len(thresholds)-1)] + \
             [f'>{thresholds[-1]*100:.1f}%']

    valid['dist_bin'] = pd.cut(valid['dist_to_support'], bins=bins, labels=labels)

    for label in labels:
        subset = valid[valid['dist_bin'] == label]
        print_detailed_stats(subset, f"距离Support {label}")


# ============ 交叉分析 ============
def cross_analysis_zone_move_ratio(results):
    """Zone × Move Ratio 交叉分析"""
    if len(results) == 0:
        return

    print(f"\n{'='*60}")
    print(f"【Zone × Move Ratio 交叉分析】")
    print(f"{'='*60}")

    # 简化Move Ratio分组
    results['mr_group'] = pd.cut(
        results['move_ratio'],
        bins=[0, 0.5, 1.0, float('inf')],
        labels=['<50%', '50-100%', '>100%']
    )

    for zone in sorted(results['zone'].unique()):
        for mr in ['<50%', '50-100%', '>100%']:
            subset = results[(results['zone'] == zone) & (results['mr_group'] == mr)]
            print_detailed_stats(subset, f"{zone} + MoveRatio {mr}")


# ============ Key Levels 距离影响分析 ============
def analyze_level_distance_impact(results, is_short=True):
    """
    分析各个Key Level距离对交易结果的影响
    验证核心假设的统计显著性
    """
    if len(results) < 20:
        print("\n【Level距离影响分析】样本不足")
        return

    print(f"\n{'='*80}")
    print(f"【Key Levels 距离影响分析】")
    print(f"{'='*80}")

    # 定义要分析的距离字段
    distance_fields = [
        ('dist_to_support', '距最近Support'),
        ('dist_to_resistance', '距最近Resistance'),
        ('dist_hedge_wall', '距Hedge Wall'),
        ('dist_implied_high', '距Implied High'),
        ('dist_implied_low', '距Implied Low'),
    ]

    # 分层阈值
    thresholds = [0.002, 0.005, 0.01]  # 0.2%, 0.5%, 1.0%

    for field, name in distance_fields:
        if field not in results.columns:
            continue

        valid = results[results[field].notna()].copy()
        if len(valid) < 20:
            continue

        # 取绝对值（对于hedge wall等可能为负的）
        if field in ['dist_hedge_wall', 'dist_implied_high', 'dist_implied_low']:
            valid['abs_dist'] = valid[field].abs()
        else:
            valid['abs_dist'] = valid[field]

        print(f"\n--- {name} ---")

        # 分层分析
        bins = [0] + thresholds + [float('inf')]
        labels = [f'<{thresholds[0]*100:.1f}%'] + \
                 [f'{thresholds[i]*100:.1f}-{thresholds[i+1]*100:.1f}%' for i in range(len(thresholds)-1)] + \
                 [f'>{thresholds[-1]*100:.1f}%']

        valid['dist_bin'] = pd.cut(valid['abs_dist'], bins=bins, labels=labels)

        print(f"{'距离':>12} | {'N':>4} | {'MFE_P50':>8} | {'MAE_P50':>8} | {'MFE/MAE':>7} | {'>0.2%':>6} | {'亏>0.2%':>7}")
        print("-" * 75)

        bin_stats = []
        for label in labels:
            subset = valid[valid['dist_bin'] == label]
            if len(subset) < 5:
                continue

            mfe_p50 = subset['mfe'].median()
            mae_p50 = subset['mae'].median()
            mfe_mae = mfe_p50 / abs(mae_p50) if mae_p50 != 0 else np.inf
            pct_02 = (subset['mfe'] > 0.002).mean()
            mae_02 = (subset['mae'] < -0.002).mean()

            print(f"{label:>12} | {len(subset):>4} | {mfe_p50*100:>7.3f}% | {mae_p50*100:>7.3f}% | "
                  f"{mfe_mae:>7.2f} | {pct_02*100:>5.1f}% | {mae_02*100:>6.1f}%")

            bin_stats.append({
                'label': label,
                'n': len(subset),
                'mfe_p50': mfe_p50,
                'mae_p50': mae_p50,
                'mfe_mae': mfe_mae,
            })

        # 简单对比：最近 vs 最远
        if len(bin_stats) >= 2:
            near = valid[valid['dist_bin'] == labels[0]]
            far_label = labels[-1] if len(valid[valid['dist_bin'] == labels[-1]]) >= 5 else labels[-2]
            far = valid[valid['dist_bin'] == far_label]

            if len(near) >= 5 and len(far) >= 5:
                near_mfe = near['mfe'].median()
                far_mfe = far['mfe'].median()
                diff = (far_mfe - near_mfe) * 100
                direction = "远>近" if diff > 0 else "近>远"
                print(f"  对比: 近({labels[0]}) MFE={near_mfe*100:.3f}% vs 远({far_label}) MFE={far_mfe*100:.3f}% → {direction} {abs(diff):.3f}%")


def analyze_hypothesis_tests(results, is_short=True):
    """
    验证核心假设（简化版，不依赖scipy）
    """
    if len(results) < 30:
        print("\n【假设检验】样本不足")
        return

    print(f"\n{'='*80}")
    print(f"【核心假设检验】")
    print(f"{'='*80}")

    hypotheses = []

    # 假设1: 做空时，距Support越近，MFE越低（支撑会反弹）
    if is_short and 'dist_to_support' in results.columns:
        valid = results[results['dist_to_support'].notna()]
        if len(valid) >= 20:
            near = valid[valid['dist_to_support'] < 0.003]  # <0.3%
            far = valid[valid['dist_to_support'] >= 0.005]  # >=0.5%

            if len(near) >= 10 and len(far) >= 10:
                near_mfe = near['mfe'].median()
                far_mfe = far['mfe'].median()
                near_mae = near['mae'].median()
                far_mae = far['mae'].median()
                diff_pct = (far_mfe - near_mfe) / near_mfe * 100 if near_mfe != 0 else 0
                confirmed = far_mfe > near_mfe  # 远离support时MFE更高
                hypotheses.append({
                    'hypothesis': '做空: 距Support近 → MFE更低',
                    'near_mfe': near_mfe,
                    'far_mfe': far_mfe,
                    'near_mae': near_mae,
                    'far_mae': far_mae,
                    'n_near': len(near),
                    'n_far': len(far),
                    'diff_pct': diff_pct,
                    'confirmed': confirmed,
                })

    # 假设2: 距Hedge Wall越远，Gamma效应越强（MFE越高）
    if 'dist_hedge_wall' in results.columns:
        valid = results[results['dist_hedge_wall'].notna()].copy()
        valid['abs_dist_hw'] = valid['dist_hedge_wall'].abs()

        if len(valid) >= 20:
            near = valid[valid['abs_dist_hw'] < 0.003]
            far = valid[valid['abs_dist_hw'] >= 0.005]

            if len(near) >= 10 and len(far) >= 10:
                near_mfe = near['mfe'].median()
                far_mfe = far['mfe'].median()
                near_mae = near['mae'].median()
                far_mae = far['mae'].median()
                diff_pct = (far_mfe - near_mfe) / near_mfe * 100 if near_mfe != 0 else 0
                confirmed = far_mfe > near_mfe
                hypotheses.append({
                    'hypothesis': '距Hedge Wall远 → MFE更高',
                    'near_mfe': near_mfe,
                    'far_mfe': far_mfe,
                    'near_mae': near_mae,
                    'far_mae': far_mae,
                    'n_near': len(near),
                    'n_far': len(far),
                    'diff_pct': diff_pct,
                    'confirmed': confirmed,
                })

    # 假设3: 做空时，距上方Resistance越近，空间越小
    if is_short and 'dist_to_resistance' in results.columns:
        valid = results[results['dist_to_resistance'].notna()]
        if len(valid) >= 20:
            near = valid[valid['dist_to_resistance'] < 0.003]
            far = valid[valid['dist_to_resistance'] >= 0.005]

            if len(near) >= 10 and len(far) >= 10:
                near_mfe = near['mfe'].median()
                far_mfe = far['mfe'].median()
                near_mae = near['mae'].median()
                far_mae = far['mae'].median()
                diff_pct = (far_mfe - near_mfe) / near_mfe * 100 if near_mfe != 0 else 0
                # 假设：距阻力近时空间被压缩，MFE应该更低
                # 但实际上对于做空，距阻力近意味着价格高，下跌空间反而大
                confirmed = near_mfe < far_mfe
                hypotheses.append({
                    'hypothesis': '做空: 距Resistance近 → MFE差异',
                    'near_mfe': near_mfe,
                    'far_mfe': far_mfe,
                    'near_mae': near_mae,
                    'far_mae': far_mae,
                    'n_near': len(near),
                    'n_far': len(far),
                    'diff_pct': diff_pct,
                    'confirmed': confirmed,
                })

    # 打印结果
    if hypotheses:
        print(f"\n{'假设':<35} | {'近MFE':>7} | {'远MFE':>7} | {'差异%':>7} | {'N近':>4} | {'N远':>4} | {'结论':>10}")
        print("-" * 95)
        for h in hypotheses:
            sig = "✓符合预期" if h['confirmed'] else "✗相反"
            print(f"{h['hypothesis']:<35} | {h['near_mfe']*100:>6.3f}% | {h['far_mfe']*100:>6.3f}% | "
                  f"{h['diff_pct']:>+6.1f}% | {h['n_near']:>4} | {h['n_far']:>4} | {sig:>10}")

        # 补充MAE对比
        print(f"\n{'假设':<35} | {'近MAE':>7} | {'远MAE':>7}")
        print("-" * 60)
        for h in hypotheses:
            print(f"{h['hypothesis']:<35} | {h['near_mae']*100:>6.3f}% | {h['far_mae']*100:>6.3f}%")
    else:
        print("样本不足，无法进行假设检验")


# ============ 参数扫描汇总 ============
def parameter_scan_summary(df, config, signal_type='gamma_neg_short'):
    """
    参数扫描，返回汇总表用于快速对比
    """
    # 定义参数范围
    # threshold相对implied_move(~0.63%)更合理的范围: 0.05%-0.2%
    lookbacks = [5, 10, 15, 30]
    horizons = [5, 10, 15, 30]
    thresholds = [0.0005, 0.001, 0.0015, 0.002]  # 0.05%, 0.1%, 0.15%, 0.2%

    # 解析信号类型
    if signal_type == 'gamma_neg_short':
        gamma_type, direction, is_short = 'negative', 'after_drop', True
        signal_name = "Gamma- 跌后做空"
    elif signal_type == 'gamma_neg_long':
        gamma_type, direction, is_short = 'negative', 'after_rise', False
        signal_name = "Gamma- 涨后做多"
    elif signal_type == 'gamma_pos_short':
        gamma_type, direction, is_short = 'positive', 'after_rise', True
        signal_name = "Gamma+ 涨后做空"
    elif signal_type == 'gamma_pos_long':
        gamma_type, direction, is_short = 'positive', 'after_drop', False
        signal_name = "Gamma+ 跌后做多"
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    print(f"\n{'='*100}")
    print(f"参数扫描: {signal_name}")
    print(f"{'='*100}")

    summary = []

    for lb in lookbacks:
        for hz in horizons:
            for th in thresholds:
                # 找信号
                signals = find_signals_enhanced(df, lb, th, gamma_type, direction, config)

                if len(signals) < config['min_samples']:
                    continue

                # 分析MFE/MAE
                results = analyze_signals_mfe_mae(df, signals, hz, is_short)

                if len(results) < config['min_samples']:
                    continue

                # 计算统计
                mfe_p25 = results['mfe'].quantile(0.25)
                mfe_p50 = results['mfe'].median()
                mfe_p75 = results['mfe'].quantile(0.75)

                mae_p25 = results['mae'].quantile(0.25)
                mae_p50 = results['mae'].median()

                mfe_mae_ratio = mfe_p50 / abs(mae_p50) if mae_p50 != 0 else np.inf

                pct_01 = (results['mfe'] > 0.001).mean()
                pct_02 = (results['mfe'] > 0.002).mean()
                pct_03 = (results['mfe'] > 0.003).mean()

                mae_02 = (results['mae'] < -0.002).mean()
                mae_03 = (results['mae'] < -0.003).mean()

                summary.append({
                    'lb': lb,
                    'hz': hz,
                    'th': f"{th*100:.1f}%",
                    'n': len(results),
                    'mfe_p25': mfe_p25,
                    'mfe_p50': mfe_p50,
                    'mfe_p75': mfe_p75,
                    'mae_p50': mae_p50,
                    'mfe_mae': mfe_mae_ratio,
                    'pct_01': pct_01,
                    'pct_02': pct_02,
                    'pct_03': pct_03,
                    'mae_02': mae_02,
                    'mae_03': mae_03,
                })

    summary_df = pd.DataFrame(summary)

    if len(summary_df) == 0:
        print("没有足够样本的参数组合")
        return None

    # 按MFE/MAE比排序
    summary_df = summary_df.sort_values('mfe_mae', ascending=False)

    # 打印汇总表
    print(f"\n{'lb':>3} | {'hz':>3} | {'th':>5} | {'N':>4} | {'MFE_P25':>7} | {'MFE_P50':>7} | {'MFE_P75':>7} | {'MAE_P50':>7} | {'MFE/MAE':>7} | {'>0.1%':>6} | {'>0.2%':>6} | {'亏>0.2%':>6}")
    print("-" * 110)

    for _, row in summary_df.head(25).iterrows():
        print(f"{row['lb']:>3} | {row['hz']:>3} | {row['th']:>5} | {row['n']:>4} | "
              f"{row['mfe_p25']*100:>6.2f}% | {row['mfe_p50']*100:>6.2f}% | {row['mfe_p75']*100:>6.2f}% | "
              f"{row['mae_p50']*100:>6.2f}% | {row['mfe_mae']:>7.2f} | "
              f"{row['pct_01']*100:>5.1f}% | {row['pct_02']*100:>5.1f}% | {row['mae_02']*100:>5.1f}%")

    return summary_df


# ============ 主分析函数 ============
def run_enhanced_analysis(df, config, lb=15, hz=15, th=0.003, signal_type='gamma_neg_short'):
    """运行增强版分析（详细版，用于选定参数后的深入分析）"""

    # 解析信号类型
    if signal_type == 'gamma_neg_short':
        gamma_type, direction, is_short = 'negative', 'after_drop', True
        signal_name = "Gamma- 跌后做空"
    elif signal_type == 'gamma_neg_long':
        gamma_type, direction, is_short = 'negative', 'after_rise', False
        signal_name = "Gamma- 涨后做多"
    elif signal_type == 'gamma_pos_short':
        gamma_type, direction, is_short = 'positive', 'after_rise', True
        signal_name = "Gamma+ 涨后做空"
    elif signal_type == 'gamma_pos_long':
        gamma_type, direction, is_short = 'positive', 'after_drop', False
        signal_name = "Gamma+ 跌后做多"
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    print(f"\n{'='*80}")
    print(f"详细分析: {signal_name}")
    print(f"参数: lookback={lb}, horizon={hz}, threshold={th*100:.1f}%")
    print(f"{'='*80}")

    # 找信号
    signals = find_signals_enhanced(df, lb, th, gamma_type, direction, config)
    print(f"\n信号数量: {len(signals)}")

    if len(signals) < 10:
        print("样本不足")
        return None

    # 分析MFE/MAE
    results = analyze_signals_mfe_mae(df, signals, hz, is_short)

    if len(results) < 10:
        print("有效结果不足")
        return None

    # 基础统计
    print(f"\n【基础统计】")
    print(f"有效信号: {len(results)}")

    print(f"\nMFE分布:")
    for p in [25, 50, 75, 90]:
        print(f"  P{p}: {results['mfe'].quantile(p/100)*100:.3f}%")

    print(f"\nMAE分布:")
    for p in [10, 25, 50, 75]:
        print(f"  P{p}: {results['mae'].quantile(p/100)*100:.3f}%")

    mfe_mae = results['mfe'].median() / abs(results['mae'].median())
    print(f"\nMFE/MAE比: {mfe_mae:.2f}")

    print(f"\n盈利阈值:")
    for th_val in [0.001, 0.002, 0.003, 0.005]:
        pct = (results['mfe'] > th_val).mean()
        print(f"  >{th_val*100:.1f}%: {pct*100:.1f}%")

    print(f"\n亏损阈值:")
    for th_val in [0.001, 0.002, 0.003, 0.005]:
        pct = (results['mae'] < -th_val).mean()
        print(f"  >{th_val*100:.1f}%: {pct*100:.1f}%")

    # ============ 止盈止损分析 ============
    optimal_tpsl = find_optimal_tpsl(results, signal_name)
    print_tpsl_analysis(optimal_tpsl)

    # 分层分析
    analyze_by_zone(results)
    analyze_by_move_ratio(results)
    analyze_by_distance_to_support(results)

    # 交叉分析
    cross_analysis_zone_move_ratio(results)

    # Key Levels距离影响分析
    analyze_level_distance_impact(results, is_short)

    # 核心假设检验
    analyze_hypothesis_tests(results, is_short)

    return results


# ============ Zone分布统计 ============
def zone_distribution_analysis(df):
    """分析数据在各Zone的分布"""
    print(f"\n{'='*80}")
    print("Zone 分布统计")
    print(f"{'='*80}")

    zone_counts = df['zone'].value_counts()
    total = len(df)

    print(f"\n{'Zone':>25} | {'数量':>8} | {'占比':>8}")
    print("-" * 50)

    for zone in sorted(zone_counts.index):
        count = zone_counts[zone]
        pct = count / total * 100
        print(f"{zone:>25} | {count:>8} | {pct:>7.1f}%")

    # Gamma分布
    print(f"\n【Gamma环境分布】")
    gamma_counts = df['gamma'].value_counts()
    for g in ['positive', 'negative']:
        if g in gamma_counts:
            print(f"{g}: {gamma_counts[g]} ({gamma_counts[g]/total*100:.1f}%)")


# ============ 三维交叉扫描：参数 × Move Ratio × Key Level距离 ============
def comprehensive_3d_scan(df, config):
    """
    三维交叉扫描：Gamma环境 × 参数组合 × Move Ratio × Key Level距离
    找出最优的组合条件，并为每个组合计算最优止盈止损
    """
    print("\n" + "="*120)
    print("三维交叉扫描: Gamma环境 × 参数 × Move Ratio × Key Level距离 (含止盈止损分析)")
    print("="*120)

    # 参数范围（简化以减少组合数）
    lookbacks = [10, 15, 30]
    horizons = [15, 30]
    thresholds = [0.0005, 0.001, 0.0015, 0.002]  # 0.05%, 0.1%, 0.15%, 0.2%

    # Move Ratio 分组
    move_ratio_bins = {
        '<30%': (0, 0.3),
        '30-50%': (0.3, 0.5),
        '50-70%': (0.5, 0.7),
        '70-100%': (0.7, 1.0),
        '>100%': (1.0, float('inf'))
    }

    # MR 分级建议函数
    def get_mr_grade(mr_label):
        """根据 Move Ratio 范围返回交易建议分级"""
        if mr_label in ['>100%', '70-100%']:
            return '★★★推荐'  # 高MR - 强烈推荐交易
        elif mr_label in ['50-70%']:
            return '★★☆可选'   # 中MR - 可选择性交易
        else:  # '<30%', '30-50%'
            return '★☆☆观望'   # 低MR - 建议观望

    # 距离分组
    distance_bins = {
        'near': (0, 0.003),      # <0.3%
        'medium': (0.003, 0.007), # 0.3-0.7%
        'far': (0.007, float('inf'))  # >0.7%
    }

    # 要分析的距离字段
    distance_fields = [
        ('dist_to_support', '距Support'),
        ('dist_hedge_wall', '距HedgeWall'),
    ]

    # 信号类型配置
    signal_configs = [
        ('gamma_neg_short', 'negative', 'after_drop', True, 'G-做空'),
        ('gamma_neg_long', 'negative', 'after_rise', False, 'G-做多'),
        ('gamma_pos_short', 'positive', 'after_rise', True, 'G+做空'),
        ('gamma_pos_long', 'positive', 'after_drop', False, 'G+做多'),
    ]

    all_results = []

    for signal_type, gamma_type, direction, is_short, signal_name in signal_configs:
        print(f"\n{'='*100}")
        print(f"【{signal_name}】三维交叉分析 (含TP/SL优化)")
        print(f"{'='*100}")

        for lb in lookbacks:
            for hz in horizons:
                for th in thresholds:
                    # 找信号
                    signals = find_signals_enhanced(df, lb, th, gamma_type, direction, config)
                    if len(signals) < config['min_samples']:
                        continue

                    # 分析MFE/MAE
                    results = analyze_signals_mfe_mae(df, signals, hz, is_short)
                    if len(results) < config['min_samples']:
                        continue

                    # 按Move Ratio分层
                    for mr_label, (mr_low, mr_high) in move_ratio_bins.items():
                        mr_subset = results[(results['move_ratio'] >= mr_low) & (results['move_ratio'] < mr_high)]
                        if len(mr_subset) < 5:
                            continue

                        # 按距离分层
                        for dist_field, dist_name in distance_fields:
                            if dist_field not in mr_subset.columns:
                                continue

                            valid = mr_subset[mr_subset[dist_field].notna()].copy()
                            if len(valid) < 5:
                                continue

                            # 取绝对值
                            if dist_field in ['dist_hedge_wall']:
                                valid['abs_dist'] = valid[dist_field].abs()
                            else:
                                valid['abs_dist'] = valid[dist_field]

                            for dist_label, (low, high) in distance_bins.items():
                                subset = valid[(valid['abs_dist'] >= low) & (valid['abs_dist'] < high)]
                                if len(subset) < 5:
                                    continue

                                mfe_p50 = subset['mfe'].median()
                                mae_p50 = subset['mae'].median()
                                mfe_mae = mfe_p50 / abs(mae_p50) if mae_p50 != 0 else np.inf
                                pct_mfe_02 = (subset['mfe'] > 0.002).mean()
                                pct_mae_02 = (subset['mae'] < -0.002).mean()  # 亏损>0.2%

                                # 为这个多维度子集计算最优TP/SL
                                subset_name = f"{signal_name}_lb{lb}_hz{hz}_th{th*100:.1f}%_MR{mr_label}_{dist_name}{dist_label}"
                                tpsl_result = find_optimal_tpsl(subset, subset_name) if len(subset) >= 10 else None

                                result_entry = {
                                    'signal': signal_name,
                                    'gamma': gamma_type,
                                    'is_short': is_short,
                                    'lb': lb,
                                    'hz': hz,
                                    'th': th,
                                    'move_ratio': mr_label,
                                    'mr_grade': get_mr_grade(mr_label),  # 添加MR分级
                                    'dist_field': dist_name,
                                    'dist_bin': dist_label,
                                    'n': len(subset),
                                    'mfe_p50': mfe_p50,
                                    'mae_p50': mae_p50,
                                    'mfe_mae': mfe_mae,
                                    'pct_mfe_02': pct_mfe_02,
                                    'pct_mae_02': pct_mae_02,
                                }

                                # 添加TP/SL推荐
                                if tpsl_result:
                                    result_entry['recommended_tp'] = tpsl_result['recommended_tp']
                                    result_entry['recommended_sl'] = tpsl_result['recommended_sl']
                                    result_entry['best_tp'] = tpsl_result['best_tp']
                                    result_entry['best_sl'] = tpsl_result['best_sl']
                                    result_entry['best_expected'] = tpsl_result['best_expected']
                                    result_entry['best_p_tp'] = tpsl_result['best_p_tp']
                                    result_entry['best_p_sl'] = tpsl_result['best_p_sl']
                                else:
                                    result_entry['recommended_tp'] = None
                                    result_entry['recommended_sl'] = None
                                    result_entry['best_tp'] = None
                                    result_entry['best_sl'] = None
                                    result_entry['best_expected'] = None
                                    result_entry['best_p_tp'] = None
                                    result_entry['best_p_sl'] = None

                                all_results.append(result_entry)

    # 汇总结果
    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n" + "="*140)
        print("【三维交叉最优组合排名】(按MFE/MAE比率, N>=10)")
        print("="*140)

        # 过滤样本量足够的
        filtered = results_df[results_df['n'] >= 10].copy()
        if len(filtered) > 0:
            filtered = filtered.sort_values('mfe_mae', ascending=False)

            print(f"\n{'信号':>6} | {'lb':>2} | {'hz':>2} | {'th':>4} | {'MoveRatio':>8} | {'MR分级':>10} | {'距离类型':>8} | {'距离':>6} | {'N':>4} | {'MFE':>6} | {'MAE':>7} | {'MFE/MAE':>7} | {'盈>0.2%':>7} | {'亏>0.2%':>7}")
            print("-" * 155)

            for _, row in filtered.head(50).iterrows():
                print(f"{row['signal']:>6} | {row['lb']:>2} | {row['hz']:>2} | {row['th']*100:.1f}% | "
                      f"{row['move_ratio']:>8} | {row['mr_grade']:>10} | {row['dist_field']:>8} | {row['dist_bin']:>6} | {row['n']:>4} | "
                      f"{row['mfe_p50']*100:>5.2f}% | {row['mae_p50']*100:>6.2f}% | {row['mfe_mae']:>7.2f} | "
                      f"{row['pct_mfe_02']*100:>6.1f}% | {row['pct_mae_02']*100:>6.1f}%")

        # 按信号类型分组的最优（含TP/SL推荐）
        print("\n" + "="*140)
        print("【各信号类型 Top 5 组合 - 含多维度止盈止损推荐】")
        print("="*140)

        for signal_name in ['G-做空', 'G-做多', 'G+做空', 'G+做多']:
            sig_df = filtered[filtered['signal'] == signal_name]
            if len(sig_df) > 0:
                print(f"\n--- {signal_name} ---")
                for i, (_, row) in enumerate(sig_df.head(5).iterrows()):
                    tp_str = f"{row['best_tp']*100:.2f}%" if row['best_tp'] else "N/A"
                    sl_str = f"{row['best_sl']*100:.2f}%" if row['best_sl'] else "N/A"
                    exp_str = f"{row['best_expected']*100:.4f}%" if row['best_expected'] else "N/A"
                    print(f"  {i+1}. lb={row['lb']}, hz={row['hz']}, th={row['th']*100:.1f}%, "
                          f"MR={row['move_ratio']} [{row['mr_grade']}], {row['dist_field']}={row['dist_bin']}, "
                          f"N={row['n']}, MFE/MAE={row['mfe_mae']:.2f}")
                    print(f"     → 推荐TP={tp_str}, SL={sl_str}, 期望收益={exp_str}")

        # 高质量策略筛选（MFE/MAE>2 且 亏>0.2% < 盈>0.2%）含TP/SL
        print("\n" + "="*160)
        print("【高质量策略 - 含多维度止盈止损推荐】(MFE/MAE>2, 亏损率<盈利率, N>=10)")
        print("="*160)

        high_quality = filtered[
            (filtered['mfe_mae'] > 2) &
            (filtered['pct_mae_02'] < filtered['pct_mfe_02']) &
            (filtered['n'] >= 10)
        ].copy()

        if len(high_quality) > 0:
            high_quality = high_quality.sort_values('mfe_mae', ascending=False)
            print(f"\n{'信号':>6} | {'lb':>2} | {'hz':>2} | {'th':>4} | {'MoveRatio':>8} | {'MR分级':>10} | {'距离类型':>8} | {'距离':>6} | {'N':>4} | {'MFE/MAE':>7} | {'推荐TP':>7} | {'推荐SL':>7} | {'期望收益':>10}")
            print("-" * 155)

            for _, row in high_quality.iterrows():
                tp_str = f"{row['best_tp']*100:.2f}%" if row['best_tp'] else "N/A"
                sl_str = f"{row['best_sl']*100:.2f}%" if row['best_sl'] else "N/A"
                exp_str = f"{row['best_expected']*100:.4f}%" if row['best_expected'] else "N/A"
                print(f"{row['signal']:>6} | {row['lb']:>2} | {row['hz']:>2} | {row['th']*100:.1f}% | "
                      f"{row['move_ratio']:>8} | {row['mr_grade']:>10} | {row['dist_field']:>8} | {row['dist_bin']:>6} | {row['n']:>4} | "
                      f"{row['mfe_mae']:>7.2f} | {tp_str:>7} | {sl_str:>7} | {exp_str:>10}")
        else:
            print("没有符合条件的高质量策略")

        return results_df
    return None


# ============ 旧版二维扫描（保留兼容）============
def comprehensive_distance_param_scan(df, config):
    """
    综合扫描：Gamma环境 × 参数组合 × Key Level距离（二维版本）
    """
    print("\n" + "="*100)
    print("综合扫描: Gamma环境 × 参数 × Key Level距离")
    print("="*100)

    # 参数范围
    lookbacks = [5, 10, 15, 30]
    horizons = [5, 10, 15, 30]
    thresholds = [0.0005, 0.001, 0.0015, 0.002]

    # 距离分组
    distance_bins = {
        'near': (0, 0.003),
        'medium': (0.003, 0.007),
        'far': (0.007, float('inf'))
    }

    # 要分析的距离字段
    distance_fields = [
        ('dist_to_support', '距Support'),
        ('dist_to_resistance', '距Resistance'),
        ('dist_hedge_wall', '距HedgeWall'),
    ]

    # 信号类型配置
    signal_configs = [
        ('gamma_neg_short', 'negative', 'after_drop', True, 'G-做空'),
        ('gamma_neg_long', 'negative', 'after_rise', False, 'G-做多'),
        ('gamma_pos_short', 'positive', 'after_rise', True, 'G+做空'),
        ('gamma_pos_long', 'positive', 'after_drop', False, 'G+做多'),
    ]

    all_results = []

    for signal_type, gamma_type, direction, is_short, signal_name in signal_configs:
        print(f"\n{'='*80}")
        print(f"【{signal_name}】Gamma={gamma_type}, Direction={direction}")
        print(f"{'='*80}")

        for dist_field, dist_name in distance_fields:
            print(f"\n--- {dist_name} 分层 ---")
            print(f"{'lb':>3} | {'hz':>3} | {'th':>5} | {'距离':>8} | {'N':>4} | {'MFE_P50':>8} | {'MAE_P50':>8} | {'MFE/MAE':>7} | {'盈>0.2%':>7} | {'亏>0.2%':>7}")
            print("-" * 100)

            for lb in lookbacks:
                for hz in horizons:
                    for th in thresholds:
                        signals = find_signals_enhanced(df, lb, th, gamma_type, direction, config)
                        if len(signals) < config['min_samples']:
                            continue

                        results = analyze_signals_mfe_mae(df, signals, hz, is_short)
                        if len(results) < config['min_samples']:
                            continue

                        if dist_field not in results.columns:
                            continue

                        valid = results[results[dist_field].notna()].copy()
                        if len(valid) < 5:
                            continue

                        if dist_field in ['dist_hedge_wall']:
                            valid['abs_dist'] = valid[dist_field].abs()
                        else:
                            valid['abs_dist'] = valid[dist_field]

                        for dist_label, (low, high) in distance_bins.items():
                            subset = valid[(valid['abs_dist'] >= low) & (valid['abs_dist'] < high)]
                            if len(subset) < 5:
                                continue

                            mfe_p50 = subset['mfe'].median()
                            mae_p50 = subset['mae'].median()
                            mfe_mae = mfe_p50 / abs(mae_p50) if mae_p50 != 0 else np.inf
                            pct_mfe_02 = (subset['mfe'] > 0.002).mean()
                            pct_mae_02 = (subset['mae'] < -0.002).mean()

                            print(f"{lb:>3} | {hz:>3} | {th*100:.1f}% | {dist_label:>8} | {len(subset):>4} | "
                                  f"{mfe_p50*100:>7.3f}% | {mae_p50*100:>7.3f}% | {mfe_mae:>7.2f} | "
                                  f"{pct_mfe_02*100:>6.1f}% | {pct_mae_02*100:>6.1f}%")

                            all_results.append({
                                'signal': signal_name,
                                'gamma': gamma_type,
                                'is_short': is_short,
                                'lb': lb,
                                'hz': hz,
                                'th': th,
                                'dist_field': dist_name,
                                'dist_bin': dist_label,
                                'n': len(subset),
                                'mfe_p50': mfe_p50,
                                'mae_p50': mae_p50,
                                'mfe_mae': mfe_mae,
                                'pct_mfe_02': pct_mfe_02,
                                'pct_mae_02': pct_mae_02,
                            })

    if all_results:
        results_df = pd.DataFrame(all_results)
        return results_df
    return None


# ============ 主程序 ============
def main():
    print("="*80)
    print("Gamma策略增强分析 - Move Ratio & Key Levels")
    print("="*80)

    # 1. 加载数据
    key_levels = load_key_levels(CONFIG)
    df = load_data(CONFIG, key_levels)

    # 2. 计算增强指标
    print("\n计算增强指标...")
    df = calculate_move_ratio(df)
    df = classify_zone(df)
    df = calculate_all_level_distances(df)

    # 3. Zone分布统计
    zone_distribution_analysis(df)

    # ============================================================
    # 第一步：参数扫描汇总（快速对比）
    # ============================================================
    print("\n" + "="*80)
    print("第一步: 参数扫描汇总")
    print("="*80)

    # --- Gamma- 环境 ---
    print("\n" + "-"*60)
    print("【Gamma- 环境】(价格 < Hedge Wall，趋势延续)")
    print("-"*60)

    # Gamma- 跌后做空 (趋势延续)
    summary_neg_short = parameter_scan_summary(df, CONFIG, 'gamma_neg_short')

    # Gamma- 涨后做多 (趋势延续)
    summary_neg_long = parameter_scan_summary(df, CONFIG, 'gamma_neg_long')

    # --- Gamma+ 环境 ---
    print("\n" + "-"*60)
    print("【Gamma+ 环境】(价格 > Hedge Wall，均值回归)")
    print("-"*60)

    # Gamma+ 涨后做空 (均值回归)
    summary_pos_short = parameter_scan_summary(df, CONFIG, 'gamma_pos_short')

    # Gamma+ 跌后做多 (均值回归)
    summary_pos_long = parameter_scan_summary(df, CONFIG, 'gamma_pos_long')

    # ============================================================
    # 第二步：选定最优参数，进行详细分析
    # ============================================================
    print("\n" + "="*80)
    print("第二步: 最优参数详细分析")
    print("="*80)

    # --- Gamma- 详细分析 ---
    print("\n" + "-"*60)
    print("【Gamma- 环境详细分析】")
    print("-"*60)

    # Gamma- 跌后做空
    if summary_neg_short is not None and len(summary_neg_short) > 0:
        best = summary_neg_short.iloc[0]
        print(f"\nGamma- 跌后做空 最优参数: lb={best['lb']}, hz={best['hz']}, th={best['th']}")
        th_val = float(best['th'].replace('%', '')) / 100
        run_enhanced_analysis(df, CONFIG, lb=int(best['lb']), hz=int(best['hz']), th=th_val, signal_type='gamma_neg_short')

    # Gamma- 涨后做多
    if summary_neg_long is not None and len(summary_neg_long) > 0:
        best = summary_neg_long.iloc[0]
        print(f"\nGamma- 涨后做多 最优参数: lb={best['lb']}, hz={best['hz']}, th={best['th']}")
        th_val = float(best['th'].replace('%', '')) / 100
        run_enhanced_analysis(df, CONFIG, lb=int(best['lb']), hz=int(best['hz']), th=th_val, signal_type='gamma_neg_long')

    # --- Gamma+ 详细分析 ---
    print("\n" + "-"*60)
    print("【Gamma+ 环境详细分析】")
    print("-"*60)

    # Gamma+ 涨后做空 (均值回归)
    if summary_pos_short is not None and len(summary_pos_short) > 0:
        best = summary_pos_short.iloc[0]
        print(f"\nGamma+ 涨后做空 最优参数: lb={best['lb']}, hz={best['hz']}, th={best['th']}")
        th_val = float(best['th'].replace('%', '')) / 100
        run_enhanced_analysis(df, CONFIG, lb=int(best['lb']), hz=int(best['hz']), th=th_val, signal_type='gamma_pos_short')

    # Gamma+ 跌后做多 (均值回归)
    if summary_pos_long is not None and len(summary_pos_long) > 0:
        best = summary_pos_long.iloc[0]
        print(f"\nGamma+ 跌后做多 最优参数: lb={best['lb']}, hz={best['hz']}, th={best['th']}")
        th_val = float(best['th'].replace('%', '')) / 100
        run_enhanced_analysis(df, CONFIG, lb=int(best['lb']), hz=int(best['hz']), th=th_val, signal_type='gamma_pos_long')

    # ============================================================
    # 第三步：综合扫描 - Gamma × 参数 × 距离
    # ============================================================
    print("\n" + "="*80)
    print("第三步: 综合扫描 (Gamma × 参数 × Key Level距离)")
    print("="*80)

    comprehensive_results = comprehensive_distance_param_scan(df, CONFIG)

    # ============================================================
    # 第四步：三维交叉扫描 - Gamma × 参数 × Move Ratio × 距离
    # ============================================================
    print("\n" + "="*80)
    print("第四步: 三维交叉扫描 (Gamma × 参数 × Move Ratio × Key Level距离)")
    print("="*80)

    results_3d = comprehensive_3d_scan(df, CONFIG)

    # ============================================================
    # 第五步：四种信号类型止盈止损汇总
    # ============================================================
    print("\n" + "="*80)
    print("第五步: 四种信号类型止盈止损推荐汇总")
    print("="*80)

    # 使用推荐参数对四种信号类型进行止盈止损分析
    signal_configs = [
        ('gamma_neg_short', 'negative', 'after_drop', True, 'G- 跌后做空'),
        ('gamma_neg_long', 'negative', 'after_rise', False, 'G- 涨后做多'),
        ('gamma_pos_short', 'positive', 'after_rise', True, 'G+ 涨后做空'),
        ('gamma_pos_long', 'positive', 'after_drop', False, 'G+ 跌后做多'),
    ]

    # 使用通用参数 (lb=15, hz=15, th=0.1%) 进行汇总分析
    tpsl_summary = []
    for signal_type, gamma_type, direction, is_short, signal_name in signal_configs:
        signals = find_signals_enhanced(df, 15, 0.001, gamma_type, direction, CONFIG)
        if len(signals) >= CONFIG['min_samples']:
            results = analyze_signals_mfe_mae(df, signals, 15, is_short)
            if len(results) >= CONFIG['min_samples']:
                optimal = find_optimal_tpsl(results, signal_name)
                if optimal:
                    tpsl_summary.append(optimal)

    # 打印汇总表
    if tpsl_summary:
        print(f"\n{'信号类型':>12} | {'样本':>5} | {'MFE_P50':>8} | {'MAE_P25':>8} | {'推荐TP':>7} | {'推荐SL':>7} | {'最优TP':>7} | {'最优SL':>7} | {'期望收益':>10}")
        print("-" * 105)
        for opt in tpsl_summary:
            print(f"{opt['signal_name']:>12} | {opt['n_samples']:>5} | "
                  f"{opt['mfe_p50']*100:>7.3f}% | {opt['mae_p25']*100:>7.3f}% | "
                  f"{opt['recommended_tp']*100:>6.2f}% | {opt['recommended_sl']*100:>6.2f}% | "
                  f"{opt['best_tp']*100:>6.2f}% | {opt['best_sl']*100:>6.2f}% | "
                  f"{opt['best_expected']*100:>9.4f}%")

        # 计算通用推荐值
        avg_tp = np.mean([opt['recommended_tp'] for opt in tpsl_summary])
        avg_sl = np.mean([opt['recommended_sl'] for opt in tpsl_summary])
        print(f"\n【通用推荐参数】(四种信号平均)")
        print(f"  止盈 (TP): {avg_tp*100:.2f}%")
        print(f"  止损 (SL): {avg_sl*100:.2f}%")

        # 保存止盈止损汇总到CSV
        tpsl_df = pd.DataFrame([{
            'signal': opt['signal_name'],
            'n_samples': opt['n_samples'],
            'mfe_p50': opt['mfe_p50'],
            'mae_p10': opt['mae_p10'],
            'mae_p25': opt['mae_p25'],
            'recommended_tp': opt['recommended_tp'],
            'recommended_sl': opt['recommended_sl'],
            'best_tp': opt['best_tp'],
            'best_sl': opt['best_sl'],
            'best_expected': opt['best_expected'],
            'best_p_tp': opt['best_p_tp'],
            'best_p_sl': opt['best_p_sl'],
        } for opt in tpsl_summary])

    # ============================================================
    # 保存结果到CSV文件
    # ============================================================
    print("\n" + "="*80)
    print("保存结果到文件")
    print("="*80)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存三维扫描结果
    if results_3d is not None:
        output_file = output_dir / f"3d_scan_results_{timestamp}.csv"
        results_3d.to_csv(output_file, index=False)
        print(f"三维扫描结果已保存: {output_file}")

        # 保存高质量策略
        high_quality = results_3d[
            (results_3d['mfe_mae'] > 2) &
            (results_3d['pct_mae_02'] < results_3d['pct_mfe_02']) &
            (results_3d['n'] >= 10)
        ].sort_values('mfe_mae', ascending=False)

        if len(high_quality) > 0:
            hq_file = output_dir / f"high_quality_strategies_{timestamp}.csv"
            high_quality.to_csv(hq_file, index=False)
            print(f"高质量策略已保存: {hq_file}")

    # 保存二维扫描结果
    if comprehensive_results is not None:
        output_file = output_dir / f"2d_scan_results_{timestamp}.csv"
        comprehensive_results.to_csv(output_file, index=False)
        print(f"二维扫描结果已保存: {output_file}")

    # 保存止盈止损汇总
    if tpsl_summary:
        tpsl_file = output_dir / f"tpsl_recommendations_{timestamp}.csv"
        tpsl_df.to_csv(tpsl_file, index=False)
        print(f"止盈止损推荐已保存: {tpsl_file}")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
