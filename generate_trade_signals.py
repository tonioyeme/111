"""
Gamma Strategy 交易信号生成器
生成46天的所有交易信号，包含:
- 策略类型
- 入场时间和价格
- 出场时间和价格
- 止盈/止损设置
- 交易结果

基于三维扫描分析结果的最优策略:
- G+做多: lb=30, hz=15, th=0.1%, MR>=70%, 距Support=medium, TP=0.20%, SL=0.25%
- G-做多: lb=30, hz=15, th=0.2%, MR>=70%, 距HW=near, TP=0.20%, SL=0.20%
- G-做空: lb=15, hz=30, th=0.2%, MR>=70%, 距HW=near, TP=0.30%, SL=0.25%
- G+做空: lb=15, hz=15, th=0.2%, MR=30-50%, 距HW=medium, TP=0.20%, SL=0.20%
"""

import os
import yaml
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime, timedelta

# ============ 策略配置 (基于三维扫描最优结果) ============
STRATEGIES = {
    'G+做多_optimal': {
        'signal_type': 'gamma_pos_long',
        'gamma': 'positive',
        'direction': 'after_drop',
        'is_short': False,
        'lookback': 30,
        'horizon': 15,
        'threshold': 0.001,  # 0.1%
        'mr_min': 0.7,
        'mr_max': float('inf'),
        'dist_field': 'dist_to_support',
        'dist_min': 0.003,
        'dist_max': 0.007,
        'take_profit': 0.002,  # 0.20%
        'stop_loss': 0.0025,   # 0.25%
        'description': 'G+做多最优: lb=30, hz=15, th=0.1%, MR>=70%, 距Support=medium'
    },
    'G-做多_optimal': {
        'signal_type': 'gamma_neg_long',
        'gamma': 'negative',
        'direction': 'after_rise',
        'is_short': False,
        'lookback': 30,
        'horizon': 15,
        'threshold': 0.002,  # 0.2%
        'mr_min': 0.7,
        'mr_max': float('inf'),
        'dist_field': 'dist_hedge_wall',
        'dist_min': 0,
        'dist_max': 0.003,
        'take_profit': 0.002,  # 0.20%
        'stop_loss': 0.002,    # 0.20%
        'description': 'G-做多最优: lb=30, hz=15, th=0.2%, MR>=70%, 距HW=near'
    },
    'G-做空_optimal': {
        'signal_type': 'gamma_neg_short',
        'gamma': 'negative',
        'direction': 'after_drop',
        'is_short': True,
        'lookback': 15,
        'horizon': 30,
        'threshold': 0.002,  # 0.2%
        'mr_min': 0.7,
        'mr_max': float('inf'),
        'dist_field': 'dist_hedge_wall',
        'dist_min': 0,
        'dist_max': 0.003,
        'take_profit': 0.003,  # 0.30%
        'stop_loss': 0.0025,   # 0.25%
        'description': 'G-做空最优: lb=15, hz=30, th=0.2%, MR>=70%, 距HW=near'
    },
    'G+做空_optimal': {
        'signal_type': 'gamma_pos_short',
        'gamma': 'positive',
        'direction': 'after_rise',
        'is_short': True,
        'lookback': 15,
        'horizon': 15,
        'threshold': 0.002,  # 0.2%
        'mr_min': 0.3,
        'mr_max': 0.5,
        'dist_field': 'dist_hedge_wall',
        'dist_min': 0.003,
        'dist_max': 0.007,
        'take_profit': 0.002,  # 0.20%
        'stop_loss': 0.002,    # 0.20%
        'description': 'G+做空最优: lb=15, hz=15, th=0.2%, MR=30-50%, 距HW=medium'
    },
    # 宽松版本 - 更多信号
    'G+做多_relaxed': {
        'signal_type': 'gamma_pos_long',
        'gamma': 'positive',
        'direction': 'after_drop',
        'is_short': False,
        'lookback': 15,
        'horizon': 15,
        'threshold': 0.001,  # 0.1%
        'mr_min': 0.5,  # 降低MR要求
        'mr_max': float('inf'),
        'dist_field': None,  # 不限距离
        'dist_min': 0,
        'dist_max': float('inf'),
        'take_profit': 0.0015,  # 0.15%
        'stop_loss': 0.002,     # 0.20%
        'description': 'G+做多宽松: lb=15, hz=15, th=0.1%, MR>=50%'
    },
    'G-做多_relaxed': {
        'signal_type': 'gamma_neg_long',
        'gamma': 'negative',
        'direction': 'after_rise',
        'is_short': False,
        'lookback': 15,
        'horizon': 15,
        'threshold': 0.001,  # 0.1%
        'mr_min': 0.5,
        'mr_max': float('inf'),
        'dist_field': None,
        'dist_min': 0,
        'dist_max': float('inf'),
        'take_profit': 0.0015,
        'stop_loss': 0.002,
        'description': 'G-做多宽松: lb=15, hz=15, th=0.1%, MR>=50%'
    },
    'G-做空_relaxed': {
        'signal_type': 'gamma_neg_short',
        'gamma': 'negative',
        'direction': 'after_drop',
        'is_short': True,
        'lookback': 15,
        'horizon': 15,
        'threshold': 0.001,
        'mr_min': 0.5,
        'mr_max': float('inf'),
        'dist_field': None,
        'dist_min': 0,
        'dist_max': float('inf'),
        'take_profit': 0.0015,
        'stop_loss': 0.002,
        'description': 'G-做空宽松: lb=15, hz=15, th=0.1%, MR>=50%'
    },
    'G+做空_relaxed': {
        'signal_type': 'gamma_pos_short',
        'gamma': 'positive',
        'direction': 'after_rise',
        'is_short': True,
        'lookback': 15,
        'horizon': 15,
        'threshold': 0.001,
        'mr_min': 0.5,
        'mr_max': float('inf'),
        'dist_field': None,
        'dist_min': 0,
        'dist_max': float('inf'),
        'take_profit': 0.0015,
        'stop_loss': 0.002,
        'description': 'G+做空宽松: lb=15, hz=15, th=0.1%, MR>=50%'
    },
}

# ============ 数据加载 ============
def load_key_levels(yaml_dir='data/key_levels'):
    """加载所有日期的key levels配置"""
    key_levels = {}
    yaml_files = glob(os.path.join(yaml_dir, '*.yaml'))

    for yaml_file in yaml_files:
        date_str = os.path.basename(yaml_file).replace('.yaml', '')
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            # 处理 implied_move 百分比格式
            if 'implied_move' in data and isinstance(data['implied_move'], str):
                data['implied_move'] = float(data['implied_move'].replace('%', '')) / 100
            key_levels[date_str] = data

    return key_levels


def load_price_data(data_dir='data/spx'):
    """加载所有日期的价格数据"""
    csv_files = glob(os.path.join(data_dir, 'SPX_*_minute.csv'))

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.concat(all_data, ignore_index=True)

    # 解析时间戳 - 处理混合时区
    df['time'] = pd.to_datetime(df['timestamp'], utc=True)
    df['time'] = df['time'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date.astype(str)

    # 转换为5分钟数据
    df = convert_to_5min(df)

    return df


def convert_to_5min(df):
    """将1分钟数据转换为5分钟数据"""
    df = df.copy()
    df['time_5min'] = df['time'].dt.floor('5min')

    # 聚合为5分钟K线
    df_5min = df.groupby(['date', 'time_5min']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).reset_index()

    df_5min = df_5min.rename(columns={'time_5min': 'time'})
    df_5min = df_5min.sort_values('time').reset_index(drop=True)

    return df_5min


def prepare_data(df, key_levels):
    """准备数据 - 添加gamma环境、move ratio、距离等"""
    df = df.copy()

    # 添加 key levels
    df['hedge_wall'] = df['date'].map(lambda d: key_levels.get(d, {}).get('hedge_wall', np.nan))
    df['implied_high'] = df['date'].map(lambda d: key_levels.get(d, {}).get('implied_high', np.nan))
    df['implied_low'] = df['date'].map(lambda d: key_levels.get(d, {}).get('implied_low', np.nan))
    df['implied_move'] = df['date'].map(lambda d: key_levels.get(d, {}).get('implied_move', 0.006))

    # 支撑阻力
    df['support1'] = df['date'].map(lambda d: key_levels.get(d, {}).get('support', [0])[0] if key_levels.get(d, {}).get('support') else 0)
    df['resistance1'] = df['date'].map(lambda d: key_levels.get(d, {}).get('resistance', [0])[0] if key_levels.get(d, {}).get('resistance') else 0)

    # 计算每日开盘价
    daily_open = df.groupby('date')['open'].first()
    df['daily_open'] = df['date'].map(daily_open)

    # 处理缺失的 implied_high/low
    mask_no_ih = df['implied_high'].isna() | (df['implied_high'] == 0)
    df.loc[mask_no_ih, 'implied_high'] = df.loc[mask_no_ih, 'daily_open'] * (1 + df.loc[mask_no_ih, 'implied_move'])

    mask_no_il = df['implied_low'].isna() | (df['implied_low'] == 0)
    df.loc[mask_no_il, 'implied_low'] = df.loc[mask_no_il, 'daily_open'] * (1 - df.loc[mask_no_il, 'implied_move'])

    # Gamma 环境
    df['gamma'] = np.where(df['close'] > df['hedge_wall'], 'positive', 'negative')

    # Move Ratio
    implied_range = df['implied_high'] - df['implied_low']
    price_from_open = np.abs(df['close'] - df['daily_open'])
    df['move_ratio'] = np.where(implied_range > 0, price_from_open / implied_range, 0)

    # 距离计算
    df['dist_to_support'] = np.where(df['support1'] > 0, (df['close'] - df['support1']) / df['close'], np.nan)
    df['dist_to_resistance'] = np.where(df['resistance1'] > 0, (df['resistance1'] - df['close']) / df['close'], np.nan)
    df['dist_hedge_wall'] = np.abs((df['close'] - df['hedge_wall']) / df['close'])

    # 时间字段
    df['minutes_from_open'] = df.groupby('date').cumcount() * 5

    return df


# ============ 信号生成 ============
def find_signals(df, strategy):
    """根据策略配置找出信号"""
    lookback = strategy['lookback']
    threshold = strategy['threshold']
    gamma_type = strategy['gamma']
    direction = strategy['direction']
    mr_min = strategy['mr_min']
    mr_max = strategy['mr_max']
    dist_field = strategy['dist_field']
    dist_min = strategy['dist_min']
    dist_max = strategy['dist_max']

    df = df.copy()

    # 计算过去收益
    df['past_return'] = df['close'].pct_change(lookback)

    # 基础过滤 - 排除开盘前30分钟
    mask = (
        (df['minutes_from_open'] >= 30) &
        (df['gamma'] == gamma_type)
    )

    # 方向过滤
    if direction == 'after_drop':
        mask = mask & (df['past_return'] < -threshold)
    else:  # after_rise
        mask = mask & (df['past_return'] > threshold)

    # Move Ratio 过滤
    mask = mask & (df['move_ratio'] >= mr_min) & (df['move_ratio'] <= mr_max)

    # 距离过滤
    if dist_field and dist_field in df.columns:
        dist_vals = np.abs(df[dist_field])
        mask = mask & (dist_vals >= dist_min) & (dist_vals < dist_max)

    signals = df[mask].copy()

    return signals


def simulate_trade(df, entry_idx, strategy):
    """模拟单个交易，返回交易详情"""
    horizon = strategy['horizon']
    is_short = strategy['is_short']
    tp = strategy['take_profit']
    sl = strategy['stop_loss']

    if entry_idx + horizon >= len(df):
        return None

    entry_row = df.loc[entry_idx]
    entry_price = entry_row['close']
    entry_time = entry_row['time']

    # 获取未来价格序列
    future_slice = df.loc[entry_idx+1 : entry_idx+horizon]

    if len(future_slice) < horizon:
        return None

    exit_price = None
    exit_time = None
    exit_reason = None
    bars_held = 0

    for i, (idx, row) in enumerate(future_slice.iterrows()):
        bars_held = i + 1

        if is_short:
            # 做空: 价格下跌盈利，价格上涨亏损
            current_return = (entry_price - row['close']) / entry_price
            high_return = (entry_price - row['high']) / entry_price  # 最坏情况 (价格涨到high)
            low_return = (entry_price - row['low']) / entry_price    # 最好情况 (价格跌到low)

            # 检查是否触发止损 (价格上涨超过止损)
            if high_return <= -sl:
                exit_price = entry_price * (1 + sl)  # 止损价
                exit_time = row['time']
                exit_reason = 'stop_loss'
                break

            # 检查是否触发止盈 (价格下跌超过止盈)
            if low_return >= tp:
                exit_price = entry_price * (1 - tp)  # 止盈价
                exit_time = row['time']
                exit_reason = 'take_profit'
                break
        else:
            # 做多: 价格上涨盈利，价格下跌亏损
            current_return = (row['close'] - entry_price) / entry_price
            high_return = (row['high'] - entry_price) / entry_price  # 最好情况
            low_return = (row['low'] - entry_price) / entry_price    # 最坏情况

            # 检查是否触发止损 (价格下跌超过止损)
            if low_return <= -sl:
                exit_price = entry_price * (1 - sl)  # 止损价
                exit_time = row['time']
                exit_reason = 'stop_loss'
                break

            # 检查是否触发止盈 (价格上涨超过止盈)
            if high_return >= tp:
                exit_price = entry_price * (1 + tp)  # 止盈价
                exit_time = row['time']
                exit_reason = 'take_profit'
                break

    # 如果没有触发TP/SL，按timeout处理
    if exit_price is None:
        last_row = future_slice.iloc[-1]
        exit_price = last_row['close']
        exit_time = last_row['time']
        exit_reason = 'timeout'

    # 计算收益
    if is_short:
        pnl_pct = (entry_price - exit_price) / entry_price
    else:
        pnl_pct = (exit_price - entry_price) / entry_price

    # MR分级
    mr = entry_row['move_ratio']
    if mr >= 0.7:
        mr_grade = '★★★推荐'
    elif mr >= 0.5:
        mr_grade = '★★☆可选'
    else:
        mr_grade = '★☆☆观望'

    return {
        'date': entry_row['date'],
        'strategy': strategy['description'],
        'signal_type': strategy['signal_type'],
        'direction': 'SHORT' if is_short else 'LONG',
        'gamma_env': entry_row['gamma'],

        'entry_time': entry_time,
        'entry_price': round(entry_price, 2),

        'exit_time': exit_time,
        'exit_price': round(exit_price, 2),
        'exit_reason': exit_reason,

        'take_profit': f"{tp*100:.2f}%",
        'stop_loss': f"{sl*100:.2f}%",

        'bars_held': bars_held,
        'pnl_pct': round(pnl_pct * 100, 4),
        'pnl_points': round(exit_price - entry_price if not is_short else entry_price - exit_price, 2),

        'move_ratio': round(mr * 100, 1),
        'mr_grade': mr_grade,

        'lookback': strategy['lookback'],
        'horizon': strategy['horizon'],
        'threshold': f"{strategy['threshold']*100:.2f}%",
    }


def generate_all_signals(df, strategies, min_bars_between=5):
    """生成所有策略的信号"""
    all_trades = []

    for strategy_name, strategy in strategies.items():
        print(f"\n处理策略: {strategy_name}")
        print(f"  {strategy['description']}")

        signals = find_signals(df, strategy)
        print(f"  找到 {len(signals)} 个原始信号")

        # 避免重复交易 - 同一策略在 min_bars_between 内不重复入场
        last_entry_idx = {}  # 按日期记录
        trades_count = 0

        for idx in signals.index:
            date = df.loc[idx, 'date']

            # 检查是否与上次入场太近
            if date in last_entry_idx:
                if idx - last_entry_idx[date] < min_bars_between:
                    continue

            trade = simulate_trade(df, idx, strategy)
            if trade:
                trade['strategy_name'] = strategy_name
                all_trades.append(trade)
                last_entry_idx[date] = idx
                trades_count += 1

        print(f"  生成 {trades_count} 个有效交易")

    return pd.DataFrame(all_trades)


# ============ 统计分析 ============
def analyze_results(trades_df):
    """分析交易结果"""
    print("\n" + "="*100)
    print("交易统计汇总")
    print("="*100)

    # 按策略统计
    print("\n【按策略统计】")
    print("-"*100)

    for strategy in trades_df['strategy_name'].unique():
        subset = trades_df[trades_df['strategy_name'] == strategy]
        n = len(subset)

        wins = (subset['pnl_pct'] > 0).sum()
        losses = (subset['pnl_pct'] < 0).sum()
        win_rate = wins / n * 100 if n > 0 else 0

        avg_pnl = subset['pnl_pct'].mean()
        total_pnl = subset['pnl_pct'].sum()

        tp_count = (subset['exit_reason'] == 'take_profit').sum()
        sl_count = (subset['exit_reason'] == 'stop_loss').sum()
        timeout_count = (subset['exit_reason'] == 'timeout').sum()

        print(f"\n{strategy}:")
        print(f"  交易次数: {n}")
        print(f"  胜率: {win_rate:.1f}% ({wins}胜/{losses}负)")
        print(f"  平均收益: {avg_pnl:.3f}%")
        print(f"  累计收益: {total_pnl:.2f}%")
        print(f"  止盈/止损/超时: {tp_count}/{sl_count}/{timeout_count}")

    # 按MR分级统计
    print("\n" + "="*100)
    print("【按MR分级统计】")
    print("-"*100)

    for grade in ['★★★推荐', '★★☆可选', '★☆☆观望']:
        subset = trades_df[trades_df['mr_grade'] == grade]
        if len(subset) == 0:
            continue

        n = len(subset)
        wins = (subset['pnl_pct'] > 0).sum()
        win_rate = wins / n * 100
        avg_pnl = subset['pnl_pct'].mean()

        print(f"\n{grade} (MR分级):")
        print(f"  交易次数: {n}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均收益: {avg_pnl:.3f}%")

    # 按日期统计
    print("\n" + "="*100)
    print("【按日期统计 (每日交易)】")
    print("-"*100)

    daily_stats = trades_df.groupby('date').agg({
        'pnl_pct': ['count', 'sum', 'mean'],
    }).round(3)
    daily_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
    print(daily_stats.to_string())

    return daily_stats


# ============ 主程序 ============
def main():
    print("="*100)
    print("Gamma Strategy 交易信号生成器")
    print("="*100)

    # 加载数据
    print("\n加载数据...")
    key_levels = load_key_levels('data/key_levels')
    print(f"  加载 {len(key_levels)} 天的 key levels")

    df = load_price_data('data/spx')
    print(f"  加载 {len(df)} 行价格数据 (5分钟K线)")

    # 准备数据
    print("\n准备数据...")
    df = prepare_data(df, key_levels)

    dates = df['date'].unique()
    print(f"  数据日期范围: {dates[0]} 至 {dates[-1]} ({len(dates)} 天)")

    # 生成信号 - 使用所有策略 (optimal + relaxed)
    print("\n生成交易信号...")
    trades_df = generate_all_signals(df, STRATEGIES)

    if len(trades_df) == 0:
        print("没有生成任何交易信号!")
        return

    # 排序
    trades_df = trades_df.sort_values(['entry_time']).reset_index(drop=True)

    # 分析结果
    daily_stats = analyze_results(trades_df)

    # 保存结果
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存完整交易记录
    output_file = f'results/trade_signals_{timestamp}.csv'
    trades_df.to_csv(output_file, index=False)
    print(f"\n交易信号已保存: {output_file}")

    # 保存汇总版本 (关键字段)
    summary_cols = [
        'date', 'strategy_name', 'direction', 'gamma_env',
        'entry_time', 'entry_price',
        'exit_time', 'exit_price', 'exit_reason',
        'take_profit', 'stop_loss',
        'pnl_pct', 'pnl_points',
        'move_ratio', 'mr_grade'
    ]
    summary_df = trades_df[summary_cols]
    summary_file = f'results/trade_signals_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"交易汇总已保存: {summary_file}")

    # 打印样本
    print("\n" + "="*100)
    print("【交易信号样本 (前20条)】")
    print("="*100)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(summary_df.head(20).to_string())

    # 总体统计
    print("\n" + "="*100)
    print("【总体统计】")
    print("="*100)

    total_trades = len(trades_df)
    total_wins = (trades_df['pnl_pct'] > 0).sum()
    total_losses = (trades_df['pnl_pct'] < 0).sum()
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()

    print(f"总交易次数: {total_trades}")
    print(f"总胜率: {total_wins/total_trades*100:.1f}% ({total_wins}胜/{total_losses}负)")
    print(f"累计收益: {total_pnl:.2f}%")
    print(f"平均收益: {avg_pnl:.3f}%")
    print(f"日均交易: {total_trades/len(dates):.1f} 次")


if __name__ == "__main__":
    main()
