"""
Gamma Strategy 统计测试框架
用于验证和优化策略参数

测试层次：
- Level 1: 单变量敏感性测试
- Level 2: 组合优化测试
- Level 3: 风控规则测试
- Level 4: 样本外验证 (含Walk-Forward)
- Level 5: 统计显著性检验 (Bootstrap + Monte Carlo)
"""

import os
import yaml
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# ============ 统计显著性检验工具 ============
def bootstrap_confidence_interval(pnls, n_bootstrap=1000, ci=0.95):
    """计算盈亏的Bootstrap置信区间"""
    if len(pnls) < 10:
        return None, None, False

    pnls = np.array(pnls)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(pnls, size=len(pnls), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1-ci)/2 * 100)
    upper = np.percentile(bootstrap_means, (1+ci)/2 * 100)

    # 判断是否显著（0是否在置信区间外）
    significant = (lower > 0) or (upper < 0)

    return lower, upper, significant


def monte_carlo_significance(pnls, n_simulations=10000):
    """蒙特卡洛显著性检验：随机打乱盈亏看是否能达到相同表现"""
    if len(pnls) < 10:
        return None

    pnls = np.array(pnls)
    actual_total = np.sum(pnls)
    actual_mean = np.mean(pnls)
    actual_sharpe = actual_mean / np.std(pnls) if np.std(pnls) > 0 else 0

    # 随机打乱盈亏的正负号
    abs_pnls = np.abs(pnls)
    random_totals = []
    random_sharpes = []

    for _ in range(n_simulations):
        signs = np.random.choice([-1, 1], size=len(abs_pnls))
        random_pnls = abs_pnls * signs
        random_totals.append(np.sum(random_pnls))
        std = np.std(random_pnls)
        random_sharpes.append(np.mean(random_pnls) / std if std > 0 else 0)

    # 计算p-value (单尾检验)
    if actual_total > 0:
        p_value_total = np.mean(np.array(random_totals) >= actual_total)
    else:
        p_value_total = np.mean(np.array(random_totals) <= actual_total)

    if actual_sharpe > 0:
        p_value_sharpe = np.mean(np.array(random_sharpes) >= actual_sharpe)
    else:
        p_value_sharpe = np.mean(np.array(random_sharpes) <= actual_sharpe)

    return {
        'actual_total': round(actual_total, 4),
        'p_value_total': round(p_value_total, 4),
        'significant_total': p_value_total < 0.05,
        'actual_sharpe': round(actual_sharpe, 4),
        'p_value_sharpe': round(p_value_sharpe, 4),
        'significant_sharpe': p_value_sharpe < 0.05,
    }


def max_consecutive_count(bool_array):
    """计算最大连续True的个数"""
    max_count = 0
    current_count = 0
    for val in bool_array:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count

# ============ 数据加载（复用 generate_trade_signals.py 的函数）============
def load_key_levels(yaml_dir='data/key_levels'):
    """加载所有日期的key levels配置"""
    key_levels = {}
    yaml_files = glob(os.path.join(yaml_dir, '*.yaml'))

    for yaml_file in yaml_files:
        date_str = os.path.basename(yaml_file).replace('.yaml', '')
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
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
    df['time'] = pd.to_datetime(df['timestamp'], utc=True)
    df['time'] = df['time'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date.astype(str)

    # 转换为5分钟数据
    df['time_5min'] = df['time'].dt.floor('5min')
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
    """准备数据"""
    df = df.copy()

    df['hedge_wall'] = df['date'].map(lambda d: key_levels.get(d, {}).get('hedge_wall', np.nan))
    df['implied_high'] = df['date'].map(lambda d: key_levels.get(d, {}).get('implied_high', np.nan))
    df['implied_low'] = df['date'].map(lambda d: key_levels.get(d, {}).get('implied_low', np.nan))
    df['implied_move'] = df['date'].map(lambda d: key_levels.get(d, {}).get('implied_move', 0.006))

    df['support1'] = df['date'].map(lambda d: key_levels.get(d, {}).get('support', [0])[0] if key_levels.get(d, {}).get('support') else 0)
    df['resistance1'] = df['date'].map(lambda d: key_levels.get(d, {}).get('resistance', [0])[0] if key_levels.get(d, {}).get('resistance') else 0)

    daily_open = df.groupby('date')['open'].first()
    df['daily_open'] = df['date'].map(daily_open)

    mask_no_ih = df['implied_high'].isna() | (df['implied_high'] == 0)
    df.loc[mask_no_ih, 'implied_high'] = df.loc[mask_no_ih, 'daily_open'] * (1 + df.loc[mask_no_ih, 'implied_move'])

    mask_no_il = df['implied_low'].isna() | (df['implied_low'] == 0)
    df.loc[mask_no_il, 'implied_low'] = df.loc[mask_no_il, 'daily_open'] * (1 - df.loc[mask_no_il, 'implied_move'])

    df['gamma'] = np.where(df['close'] > df['hedge_wall'], 'positive', 'negative')

    implied_range = df['implied_high'] - df['implied_low']
    price_from_open = np.abs(df['close'] - df['daily_open'])
    df['move_ratio'] = np.where(implied_range > 0, price_from_open / implied_range, 0)

    df['dist_to_support'] = np.where(df['support1'] > 0, (df['close'] - df['support1']) / df['close'], np.nan)
    df['dist_to_resistance'] = np.where(df['resistance1'] > 0, (df['resistance1'] - df['close']) / df['close'], np.nan)
    df['dist_hedge_wall'] = np.abs((df['close'] - df['hedge_wall']) / df['close'])

    df['minutes_from_open'] = df.groupby('date').cumcount() * 5

    return df


# ============ 回测引擎 ============
class Backtester:
    """策略回测引擎"""

    def __init__(self, df):
        self.df = df.copy()

    def find_signals(self, strategy_type, lookback, threshold, mr_min, mr_max,
                     dist_field=None, dist_min=0, dist_max=float('inf')):
        """根据参数找出信号"""
        df = self.df.copy()

        # 策略配置
        strategy_config = {
            'G+做多': ('positive', 'after_drop', False),
            'G-做多': ('negative', 'after_rise', False),
            'G-做空': ('negative', 'after_drop', True),
            'G+做空': ('positive', 'after_rise', True),
        }

        gamma_type, direction, is_short = strategy_config[strategy_type]

        # 计算过去收益
        df['past_return'] = df['close'].pct_change(lookback)

        # 基础过滤
        mask = (
            (df['minutes_from_open'] >= 30) &
            (df['gamma'] == gamma_type)
        )

        # 方向过滤
        if direction == 'after_drop':
            mask = mask & (df['past_return'] < -threshold)
        else:
            mask = mask & (df['past_return'] > threshold)

        # Move Ratio 过滤
        mask = mask & (df['move_ratio'] >= mr_min) & (df['move_ratio'] <= mr_max)

        # 距离过滤
        if dist_field and dist_field in df.columns:
            dist_vals = np.abs(df[dist_field])
            mask = mask & (dist_vals >= dist_min) & (dist_vals < dist_max)

        signals = df[mask].copy()
        return signals, is_short

    def simulate_trades(self, signals, is_short, horizon, tp, sl,
                        cooldown_bars=0, max_daily_loss=None, extreme_mr_rule=None):
        """模拟交易"""
        trades = []
        last_exit_idx = {}  # 按日期记录上次出场位置
        daily_pnl = {}  # 按日期记录当日盈亏

        for idx in signals.index:
            date = self.df.loc[idx, 'date']
            mr = self.df.loc[idx, 'move_ratio']

            # 冷却期检查
            if date in last_exit_idx:
                if idx - last_exit_idx[date] < cooldown_bars:
                    continue

            # 单日亏损限制检查
            if max_daily_loss and date in daily_pnl:
                if daily_pnl[date] <= -max_daily_loss:
                    continue

            # 极端MR处理 (修正逻辑)
            if extreme_mr_rule and mr > 1.0:
                if extreme_mr_rule == 'no_trade':
                    continue
                elif extreme_mr_rule == 'no_counter_trend':
                    # 在MR>100%时判断是否逆势交易
                    # MR>100%意味着价格已经超出implied range
                    gamma = self.df.loc[idx, 'gamma']
                    price_vs_open = self.df.loc[idx, 'close'] - self.df.loc[idx, 'daily_open']

                    # G+做多 + 价格在跌（极端超跌后抄底）= 逆势
                    if gamma == 'positive' and not is_short and price_vs_open < 0:
                        continue
                    # G-做空 + 价格在涨（极端超涨后做空）= 逆势
                    if gamma == 'negative' and is_short and price_vs_open > 0:
                        continue
                    # G+做空 + 价格在涨 = 逆势
                    if gamma == 'positive' and is_short and price_vs_open > 0:
                        continue
                    # G-做多 + 价格在跌 = 逆势
                    if gamma == 'negative' and not is_short and price_vs_open < 0:
                        continue

            # 模拟交易
            trade = self._simulate_single_trade(idx, is_short, horizon, tp, sl)
            if trade:
                trades.append(trade)
                last_exit_idx[date] = idx + trade['bars_held']

                # 更新当日盈亏
                if date not in daily_pnl:
                    daily_pnl[date] = 0
                daily_pnl[date] += trade['pnl_pct']

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    def _simulate_single_trade(self, entry_idx, is_short, horizon, tp, sl):
        """模拟单个交易"""
        if entry_idx + horizon >= len(self.df):
            return None

        entry_row = self.df.loc[entry_idx]
        entry_price = entry_row['close']
        entry_time = entry_row['time']

        future_slice = self.df.loc[entry_idx+1 : entry_idx+horizon]
        if len(future_slice) < horizon:
            return None

        exit_price = None
        exit_time = None
        exit_reason = None
        bars_held = 0

        for i, (idx, row) in enumerate(future_slice.iterrows()):
            bars_held = i + 1

            if is_short:
                high_return = (entry_price - row['high']) / entry_price
                low_return = (entry_price - row['low']) / entry_price

                if high_return <= -sl:
                    exit_price = entry_price * (1 + sl)
                    exit_time = row['time']
                    exit_reason = 'stop_loss'
                    break

                if low_return >= tp:
                    exit_price = entry_price * (1 - tp)
                    exit_time = row['time']
                    exit_reason = 'take_profit'
                    break
            else:
                high_return = (row['high'] - entry_price) / entry_price
                low_return = (row['low'] - entry_price) / entry_price

                if low_return <= -sl:
                    exit_price = entry_price * (1 - sl)
                    exit_time = row['time']
                    exit_reason = 'stop_loss'
                    break

                if high_return >= tp:
                    exit_price = entry_price * (1 + tp)
                    exit_time = row['time']
                    exit_reason = 'take_profit'
                    break

        if exit_price is None:
            last_row = future_slice.iloc[-1]
            exit_price = last_row['close']
            exit_time = last_row['time']
            exit_reason = 'timeout'

        if is_short:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        else:
            pnl_pct = (exit_price - entry_price) / entry_price * 100

        return {
            'date': entry_row['date'],
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl_pct': pnl_pct,
            'move_ratio': entry_row['move_ratio'],
        }


def calculate_stats(trades_df, include_extended=False):
    """计算统计指标"""
    if len(trades_df) == 0:
        return {
            'n': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0,
            'max_dd': 0, 'tp_rate': 0, 'sl_rate': 0, 'timeout_rate': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0
        }

    n = len(trades_df)
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] < 0]

    win_rate = len(wins) / n * 100
    avg_pnl = trades_df['pnl_pct'].mean()
    total_pnl = trades_df['pnl_pct'].sum()

    # 最大回撤
    cumsum = trades_df['pnl_pct'].cumsum()
    max_dd = (cumsum - cumsum.cummax()).min()

    # 出场原因比例
    tp_rate = (trades_df['exit_reason'] == 'take_profit').sum() / n * 100
    sl_rate = (trades_df['exit_reason'] == 'stop_loss').sum() / n * 100
    timeout_rate = (trades_df['exit_reason'] == 'timeout').sum() / n * 100

    # 盈亏比
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0
    profit_factor = (wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum())) if len(losses) > 0 and losses['pnl_pct'].sum() != 0 else 0

    result = {
        'n': n,
        'win_rate': round(win_rate, 1),
        'avg_pnl': round(avg_pnl, 4),
        'total_pnl': round(total_pnl, 2),
        'max_dd': round(max_dd, 2),
        'tp_rate': round(tp_rate, 1),
        'sl_rate': round(sl_rate, 1),
        'timeout_rate': round(timeout_rate, 1),
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'profit_factor': round(profit_factor, 2),
    }

    # 扩展统计指标
    if include_extended and n >= 2:
        pnls = trades_df['pnl_pct'].values

        # Sharpe Ratio (按交易计算)
        sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

        # Calmar Ratio (总收益 / 最大回撤)
        calmar = total_pnl / abs(max_dd) if max_dd != 0 else 0

        # 连续亏损
        is_loss = pnls < 0
        max_consec_loss = max_consecutive_count(is_loss)

        # Recovery factor
        recovery_factor = total_pnl / abs(max_dd) if max_dd != 0 else 0

        # Bootstrap置信区间
        ci_lower, ci_upper, ci_significant = bootstrap_confidence_interval(pnls)

        result.update({
            'sharpe': round(sharpe, 2),
            'calmar': round(calmar, 2),
            'max_consec_loss': max_consec_loss,
            'recovery_factor': round(recovery_factor, 2),
            'ci_95_lower': round(ci_lower, 4) if ci_lower else None,
            'ci_95_upper': round(ci_upper, 4) if ci_upper else None,
            'ci_significant': ci_significant,
        })

    return result


# ============ Level 1: 单变量敏感性测试 ============
def test_mr_threshold(backtester, strategy_type):
    """测试MR阈值对策略表现的影响"""
    print(f"\n{'='*80}")
    print(f"测试1: MR阈值敏感性 - {strategy_type}")
    print(f"{'='*80}")

    # 固定参数
    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    mr_thresholds = [
        (0.0, 0.3, '<30%'),
        (0.3, 0.5, '30-50%'),
        (0.5, 0.7, '50-70%'),
        (0.7, 1.0, '70-100%'),
        (1.0, float('inf'), '>100%'),
        (0.5, float('inf'), '>=50%'),
        (0.7, float('inf'), '>=70%'),
    ]

    results = []
    for mr_min, mr_max, label in mr_thresholds:
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, mr_min, mr_max
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl)
        stats = calculate_stats(trades)
        stats['mr_range'] = label
        results.append(stats)

    # 输出结果
    print(f"\n{'MR范围':>10} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8} | {'TP率':>5} | {'SL率':>5} | {'PF':>5}")
    print("-" * 80)
    for r in results:
        print(f"{r['mr_range']:>10} | {r['n']:>4} | {r['win_rate']:>5.1f}% | "
              f"{r['avg_pnl']:>7.3f}% | {r['total_pnl']:>7.2f}% | "
              f"{r['tp_rate']:>4.1f}% | {r['sl_rate']:>4.1f}% | {r['profit_factor']:>5.2f}")

    return pd.DataFrame(results)


def test_tpsl_grid(backtester, strategy_type, mr_min=0.5):
    """测试TP/SL组合"""
    print(f"\n{'='*80}")
    print(f"测试2: TP/SL网格搜索 - {strategy_type} (MR>={mr_min*100:.0f}%)")
    print(f"{'='*80}")

    lb, hz, th = 15, 15, 0.001

    tp_range = [0.001, 0.0015, 0.002, 0.0025, 0.003]  # 0.1%-0.3%
    sl_range = [0.0015, 0.002, 0.0025, 0.003]  # 0.15%-0.3%

    results = []
    for tp, sl in product(tp_range, sl_range):
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, mr_min, float('inf')
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl)
        stats = calculate_stats(trades)
        stats['tp'] = f"{tp*100:.2f}%"
        stats['sl'] = f"{sl*100:.2f}%"
        stats['tp_sl_ratio'] = round(tp/sl, 2)
        results.append(stats)

    # 按总盈亏排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_pnl', ascending=False)

    # 输出Top 10
    print(f"\n{'TP':>6} | {'SL':>6} | {'比例':>5} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8} | {'PF':>5}")
    print("-" * 75)
    for _, r in results_df.head(10).iterrows():
        print(f"{r['tp']:>6} | {r['sl']:>6} | {r['tp_sl_ratio']:>5.2f} | {r['n']:>4} | "
              f"{r['win_rate']:>5.1f}% | {r['avg_pnl']:>7.3f}% | {r['total_pnl']:>7.2f}% | {r['profit_factor']:>5.2f}")

    return results_df


def test_lookback(backtester, strategy_type, mr_min=0.5):
    """测试Lookback周期"""
    print(f"\n{'='*80}")
    print(f"测试3: Lookback周期 - {strategy_type} (MR>={mr_min*100:.0f}%)")
    print(f"{'='*80}")

    hz, th = 15, 0.001
    tp, sl = 0.0015, 0.002

    lookbacks = [5, 10, 15, 20, 30, 45]

    results = []
    for lb in lookbacks:
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, mr_min, float('inf')
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl)
        stats = calculate_stats(trades)
        stats['lookback'] = lb
        results.append(stats)

    print(f"\n{'LB':>5} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8} | {'PF':>5}")
    print("-" * 55)
    for r in results:
        print(f"{r['lookback']:>5} | {r['n']:>4} | {r['win_rate']:>5.1f}% | "
              f"{r['avg_pnl']:>7.3f}% | {r['total_pnl']:>7.2f}% | {r['profit_factor']:>5.2f}")

    return pd.DataFrame(results)


# ============ Level 3: 风控规则测试 ============
def test_cooldown(backtester, strategy_type, mr_min=0.5):
    """测试入场冷却期"""
    print(f"\n{'='*80}")
    print(f"测试4: 入场冷却期 - {strategy_type}")
    print(f"{'='*80}")

    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    cooldowns = [0, 1, 3, 5, 10, 15]  # bars (5min each)

    results = []
    for cd in cooldowns:
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, mr_min, float('inf')
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl, cooldown_bars=cd)
        stats = calculate_stats(trades)
        stats['cooldown_min'] = cd * 5
        results.append(stats)

    print(f"\n{'冷却期':>8} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8} | {'MaxDD':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['cooldown_min']:>6}min | {r['n']:>4} | {r['win_rate']:>5.1f}% | "
              f"{r['avg_pnl']:>7.3f}% | {r['total_pnl']:>7.2f}% | {r['max_dd']:>6.2f}%")

    return pd.DataFrame(results)


def test_extreme_mr_rule(backtester, strategy_type):
    """测试极端MR日处理规则"""
    print(f"\n{'='*80}")
    print(f"测试5: 极端MR(>100%)处理规则 - {strategy_type}")
    print(f"{'='*80}")

    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    rules = [
        (None, '正常交易'),
        ('no_trade', '禁止交易'),
        ('no_counter_trend', '禁止逆势'),
    ]

    results = []
    for rule, label in rules:
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, 0.5, float('inf')
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl, extreme_mr_rule=rule)
        stats = calculate_stats(trades)
        stats['rule'] = label
        results.append(stats)

    print(f"\n{'规则':>12} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['rule']:>12} | {r['n']:>4} | {r['win_rate']:>5.1f}% | "
              f"{r['avg_pnl']:>7.3f}% | {r['total_pnl']:>7.2f}%")

    return pd.DataFrame(results)


def test_daily_loss_limit(backtester, strategy_type, mr_min=0.5):
    """测试单日亏损限制"""
    print(f"\n{'='*80}")
    print(f"测试6: 单日亏损限制 - {strategy_type}")
    print(f"{'='*80}")

    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    limits = [None, 0.5, 0.3, 0.2]  # 无限制, -0.5%, -0.3%, -0.2%

    results = []
    for limit in limits:
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, mr_min, float('inf')
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl, max_daily_loss=limit)
        stats = calculate_stats(trades)
        stats['daily_limit'] = f"-{limit}%" if limit else "无限制"
        results.append(stats)

    print(f"\n{'限制':>10} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8} | {'MaxDD':>7}")
    print("-" * 65)
    for r in results:
        print(f"{r['daily_limit']:>10} | {r['n']:>4} | {r['win_rate']:>5.1f}% | "
              f"{r['avg_pnl']:>7.3f}% | {r['total_pnl']:>7.2f}% | {r['max_dd']:>6.2f}%")

    return pd.DataFrame(results)


# ============ Level 2: 组合优化测试 ============
def grid_search_full(backtester, strategy_type):
    """完整网格搜索"""
    print(f"\n{'='*80}")
    print(f"组合优化: 网格搜索 - {strategy_type}")
    print(f"{'='*80}")

    param_grid = {
        'mr_min': [0.5, 0.7],
        'lb': [10, 15, 30],
        'th': [0.001, 0.0015, 0.002],
        'tp': [0.0015, 0.002, 0.0025],
        'sl': [0.002, 0.0025],
        'hz': [15, 30],
    }

    all_combinations = list(product(
        param_grid['mr_min'],
        param_grid['lb'],
        param_grid['th'],
        param_grid['tp'],
        param_grid['sl'],
        param_grid['hz'],
    ))

    print(f"总组合数: {len(all_combinations)}")

    results = []
    for i, (mr_min, lb, th, tp, sl, hz) in enumerate(all_combinations):
        signals, is_short = backtester.find_signals(
            strategy_type, lb, th, mr_min, float('inf')
        )
        trades = backtester.simulate_trades(signals, is_short, hz, tp, sl)
        stats = calculate_stats(trades)

        stats['mr_min'] = f">={mr_min*100:.0f}%"
        stats['lb'] = lb
        stats['th'] = f"{th*100:.2f}%"
        stats['tp'] = f"{tp*100:.2f}%"
        stats['sl'] = f"{sl*100:.2f}%"
        stats['hz'] = hz

        results.append(stats)

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(all_combinations)}")

    results_df = pd.DataFrame(results)

    # 过滤N>=10的结果
    valid_results = results_df[results_df['n'] >= 10].copy()

    if len(valid_results) > 0:
        # 按总盈亏排序
        valid_results = valid_results.sort_values('total_pnl', ascending=False)

        print(f"\n【Top 15 参数组合】(N>=10)")
        print(f"{'MR':>6} | {'LB':>3} | {'TH':>6} | {'TP':>6} | {'SL':>6} | {'HZ':>3} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8} | {'PF':>5}")
        print("-" * 85)
        for _, r in valid_results.head(15).iterrows():
            print(f"{r['mr_min']:>6} | {r['lb']:>3} | {r['th']:>6} | {r['tp']:>6} | {r['sl']:>6} | "
                  f"{r['hz']:>3} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}% | {r['profit_factor']:>5.2f}")

    return results_df


# ============ 新增: 按日分析 ============
def test_by_date(backtester, strategy_type, mr_min=0.5):
    """按日期分析交易表现，找出异常日"""
    print(f"\n{'='*80}")
    print(f"按日分析 - {strategy_type} (MR>={mr_min*100:.0f}%)")
    print(f"{'='*80}")

    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    signals, is_short = backtester.find_signals(strategy_type, lb, th, mr_min, float('inf'))
    trades = backtester.simulate_trades(signals, is_short, hz, tp, sl)

    if len(trades) == 0:
        print("没有交易记录")
        return None

    # 按日期汇总
    daily_stats = trades.groupby('date').agg({
        'pnl_pct': ['count', 'sum', 'mean'],
        'exit_reason': lambda x: (x == 'stop_loss').sum()
    })
    daily_stats.columns = ['n_trades', 'total_pnl', 'avg_pnl', 'n_stops']
    daily_stats = daily_stats.reset_index()

    # 排序
    worst_days = daily_stats.nsmallest(5, 'total_pnl')
    best_days = daily_stats.nlargest(5, 'total_pnl')

    print("\n【最差5天】")
    print(f"{'日期':>12} | {'交易数':>5} | {'总盈亏':>8} | {'平均盈亏':>8} | {'止损数':>5}")
    print("-" * 55)
    for _, row in worst_days.iterrows():
        print(f"{row['date']:>12} | {row['n_trades']:>5} | {row['total_pnl']:>7.2f}% | "
              f"{row['avg_pnl']:>7.3f}% | {row['n_stops']:>5}")

    print("\n【最好5天】")
    print(f"{'日期':>12} | {'交易数':>5} | {'总盈亏':>8} | {'平均盈亏':>8} | {'止损数':>5}")
    print("-" * 55)
    for _, row in best_days.iterrows():
        print(f"{row['date']:>12} | {row['n_trades']:>5} | {row['total_pnl']:>7.2f}% | "
              f"{row['avg_pnl']:>7.3f}% | {row['n_stops']:>5}")

    # 计算去掉最好/最差天后的表现
    total_pnl_all = trades['pnl_pct'].sum()
    if len(daily_stats) > 2:
        best_day_pnl = best_days.iloc[0]['total_pnl']
        worst_day_pnl = worst_days.iloc[0]['total_pnl']
        print(f"\n总盈亏: {total_pnl_all:.2f}%")
        print(f"去掉最好一天后: {total_pnl_all - best_day_pnl:.2f}%")
        print(f"去掉最差一天后: {total_pnl_all - worst_day_pnl:.2f}%")

    return daily_stats


# ============ Level 4: 样本外验证 (改进版) ============
def time_split_validation_v2(backtester, strategy_type, split_date='2025-11-15'):
    """改进版时间切分验证 - 在训练集上优化参数，测试集验证"""
    print(f"\n{'='*80}")
    print(f"样本外验证(改进版): 时间切分 - {strategy_type}")
    print(f"切分日期: {split_date}")
    print(f"{'='*80}")

    # 分割数据 - 重要：必须reset_index，否则索引比较会出错
    df_train = backtester.df[backtester.df['date'] < split_date].copy().reset_index(drop=True)
    df_test = backtester.df[backtester.df['date'] >= split_date].copy().reset_index(drop=True)

    if len(df_train) == 0 or len(df_test) == 0:
        print("训练集或测试集为空")
        return None

    bt_train = Backtester(df_train)
    bt_test = Backtester(df_test)

    print(f"训练集: {df_train['date'].min()} ~ {df_train['date'].max()} ({df_train['date'].nunique()}天)")
    print(f"测试集: {df_test['date'].min()} ~ {df_test['date'].max()} ({df_test['date'].nunique()}天)")

    # Step 1: 在训练集上简化网格搜索找最优参数
    print("\n[Step 1] 在训练集上搜索最优参数...")
    param_grid = {
        'mr_min': [0.5, 0.7],
        'lb': [10, 15],
        'th': [0.001, 0.0015],
        'tp': [0.0015, 0.002, 0.0025],
        'sl': [0.002, 0.0025],
        'hz': [15],
    }

    best_result = None
    best_pnl = -float('inf')

    for mr_min, lb, th, tp, sl, hz in product(
        param_grid['mr_min'], param_grid['lb'], param_grid['th'],
        param_grid['tp'], param_grid['sl'], param_grid['hz']
    ):
        signals, is_short = bt_train.find_signals(strategy_type, lb, th, mr_min, float('inf'))
        trades = bt_train.simulate_trades(signals, is_short, hz, tp, sl)
        if len(trades) >= 5:
            total_pnl = trades['pnl_pct'].sum()
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_result = {
                    'mr_min': mr_min, 'lb': lb, 'th': th, 'tp': tp, 'sl': sl, 'hz': hz,
                    'train_pnl': total_pnl, 'train_n': len(trades),
                    'train_wr': (trades['pnl_pct'] > 0).mean() * 100
                }

    if best_result is None:
        print("训练集上没有找到有效参数组合")
        return None

    print(f"最优参数: MR>={best_result['mr_min']*100:.0f}%, lb={best_result['lb']}, th={best_result['th']*100:.1f}%, "
          f"TP={best_result['tp']*100:.2f}%, SL={best_result['sl']*100:.2f}%")
    print(f"训练集表现: N={best_result['train_n']}, 胜率={best_result['train_wr']:.1f}%, 总盈亏={best_result['train_pnl']:.2f}%")

    # Step 2: 用最优参数在测试集验证
    print("\n[Step 2] 在测试集上验证...")
    signals_test, is_short = bt_test.find_signals(
        strategy_type, best_result['lb'], best_result['th'],
        best_result['mr_min'], float('inf')
    )
    trades_test = bt_test.simulate_trades(
        signals_test, is_short, best_result['hz'],
        best_result['tp'], best_result['sl']
    )
    stats_test = calculate_stats(trades_test, include_extended=True)

    print(f"测试集表现: N={stats_test['n']}, 胜率={stats_test['win_rate']:.1f}%, 总盈亏={stats_test['total_pnl']:.2f}%")

    if stats_test['n'] >= 10:
        print(f"95%置信区间: [{stats_test.get('ci_95_lower', 'N/A')}, {stats_test.get('ci_95_upper', 'N/A')}]")
        print(f"显著性: {'✓ 显著' if stats_test.get('ci_significant') else '✗ 不显著'}")

    return {'best_params': best_result, 'test_stats': stats_test}


def walk_forward_validation(backtester, strategy_type, train_days=25, test_days=5):
    """Walk-Forward滚动窗口验证"""
    print(f"\n{'='*80}")
    print(f"Walk-Forward验证 - {strategy_type}")
    print(f"训练窗口: {train_days}天, 测试窗口: {test_days}天")
    print(f"{'='*80}")

    df = backtester.df
    dates = sorted(df['date'].unique())

    if len(dates) < train_days + test_days:
        print("数据不足以进行Walk-Forward验证")
        return None

    results = []
    all_test_trades = []

    # 滚动窗口
    for i in range(train_days, len(dates) - test_days + 1, test_days):
        train_dates = dates[i-train_days:i]
        test_dates = dates[i:i+test_days]

        # 重要：必须reset_index，否则索引比较会出错
        df_train = df[df['date'].isin(train_dates)].copy().reset_index(drop=True)
        df_test = df[df['date'].isin(test_dates)].copy().reset_index(drop=True)

        bt_train = Backtester(df_train)
        bt_test = Backtester(df_test)

        # 在训练集上快速搜索最优参数
        best_pnl = -float('inf')
        best_params = None

        for mr_min in [0.5, 0.7]:
            for lb in [10, 15]:
                for th in [0.001, 0.0015]:
                    tp, sl, hz = 0.002, 0.002, 15
                    signals, is_short = bt_train.find_signals(strategy_type, lb, th, mr_min, float('inf'))
                    trades = bt_train.simulate_trades(signals, is_short, hz, tp, sl)
                    if len(trades) >= 3:
                        total_pnl = trades['pnl_pct'].sum()
                        if total_pnl > best_pnl:
                            best_pnl = total_pnl
                            best_params = {'mr_min': mr_min, 'lb': lb, 'th': th, 'tp': tp, 'sl': sl, 'hz': hz}

        if best_params is None:
            continue

        # 在测试集上验证
        signals_test, is_short = bt_test.find_signals(
            strategy_type, best_params['lb'], best_params['th'],
            best_params['mr_min'], float('inf')
        )
        trades_test = bt_test.simulate_trades(
            signals_test, is_short, best_params['hz'],
            best_params['tp'], best_params['sl']
        )

        if len(trades_test) > 0:
            all_test_trades.append(trades_test)
            results.append({
                'train_period': f"{train_dates[0]}~{train_dates[-1]}",
                'test_period': f"{test_dates[0]}~{test_dates[-1]}",
                'train_pnl': round(best_pnl, 2),
                'test_n': len(trades_test),
                'test_pnl': round(trades_test['pnl_pct'].sum(), 2),
                'test_wr': round((trades_test['pnl_pct'] > 0).mean() * 100, 1),
            })

    if not results:
        print("没有足够的Walk-Forward结果")
        return None

    # 输出每个窗口的结果
    print(f"\n{'训练期':>25} | {'测试期':>25} | {'训练盈亏':>8} | {'测试N':>5} | {'测试盈亏':>8} | {'测试胜率':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['train_period']:>25} | {r['test_period']:>25} | {r['train_pnl']:>7.2f}% | "
              f"{r['test_n']:>5} | {r['test_pnl']:>7.2f}% | {r['test_wr']:>6.1f}%")

    # 汇总统计
    if all_test_trades:
        combined_trades = pd.concat(all_test_trades, ignore_index=True)
        total_test_pnl = combined_trades['pnl_pct'].sum()
        total_test_n = len(combined_trades)
        total_test_wr = (combined_trades['pnl_pct'] > 0).mean() * 100

        print(f"\n【汇总】测试集总交易: {total_test_n}, 总盈亏: {total_test_pnl:.2f}%, 胜率: {total_test_wr:.1f}%")

        # 蒙特卡洛显著性检验
        mc_result = monte_carlo_significance(combined_trades['pnl_pct'].values)
        if mc_result:
            print(f"蒙特卡洛检验: p-value={mc_result['p_value_total']:.4f}, "
                  f"{'✓ 显著' if mc_result['significant_total'] else '✗ 不显著'}")

    return pd.DataFrame(results)


def time_split_validation(backtester, strategy_type, split_date='2025-11-15'):
    """时间切分验证 (保留旧版兼容)"""
    print(f"\n{'='*80}")
    print(f"样本外验证: 时间切分 - {strategy_type}")
    print(f"切分日期: {split_date}")
    print(f"{'='*80}")

    # 最佳参数（基于前面的测试结果）
    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002
    mr_min = 0.5

    # 分割数据 - 重要：必须reset_index，否则索引比较会出错
    df_train = backtester.df[backtester.df['date'] < split_date].copy().reset_index(drop=True)
    df_test = backtester.df[backtester.df['date'] >= split_date].copy().reset_index(drop=True)

    bt_train = Backtester(df_train)
    bt_test = Backtester(df_test)

    # 训练集表现
    signals_train, is_short = bt_train.find_signals(strategy_type, lb, th, mr_min, float('inf'))
    trades_train = bt_train.simulate_trades(signals_train, is_short, hz, tp, sl)
    stats_train = calculate_stats(trades_train)

    # 测试集表现
    signals_test, is_short = bt_test.find_signals(strategy_type, lb, th, mr_min, float('inf'))
    trades_test = bt_test.simulate_trades(signals_test, is_short, hz, tp, sl)
    stats_test = calculate_stats(trades_test)

    print(f"\n{'数据集':>10} | {'日期范围':>25} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8}")
    print("-" * 80)
    train_dates = f"{df_train['date'].min()} ~ {df_train['date'].max()}"
    test_dates = f"{df_test['date'].min()} ~ {df_test['date'].max()}"
    print(f"{'训练集':>10} | {train_dates:>25} | {stats_train['n']:>4} | {stats_train['win_rate']:>5.1f}% | "
          f"{stats_train['avg_pnl']:>7.3f}% | {stats_train['total_pnl']:>7.2f}%")
    print(f"{'测试集':>10} | {test_dates:>25} | {stats_test['n']:>4} | {stats_test['win_rate']:>5.1f}% | "
          f"{stats_test['avg_pnl']:>7.3f}% | {stats_test['total_pnl']:>7.2f}%")

    return {'train': stats_train, 'test': stats_test}


# ============ 新增: 综合显著性检验 ============
def comprehensive_significance_test(backtester, strategy_type, mr_min=0.5):
    """综合显著性检验 - Bootstrap + Monte Carlo"""
    print(f"\n{'='*80}")
    print(f"综合显著性检验 - {strategy_type} (MR>={mr_min*100:.0f}%)")
    print(f"{'='*80}")

    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    signals, is_short = backtester.find_signals(strategy_type, lb, th, mr_min, float('inf'))
    trades = backtester.simulate_trades(signals, is_short, hz, tp, sl)

    if len(trades) < 10:
        print(f"交易数不足 ({len(trades)} < 10)，无法进行显著性检验")
        return None

    pnls = trades['pnl_pct'].values
    stats = calculate_stats(trades, include_extended=True)

    print(f"\n基础统计:")
    print(f"  交易数: {stats['n']}")
    print(f"  胜率: {stats['win_rate']:.1f}%")
    print(f"  总盈亏: {stats['total_pnl']:.2f}%")
    print(f"  平均盈亏: {stats['avg_pnl']:.4f}%")
    print(f"  Sharpe: {stats.get('sharpe', 'N/A')}")
    print(f"  最大连续亏损: {stats.get('max_consec_loss', 'N/A')}")

    # Bootstrap置信区间
    ci_lower, ci_upper, ci_significant = bootstrap_confidence_interval(pnls)
    print(f"\nBootstrap 95%置信区间:")
    print(f"  区间: [{ci_lower:.4f}%, {ci_upper:.4f}%]")
    print(f"  显著性: {'✓ 显著 (0不在区间内)' if ci_significant else '✗ 不显著 (0在区间内)'}")

    # 蒙特卡洛检验
    mc_result = monte_carlo_significance(pnls)
    if mc_result:
        print(f"\n蒙特卡洛显著性检验 (10000次模拟):")
        print(f"  总盈亏 p-value: {mc_result['p_value_total']:.4f}")
        print(f"  总盈亏显著性: {'✓ p<0.05' if mc_result['significant_total'] else '✗ p>=0.05'}")
        print(f"  Sharpe p-value: {mc_result['p_value_sharpe']:.4f}")
        print(f"  Sharpe显著性: {'✓ p<0.05' if mc_result['significant_sharpe'] else '✗ p>=0.05'}")

    # 综合判断
    is_significant = ci_significant and (mc_result and mc_result['significant_total'])
    print(f"\n【综合结论】: {'✓ 策略表现显著' if is_significant else '✗ 策略表现不显著'}")

    return {
        'stats': stats,
        'bootstrap': {'lower': ci_lower, 'upper': ci_upper, 'significant': ci_significant},
        'monte_carlo': mc_result,
        'overall_significant': is_significant
    }


# ============ 分层验证方法 (Stratified Validation) ============
def get_gamma_days(df):
    """获取G+和G-交易日列表 (基于gamma列: 'positive'/'negative')"""
    # 每日开盘时的gamma值决定当日状态
    daily_gamma = df.groupby('date')['gamma'].first()
    g_plus_days = daily_gamma[daily_gamma == 'positive'].index.tolist()
    g_minus_days = daily_gamma[daily_gamma == 'negative'].index.tolist()
    return g_plus_days, g_minus_days


def check_gamma_distribution(df):
    """检查数据中的Gamma分布"""
    g_plus_days, g_minus_days = get_gamma_days(df)
    total_days = len(g_plus_days) + len(g_minus_days)

    print(f"\n=== Gamma分布分析 ===")
    print(f"总交易日: {total_days}")
    print(f"G+ 交易日: {len(g_plus_days)} ({len(g_plus_days)/total_days*100:.1f}%)")
    print(f"G- 交易日: {len(g_minus_days)} ({len(g_minus_days)/total_days*100:.1f}%)")

    return {'g_plus': g_plus_days, 'g_minus': g_minus_days}


def stratified_time_split(df, strategy_type, test_ratio=0.3, mr_min=0.5):
    """
    分层时间切分验证
    按Gamma状态分层后再进行时间切分，确保测试集有足够的相关Gamma日

    Args:
        df: 完整数据集
        strategy_type: 策略类型 ('G+做多', 'G-做空' 等)
        test_ratio: 测试集比例
        mr_min: 最小MR阈值

    Returns:
        验证结果字典
    """
    print(f"\n{'='*80}")
    print(f"分层时间切分验证 - {strategy_type}")
    print(f"{'='*80}")

    # 确定需要的Gamma状态
    is_gplus_strategy = 'G+' in strategy_type

    # 获取目标Gamma状态的交易日（按时间排序）
    daily_gamma = df.groupby('date')['gamma'].first()
    if is_gplus_strategy:
        gamma_days = sorted([d for d in daily_gamma.index if daily_gamma[d] == 'positive'])
        target_gamma = 'G+'
    else:
        gamma_days = sorted([d for d in daily_gamma.index if daily_gamma[d] == 'negative'])
        target_gamma = 'G-'

    if len(gamma_days) < 5:
        print(f"警告: {target_gamma}交易日太少 ({len(gamma_days)}天)，无法进行有效验证")
        return None

    # 按时间切分（保留顺序）
    n_train = int(len(gamma_days) * (1 - test_ratio))
    train_days = gamma_days[:n_train]
    test_days = gamma_days[n_train:]

    print(f"\n{target_gamma}交易日分层:")
    print(f"  总计: {len(gamma_days)} 天")
    print(f"  训练集: {len(train_days)} 天 ({train_days[0]} ~ {train_days[-1]})")
    print(f"  测试集: {len(test_days)} 天 ({test_days[0]} ~ {test_days[-1]})")

    # 创建训练集和测试集
    df_train = df[df['date'].isin(train_days)].copy().reset_index(drop=True)
    df_test = df[df['date'].isin(test_days)].copy().reset_index(drop=True)

    # 回测参数
    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    # 训练集回测
    bt_train = Backtester(df_train)
    signals_train, is_short = bt_train.find_signals(strategy_type, lb, th, mr_min, float('inf'))
    trades_train = bt_train.simulate_trades(signals_train, is_short, hz, tp, sl)
    stats_train = calculate_stats(trades_train)

    # 测试集回测
    bt_test = Backtester(df_test)
    signals_test, is_short = bt_test.find_signals(strategy_type, lb, th, mr_min, float('inf'))
    trades_test = bt_test.simulate_trades(signals_test, is_short, hz, tp, sl)
    stats_test = calculate_stats(trades_test)

    print(f"\n{'数据集':>10} | {'天数':>4} | {'N':>4} | {'胜率':>6} | {'平均盈亏':>8} | {'总盈亏':>8}")
    print("-" * 70)
    print(f"{'训练集':>10} | {len(train_days):>4} | {stats_train['n']:>4} | {stats_train['win_rate']:>5.1f}% | "
          f"{stats_train['avg_pnl']:>7.3f}% | {stats_train['total_pnl']:>7.2f}%")
    print(f"{'测试集':>10} | {len(test_days):>4} | {stats_test['n']:>4} | {stats_test['win_rate']:>5.1f}% | "
          f"{stats_test['avg_pnl']:>7.3f}% | {stats_test['total_pnl']:>7.2f}%")

    # 测试集Bootstrap置信区间
    if len(trades_test) >= 10:
        pnls_test = trades_test['pnl_pct'].values
        ci_lower, ci_upper, ci_significant = bootstrap_confidence_interval(pnls_test)
        print(f"\n测试集Bootstrap 95%置信区间:")
        print(f"  区间: [{ci_lower:.4f}%, {ci_upper:.4f}%]")
        print(f"  显著性: {'✓ 显著' if ci_significant else '✗ 不显著'}")
    else:
        ci_lower, ci_upper, ci_significant = None, None, False
        print(f"\n测试集交易数不足，无法计算置信区间")

    return {
        'train': stats_train,
        'test': stats_test,
        'train_days': train_days,
        'test_days': test_days,
        'bootstrap_ci': {'lower': ci_lower, 'upper': ci_upper, 'significant': ci_significant}
    }


def random_day_sampling_validation(df, strategy_type, n_iterations=50, test_ratio=0.3, mr_min=0.5):
    """
    随机抽样稳健性检验
    多次随机抽取训练/测试集，检验策略稳健性

    Args:
        df: 完整数据集
        strategy_type: 策略类型
        n_iterations: 迭代次数
        test_ratio: 测试集比例
        mr_min: 最小MR阈值

    Returns:
        验证结果字典
    """
    print(f"\n{'='*80}")
    print(f"随机抽样稳健性检验 - {strategy_type} ({n_iterations}次迭代)")
    print(f"{'='*80}")

    # 确定需要的Gamma状态
    is_gplus_strategy = 'G+' in strategy_type

    # 获取目标Gamma状态的交易日
    daily_gamma = df.groupby('date')['gamma'].first()
    if is_gplus_strategy:
        gamma_days = [d for d in daily_gamma.index if daily_gamma[d] == 'positive']
        target_gamma = 'G+'
    else:
        gamma_days = [d for d in daily_gamma.index if daily_gamma[d] == 'negative']
        target_gamma = 'G-'

    if len(gamma_days) < 5:
        print(f"警告: {target_gamma}交易日太少")
        return None

    n_test = max(2, int(len(gamma_days) * test_ratio))

    # 回测参数
    lb, hz, th = 15, 15, 0.001
    tp, sl = 0.0015, 0.002

    test_results = []

    for i in range(n_iterations):
        # 随机抽取测试集
        np.random.seed(i)
        test_days = np.random.choice(gamma_days, size=n_test, replace=False).tolist()
        train_days = [d for d in gamma_days if d not in test_days]

        # 测试集回测
        df_test = df[df['date'].isin(test_days)].copy().reset_index(drop=True)
        bt_test = Backtester(df_test)
        signals_test, is_short = bt_test.find_signals(strategy_type, lb, th, mr_min, float('inf'))
        trades_test = bt_test.simulate_trades(signals_test, is_short, hz, tp, sl)

        if len(trades_test) > 0:
            stats = calculate_stats(trades_test)
            test_results.append({
                'iteration': i,
                'n_trades': stats['n'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'total_pnl': stats['total_pnl']
            })

    if len(test_results) == 0:
        print("没有有效的测试结果")
        return None

    results_df = pd.DataFrame(test_results)

    # 统计分析
    profitable_count = (results_df['total_pnl'] > 0).sum()
    profitable_pct = profitable_count / len(results_df) * 100

    print(f"\n{n_iterations}次随机抽样结果:")
    print(f"  有效迭代: {len(results_df)}")
    print(f"  盈利次数: {profitable_count} ({profitable_pct:.1f}%)")
    print(f"  平均盈亏: {results_df['avg_pnl'].mean():.4f}%")
    print(f"  盈亏标准差: {results_df['avg_pnl'].std():.4f}%")
    print(f"  最差表现: {results_df['total_pnl'].min():.2f}%")
    print(f"  最佳表现: {results_df['total_pnl'].max():.2f}%")

    # 判断稳健性
    is_robust = profitable_pct >= 70  # 70%以上迭代盈利认为稳健
    print(f"\n【稳健性判断】: {'✓ 稳健 (>70%迭代盈利)' if is_robust else '✗ 不稳健'}")

    return {
        'results': results_df,
        'profitable_pct': profitable_pct,
        'mean_pnl': results_df['avg_pnl'].mean(),
        'std_pnl': results_df['avg_pnl'].std(),
        'is_robust': is_robust
    }


def stratified_validation_full(backtester, strategy_type, mr_min=0.5):
    """
    完整分层验证流程
    包括: 分布检查 + 分层时间切分 + 随机抽样稳健性检验
    """
    print(f"\n{'#'*80}")
    print(f"# 完整分层验证 - {strategy_type}")
    print(f"{'#'*80}")

    df = backtester.df

    # Step 1: Gamma分布检查
    gamma_dist = check_gamma_distribution(df)

    # Step 2: 分层时间切分验证
    stratified_result = stratified_time_split(df, strategy_type, test_ratio=0.3, mr_min=mr_min)

    # Step 3: 随机抽样稳健性检验
    random_result = random_day_sampling_validation(df, strategy_type, n_iterations=50, mr_min=mr_min)

    # 综合判断
    print(f"\n{'='*80}")
    print(f"综合判断 - {strategy_type}")
    print(f"{'='*80}")

    is_significant = False
    is_robust = False

    if stratified_result and stratified_result['bootstrap_ci']['significant']:
        is_significant = True
        print(f"✓ 分层验证显著 (Bootstrap CI不包含0)")
    else:
        print(f"✗ 分层验证不显著")

    if random_result and random_result['is_robust']:
        is_robust = True
        print(f"✓ 随机抽样稳健 (>{random_result['profitable_pct']:.0f}%迭代盈利)")
    else:
        print(f"✗ 随机抽样不稳健")

    recommendation = "★★★推荐" if (is_significant and is_robust) else \
                     "★★☆可选" if (is_significant or is_robust) else "★☆☆观望"
    print(f"\n【策略评级】: {recommendation}")

    return {
        'gamma_distribution': gamma_dist,
        'stratified_split': stratified_result,
        'random_sampling': random_result,
        'is_significant': is_significant,
        'is_robust': is_robust,
        'recommendation': recommendation
    }


# ============ 主程序 ============
def main():
    print("="*80)
    print("Gamma Strategy 统计测试框架 (增强版)")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    key_levels = load_key_levels('data/key_levels')
    df = load_price_data('data/spx')
    df = prepare_data(df, key_levels)
    print(f"  数据日期: {df['date'].min()} ~ {df['date'].max()} ({df['date'].nunique()} 天)")

    # 创建回测器
    backtester = Backtester(df)

    # 存储所有结果
    all_results = {}

    # 四种策略类型
    strategies = ['G+做多', 'G-做多', 'G-做空', 'G+做空']

    for strategy in strategies:
        print(f"\n\n{'#'*80}")
        print(f"# 策略: {strategy}")
        print(f"{'#'*80}")

        # Level 1: 单变量测试
        all_results[f'{strategy}_mr'] = test_mr_threshold(backtester, strategy)
        all_results[f'{strategy}_tpsl'] = test_tpsl_grid(backtester, strategy)
        all_results[f'{strategy}_lb'] = test_lookback(backtester, strategy)

        # Level 3: 风控规则测试
        all_results[f'{strategy}_cooldown'] = test_cooldown(backtester, strategy)
        all_results[f'{strategy}_extreme_mr'] = test_extreme_mr_rule(backtester, strategy)
        all_results[f'{strategy}_daily_limit'] = test_daily_loss_limit(backtester, strategy)

        # Level 4: 样本外验证
        all_results[f'{strategy}_validation'] = time_split_validation(backtester, strategy)

    # Level 2: 组合优化（只对表现最好的策略做）
    print("\n\n" + "="*80)
    print("组合优化测试")
    print("="*80)

    for strategy in ['G-做空', 'G+做多']:  # 基于之前分析这两个表现较好
        all_results[f'{strategy}_grid'] = grid_search_full(backtester, strategy)

    # ============ 新增测试 ============
    print("\n\n" + "="*80)
    print("增强测试 (按日分析 + Walk-Forward + 显著性检验)")
    print("="*80)

    # 只对可能有效的策略进行深度测试
    for strategy in ['G+做多', 'G-做空']:
        print(f"\n\n{'#'*80}")
        print(f"# 深度分析: {strategy}")
        print(f"{'#'*80}")

        # 按日分析
        all_results[f'{strategy}_daily'] = test_by_date(backtester, strategy)

        # Walk-Forward验证
        all_results[f'{strategy}_wf'] = walk_forward_validation(backtester, strategy)

        # 改进版时间切分验证
        all_results[f'{strategy}_validation_v2'] = time_split_validation_v2(backtester, strategy)

        # 综合显著性检验
        all_results[f'{strategy}_significance'] = comprehensive_significance_test(backtester, strategy)

    # 保存结果到单一文件
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/gamma_strategy_test_results_{timestamp}.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Gamma Strategy 统计测试完整报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据日期: {df['date'].min()} ~ {df['date'].max()} ({df['date'].nunique()} 天)\n")
        f.write("="*80 + "\n\n")

        # 按策略组织结果
        for strategy in strategies:
            f.write("\n" + "#"*80 + "\n")
            f.write(f"# 策略: {strategy}\n")
            f.write("#"*80 + "\n\n")

            # MR阈值测试
            if f'{strategy}_mr' in all_results and isinstance(all_results[f'{strategy}_mr'], pd.DataFrame):
                f.write("【MR阈值敏感性测试】\n")
                f.write("-"*60 + "\n")
                mr_df = all_results[f'{strategy}_mr']
                f.write(f"{'MR范围':>10} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8} | {'PF':>5}\n")
                for _, r in mr_df.iterrows():
                    f.write(f"{r['mr_range']:>10} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}% | {r['profit_factor']:>5.2f}\n")
                f.write("\n")

            # TP/SL网格测试 (Top 5)
            if f'{strategy}_tpsl' in all_results and isinstance(all_results[f'{strategy}_tpsl'], pd.DataFrame):
                f.write("【TP/SL网格测试】(Top 5)\n")
                f.write("-"*60 + "\n")
                tpsl_df = all_results[f'{strategy}_tpsl'].head(5)
                f.write(f"{'TP':>6} | {'SL':>6} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8}\n")
                for _, r in tpsl_df.iterrows():
                    f.write(f"{r['tp']:>6} | {r['sl']:>6} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}%\n")
                f.write("\n")

            # Lookback测试
            if f'{strategy}_lb' in all_results and isinstance(all_results[f'{strategy}_lb'], pd.DataFrame):
                f.write("【Lookback周期测试】\n")
                f.write("-"*60 + "\n")
                lb_df = all_results[f'{strategy}_lb']
                f.write(f"{'LB':>5} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8}\n")
                for _, r in lb_df.iterrows():
                    f.write(f"{r['lookback']:>5} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}%\n")
                f.write("\n")

            # 冷却期测试
            if f'{strategy}_cooldown' in all_results and isinstance(all_results[f'{strategy}_cooldown'], pd.DataFrame):
                f.write("【入场冷却期测试】\n")
                f.write("-"*60 + "\n")
                cd_df = all_results[f'{strategy}_cooldown']
                f.write(f"{'冷却期':>8} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8}\n")
                for _, r in cd_df.iterrows():
                    f.write(f"{r['cooldown_min']:>6}min | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}%\n")
                f.write("\n")

            # 极端MR规则测试
            if f'{strategy}_extreme_mr' in all_results and isinstance(all_results[f'{strategy}_extreme_mr'], pd.DataFrame):
                f.write("【极端MR(>100%)处理规则测试】\n")
                f.write("-"*60 + "\n")
                emr_df = all_results[f'{strategy}_extreme_mr']
                f.write(f"{'规则':>12} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8}\n")
                for _, r in emr_df.iterrows():
                    f.write(f"{r['rule']:>12} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}%\n")
                f.write("\n")

            # 单日亏损限制测试
            if f'{strategy}_daily_limit' in all_results and isinstance(all_results[f'{strategy}_daily_limit'], pd.DataFrame):
                f.write("【单日亏损限制测试】\n")
                f.write("-"*60 + "\n")
                dl_df = all_results[f'{strategy}_daily_limit']
                f.write(f"{'限制':>10} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8}\n")
                for _, r in dl_df.iterrows():
                    f.write(f"{r['daily_limit']:>10} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}%\n")
                f.write("\n")

            # 时间切分验证
            if f'{strategy}_validation' in all_results and isinstance(all_results[f'{strategy}_validation'], dict):
                f.write("【样本外验证 (时间切分)】\n")
                f.write("-"*60 + "\n")
                val = all_results[f'{strategy}_validation']
                if 'train' in val and 'test' in val:
                    f.write(f"训练集: N={val['train']['n']}, 胜率={val['train']['win_rate']:.1f}%, 总盈亏={val['train']['total_pnl']:.2f}%\n")
                    f.write(f"测试集: N={val['test']['n']}, 胜率={val['test']['win_rate']:.1f}%, 总盈亏={val['test']['total_pnl']:.2f}%\n")
                f.write("\n")

        # 组合优化结果
        f.write("\n" + "="*80 + "\n")
        f.write("组合优化结果 (Top 10)\n")
        f.write("="*80 + "\n\n")

        for strategy in ['G-做空', 'G+做多']:
            if f'{strategy}_grid' in all_results and isinstance(all_results[f'{strategy}_grid'], pd.DataFrame):
                f.write(f"【{strategy} 最优参数组合】\n")
                f.write("-"*80 + "\n")
                grid_df = all_results[f'{strategy}_grid']
                valid_grid = grid_df[grid_df['n'] >= 10].head(10) if len(grid_df) > 0 else pd.DataFrame()
                if len(valid_grid) > 0:
                    f.write(f"{'MR':>6} | {'LB':>3} | {'TH':>6} | {'TP':>6} | {'SL':>6} | {'HZ':>3} | {'N':>4} | {'胜率':>6} | {'总盈亏':>8}\n")
                    for _, r in valid_grid.iterrows():
                        f.write(f"{r['mr_min']:>6} | {r['lb']:>3} | {r['th']:>6} | {r['tp']:>6} | {r['sl']:>6} | "
                               f"{r['hz']:>3} | {r['n']:>4} | {r['win_rate']:>5.1f}% | {r['total_pnl']:>7.2f}%\n")
                f.write("\n")

        # 深度分析结果
        f.write("\n" + "="*80 + "\n")
        f.write("深度分析 (按日分析 + Walk-Forward + 显著性检验)\n")
        f.write("="*80 + "\n\n")

        for strategy in ['G+做多', 'G-做空']:
            f.write(f"\n{'#'*60}\n")
            f.write(f"# 深度分析: {strategy}\n")
            f.write(f"{'#'*60}\n\n")

            # 按日分析
            if f'{strategy}_daily' in all_results and isinstance(all_results[f'{strategy}_daily'], pd.DataFrame):
                f.write("【按日分析】\n")
                daily_df = all_results[f'{strategy}_daily']
                worst = daily_df.nsmallest(3, 'total_pnl')
                best = daily_df.nlargest(3, 'total_pnl')
                f.write("最差3天:\n")
                for _, row in worst.iterrows():
                    f.write(f"  {row['date']}: {row['n_trades']}笔, 盈亏={row['total_pnl']:.2f}%\n")
                f.write("最好3天:\n")
                for _, row in best.iterrows():
                    f.write(f"  {row['date']}: {row['n_trades']}笔, 盈亏={row['total_pnl']:.2f}%\n")
                f.write("\n")

            # Walk-Forward验证
            if f'{strategy}_wf' in all_results and isinstance(all_results[f'{strategy}_wf'], pd.DataFrame):
                f.write("【Walk-Forward验证】\n")
                wf_df = all_results[f'{strategy}_wf']
                total_test_pnl = wf_df['test_pnl'].sum()
                total_test_n = wf_df['test_n'].sum()
                f.write(f"总测试交易: {total_test_n}, 总测试盈亏: {total_test_pnl:.2f}%\n")
                f.write("\n")

            # 改进版时间切分
            if f'{strategy}_validation_v2' in all_results and all_results[f'{strategy}_validation_v2']:
                f.write("【改进版时间切分验证】\n")
                val_v2 = all_results[f'{strategy}_validation_v2']
                if 'best_params' in val_v2 and val_v2['best_params']:
                    bp = val_v2['best_params']
                    f.write(f"最优参数: MR>={bp['mr_min']*100:.0f}%, LB={bp['lb']}, TP={bp['tp']*100:.2f}%, SL={bp['sl']*100:.2f}%\n")
                    f.write(f"训练集: N={bp['train_n']}, 总盈亏={bp['train_pnl']:.2f}%\n")
                if 'test_stats' in val_v2:
                    ts = val_v2['test_stats']
                    f.write(f"测试集: N={ts['n']}, 胜率={ts['win_rate']:.1f}%, 总盈亏={ts['total_pnl']:.2f}%\n")
                f.write("\n")

            # 显著性检验
            if f'{strategy}_significance' in all_results and all_results[f'{strategy}_significance']:
                f.write("【综合显著性检验】\n")
                sig = all_results[f'{strategy}_significance']
                if 'stats' in sig:
                    st = sig['stats']
                    f.write(f"基础统计: N={st['n']}, 胜率={st['win_rate']:.1f}%, 总盈亏={st['total_pnl']:.2f}%\n")
                    if 'sharpe' in st:
                        f.write(f"Sharpe={st['sharpe']}, 最大连续亏损={st.get('max_consec_loss', 'N/A')}\n")
                if 'bootstrap' in sig:
                    bs = sig['bootstrap']
                    f.write(f"Bootstrap 95% CI: [{bs['lower']:.4f}%, {bs['upper']:.4f}%]\n")
                    f.write(f"Bootstrap显著性: {'✓ 显著' if bs['significant'] else '✗ 不显著'}\n")
                if 'monte_carlo' in sig and sig['monte_carlo']:
                    mc = sig['monte_carlo']
                    f.write(f"Monte Carlo p-value: {mc['p_value_total']:.4f}\n")
                    f.write(f"Monte Carlo显著性: {'✓ p<0.05' if mc['significant_total'] else '✗ p>=0.05'}\n")
                if 'overall_significant' in sig:
                    f.write(f"【综合结论】: {'✓ 策略表现显著' if sig['overall_significant'] else '✗ 策略表现不显著'}\n")
                f.write("\n")

        # 最终总结
        f.write("\n" + "="*80 + "\n")
        f.write("最终总结\n")
        f.write("="*80 + "\n\n")

        f.write("各策略显著性汇总:\n")
        f.write("-"*60 + "\n")
        for strategy in strategies:
            sig_result = all_results.get(f'{strategy}_significance')
            if sig_result and 'overall_significant' in sig_result:
                status = "✓ 显著" if sig_result['overall_significant'] else "✗ 不显著"
                stats = sig_result.get('stats', {})
                f.write(f"{strategy}: {status} (N={stats.get('n', 0)}, 总盈亏={stats.get('total_pnl', 0):.2f}%)\n")
            else:
                f.write(f"{strategy}: 未进行深度检验\n")

        f.write("\n注意: 只有同时通过Bootstrap置信区间检验和Monte Carlo显著性检验的策略才被标记为'显著'\n")

    print(f"\n\n" + "="*80)
    print(f"测试完成！所有结果已保存到: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
