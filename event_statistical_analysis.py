"""
SPX 事件驱动统计分析框架
================================================================================

核心问题:
"事件发生后（过去X分钟涨/跌X%），未来X分钟价格变化是否有可预测性？"
"不同条件（Gamma、MR、距离关键位）下，是否有统计显著差异？"

分析流程:
1. 事件定义 → 2. 条件分组 → 3. 结果测量 → 4. 统计检验

这是纯统计分析，不涉及TP/SL策略回测。
"""

import os
import yaml
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# ============ 数据加载 ============
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

    return df


def convert_to_5min(df):
    """将1分钟数据转换为5分钟数据"""
    df = df.copy()
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
    """准备数据 - 添加gamma环境、move ratio、距离等"""
    df = df.copy()

    # 添加 key levels
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


def create_analysis_groups(df):
    """创建分析用的分组变量"""
    df = df.copy()

    # MR分组
    df['mr_group'] = pd.cut(
        df['move_ratio'],
        bins=[-0.001, 0.3, 0.5, 0.7, 1.0, float('inf')],
        labels=['<30%', '30-50%', '50-70%', '70-100%', '>100%']
    )

    # MR简化分组 (用于二分类检验)
    df['mr_high'] = df['move_ratio'] >= 0.5

    # 距离Hedge Wall分组
    df['dist_hw_group'] = pd.cut(
        df['dist_hedge_wall'],
        bins=[-0.001, 0.001, 0.003, 0.005, float('inf')],
        labels=['<0.1%', '0.1-0.3%', '0.3-0.5%', '>0.5%']
    )

    # 时间段分组
    def time_period(minutes):
        if minutes < 30:
            return '1_开盘30min'
        elif minutes < 60:
            return '2_30-60min'
        elif minutes < 120:
            return '3_60-120min'
        elif minutes < 240:
            return '4_午盘'
        else:
            return '5_尾盘'

    df['time_period'] = df['minutes_from_open'].apply(time_period)

    # 价格相对于开盘的位置
    df['price_vs_open'] = (df['close'] - df['daily_open']) / df['daily_open']
    df['above_open'] = df['price_vs_open'] > 0

    return df


# ============ 事件分析器 ============
class EventAnalyzer:
    """事件驱动的价格变化统计分析"""

    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.events = None
        self.event_params = {}

    def define_events(self, lookback=15, threshold=0.001, direction='drop',
                      min_minutes_from_open=30):
        """
        定义事件

        Args:
            lookback: 回看周期 (5分钟bar数，如3=15分钟)
            threshold: 变化阈值 (0.001 = 0.1%)
            direction: 'drop' 或 'rise'
            min_minutes_from_open: 排除开盘后多少分钟
        """
        df = self.df.copy()

        # 计算过去收益
        df['past_return'] = df['close'].pct_change(lookback)

        # 定义事件
        if direction == 'drop':
            mask = df['past_return'] <= -threshold
            event_desc = f"下跌>={threshold*100:.2f}%"
        else:
            mask = df['past_return'] >= threshold
            event_desc = f"上涨>={threshold*100:.2f}%"

        # 过滤开盘时间
        mask = mask & (df['minutes_from_open'] >= min_minutes_from_open)

        # 计算未来收益 (多个horizon)
        for bars in [1, 2, 3, 6, 12]:  # 5, 10, 15, 30, 60分钟
            minutes = bars * 5
            df[f'forward_{minutes}min'] = df['close'].shift(-bars) / df['close'] - 1

        # 计算MFE/MAE (未来15分钟 = 3 bars)
        horizon_bars = 3
        mfe_list = []
        mae_list = []

        for idx in range(len(df)):
            if idx + horizon_bars >= len(df):
                mfe_list.append(np.nan)
                mae_list.append(np.nan)
                continue

            entry_price = df.loc[idx, 'close']
            future_slice = df.loc[idx+1:idx+horizon_bars]

            if len(future_slice) == 0:
                mfe_list.append(np.nan)
                mae_list.append(np.nan)
                continue

            # MFE: 最大有利偏移 (做多假设)
            max_high = future_slice['high'].max()
            mfe = (max_high - entry_price) / entry_price

            # MAE: 最大不利偏移 (做多假设)
            min_low = future_slice['low'].min()
            mae = (entry_price - min_low) / entry_price

            mfe_list.append(mfe)
            mae_list.append(mae)

        df['mfe_15min'] = mfe_list
        df['mae_15min'] = mae_list

        # 保存事件
        self.events = df[mask].copy()
        self.event_params = {
            'lookback_bars': lookback,
            'lookback_minutes': lookback * 5,
            'threshold': threshold,
            'direction': direction,
        }

        print(f"\n{'='*70}")
        print(f"事件定义: 过去{lookback*5}分钟内 {event_desc}")
        print(f"总事件数: {len(self.events)}")
        print(f"覆盖天数: {self.events['date'].nunique()}")
        print(f"{'='*70}")

        return self.events

    def basic_stats(self, outcome_col='forward_15min'):
        """基础统计描述"""
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        data = self.events[outcome_col].dropna()

        if len(data) < 5:
            print("数据不足")
            return None

        print(f"\n{'='*70}")
        print(f"基础统计: {outcome_col}")
        print(f"{'='*70}")
        print(f"  样本数: {len(data)}")
        print(f"  均值: {data.mean()*100:.4f}%")
        print(f"  中位数: {data.median()*100:.4f}%")
        print(f"  标准差: {data.std()*100:.4f}%")
        print(f"  正向比例: {(data > 0).mean()*100:.1f}%")
        print(f"  最小值: {data.min()*100:.4f}%")
        print(f"  最大值: {data.max()*100:.4f}%")

        # 单样本t检验
        if len(data) >= 10:
            t_stat, p_value = stats.ttest_1samp(data, 0)
            print(f"\n  单样本t检验 (H0: 均值=0)")
            print(f"  t统计量: {t_stat:.3f}")
            print(f"  p值: {p_value:.4f}")
            print(f"  结论: {'✓ 显著不为0 (p<0.05)' if p_value < 0.05 else '✗ 不显著 (p>=0.05)'}")

        return {
            'n': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'positive_pct': (data > 0).mean(),
        }

    def analyze_by_group(self, group_col, outcome_col='forward_15min', min_samples=5):
        """
        按分组分析结果

        Args:
            group_col: 分组列名
            outcome_col: 结果列名
            min_samples: 最小样本数
        """
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        results = []
        groups = self.events[group_col].dropna().unique()

        for group in sorted(groups):
            subset = self.events[self.events[group_col] == group][outcome_col].dropna()

            if len(subset) < min_samples:
                continue

            stats_dict = {
                'group': group,
                'n': len(subset),
                'mean': subset.mean() * 100,
                'std': subset.std() * 100,
                'median': subset.median() * 100,
                'positive_pct': (subset > 0).mean() * 100,
            }

            # 单样本t检验
            if len(subset) >= 10:
                t_stat, p_value = stats.ttest_1samp(subset, 0)
                stats_dict['t_stat'] = t_stat
                stats_dict['p_value'] = p_value
                stats_dict['significant'] = p_value < 0.05
            else:
                stats_dict['t_stat'] = np.nan
                stats_dict['p_value'] = np.nan
                stats_dict['significant'] = False

            results.append(stats_dict)

        results_df = pd.DataFrame(results)

        # 打印结果
        print(f"\n{'='*70}")
        print(f"分组分析: {group_col} → {outcome_col}")
        print(f"{'='*70}")
        print(f"{'组':>12} | {'N':>5} | {'均值':>8} | {'标准差':>8} | {'正向%':>7} | {'p值':>8} | {'显著':>4}")
        print("-" * 70)

        for _, r in results_df.iterrows():
            sig = '✓' if r.get('significant', False) else '✗'
            p_val = f"{r['p_value']:.4f}" if not pd.isna(r.get('p_value')) else 'N/A'
            print(f"{str(r['group']):>12} | {r['n']:>5} | {r['mean']:>7.3f}% | {r['std']:>7.3f}% | "
                  f"{r['positive_pct']:>6.1f}% | {p_val:>8} | {sig:>4}")

        return results_df

    def compare_two_groups(self, group_col, outcome_col='forward_15min'):
        """
        两组比较 (独立样本t检验 + Mann-Whitney U检验)
        """
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        groups = self.events[group_col].dropna().unique()

        if len(groups) != 2:
            print(f"警告: {group_col} 有 {len(groups)} 个组，需要恰好2个组")
            return None

        group1, group2 = sorted(groups)
        data1 = self.events[self.events[group_col] == group1][outcome_col].dropna()
        data2 = self.events[self.events[group_col] == group2][outcome_col].dropna()

        if len(data1) < 5 or len(data2) < 5:
            print("某组样本不足")
            return None

        print(f"\n{'='*70}")
        print(f"两组比较: {group_col} ({group1} vs {group2})")
        print(f"{'='*70}")
        print(f"  {group1}: N={len(data1)}, 均值={data1.mean()*100:.3f}%, 正向率={100*(data1>0).mean():.1f}%")
        print(f"  {group2}: N={len(data2)}, 均值={data2.mean()*100:.3f}%, 正向率={100*(data2>0).mean():.1f}%")

        # 独立样本t检验 (参数检验)
        t_stat, p_t = stats.ttest_ind(data1, data2)
        print(f"\n  独立样本t检验:")
        print(f"    t = {t_stat:.3f}, p = {p_t:.4f}")
        print(f"    结论: {'✓ 两组有显著差异' if p_t < 0.05 else '✗ 两组无显著差异'}")

        # Mann-Whitney U检验 (非参数检验)
        u_stat, p_u = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"\n  Mann-Whitney U检验 (非参数):")
        print(f"    U = {u_stat:.1f}, p = {p_u:.4f}")
        print(f"    结论: {'✓ 两组有显著差异' if p_u < 0.05 else '✗ 两组无显著差异'}")

        # 效应量 (Cohen's d)
        pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
        print(f"\n  效应量 (Cohen's d): {cohens_d:.3f}")
        if abs(cohens_d) < 0.2:
            print(f"    解释: 效应很小")
        elif abs(cohens_d) < 0.5:
            print(f"    解释: 效应较小")
        elif abs(cohens_d) < 0.8:
            print(f"    解释: 效应中等")
        else:
            print(f"    解释: 效应较大")

        return {
            't_test': {'statistic': t_stat, 'p_value': p_t},
            'mann_whitney': {'statistic': u_stat, 'p_value': p_u},
            'cohens_d': cohens_d
        }

    def compare_multiple_groups(self, group_col, outcome_col='forward_15min'):
        """
        多组比较 (ANOVA + Kruskal-Wallis)
        """
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        groups = self.events[group_col].dropna().unique()
        group_data = []

        for g in sorted(groups):
            data = self.events[self.events[group_col] == g][outcome_col].dropna()
            if len(data) >= 5:
                group_data.append(data.values)

        if len(group_data) < 2:
            print("有效组数不足")
            return None

        print(f"\n{'='*70}")
        print(f"多组比较: {group_col} ({len(group_data)} 组)")
        print(f"{'='*70}")

        # ANOVA (参数检验)
        f_stat, p_anova = stats.f_oneway(*group_data)
        print(f"\n  单因素ANOVA:")
        print(f"    F = {f_stat:.3f}, p = {p_anova:.4f}")
        print(f"    结论: {'✓ 组间有显著差异' if p_anova < 0.05 else '✗ 组间无显著差异'}")

        # Kruskal-Wallis (非参数检验)
        h_stat, p_kw = stats.kruskal(*group_data)
        print(f"\n  Kruskal-Wallis检验 (非参数):")
        print(f"    H = {h_stat:.3f}, p = {p_kw:.4f}")
        print(f"    结论: {'✓ 组间有显著差异' if p_kw < 0.05 else '✗ 组间无显著差异'}")

        return {
            'anova': {'statistic': f_stat, 'p_value': p_anova},
            'kruskal': {'statistic': h_stat, 'p_value': p_kw}
        }

    def correlation_analysis(self, continuous_col, outcome_col='forward_15min'):
        """
        相关性分析 (连续变量)
        """
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        valid = self.events[[continuous_col, outcome_col]].dropna()

        if len(valid) < 10:
            print("样本不足，无法计算相关性")
            return None

        x = valid[continuous_col]
        y = valid[outcome_col]

        # Pearson相关 (线性)
        r_pearson, p_pearson = stats.pearsonr(x, y)

        # Spearman相关 (单调关系，更稳健)
        r_spearman, p_spearman = stats.spearmanr(x, y)

        print(f"\n{'='*70}")
        print(f"相关性分析: {continuous_col} vs {outcome_col}")
        print(f"{'='*70}")
        print(f"  样本数: {len(valid)}")
        print(f"\n  Pearson相关 (线性关系):")
        print(f"    r = {r_pearson:.3f}, p = {p_pearson:.4f}")
        print(f"    结论: {'✓ 显著相关' if p_pearson < 0.05 else '✗ 不显著'}")
        print(f"\n  Spearman相关 (单调关系):")
        print(f"    ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
        print(f"    结论: {'✓ 显著相关' if p_spearman < 0.05 else '✗ 不显著'}")

        # 解释相关强度
        r = abs(r_spearman)
        if r < 0.1:
            strength = "几乎无相关"
        elif r < 0.3:
            strength = "弱相关"
        elif r < 0.5:
            strength = "中等相关"
        elif r < 0.7:
            strength = "较强相关"
        else:
            strength = "强相关"
        print(f"\n  相关强度: {strength}")

        return {
            'pearson': {'r': r_pearson, 'p': p_pearson},
            'spearman': {'r': r_spearman, 'p': p_spearman}
        }

    def cross_analysis(self, group1_col, group2_col, outcome_col='forward_15min'):
        """交叉分析: 两个分组变量的组合"""
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        print(f"\n{'='*70}")
        print(f"交叉分析: {group1_col} × {group2_col}")
        print(f"{'='*70}")

        groups1 = sorted(self.events[group1_col].dropna().unique())
        groups2 = sorted(self.events[group2_col].dropna().unique())

        print(f"\n{group1_col:>10} × {group2_col:>10} | {'N':>5} | {'均值':>8} | {'正向%':>7} | {'p值':>8} | {'显著':>4}")
        print("-" * 75)

        results = []
        for g1 in groups1:
            for g2 in groups2:
                subset = self.events[
                    (self.events[group1_col] == g1) &
                    (self.events[group2_col] == g2)
                ][outcome_col].dropna()

                if len(subset) < 5:
                    continue

                mean = subset.mean() * 100
                pos_pct = (subset > 0).mean() * 100

                if len(subset) >= 10:
                    _, p_val = stats.ttest_1samp(subset, 0)
                    p_str = f"{p_val:.4f}"
                    sig = '✓' if p_val < 0.05 else '✗'
                else:
                    p_str = "N/A"
                    sig = '-'
                    p_val = np.nan

                print(f"{str(g1):>10} × {str(g2):>10} | {len(subset):>5} | {mean:>7.3f}% | {pos_pct:>6.1f}% | {p_str:>8} | {sig:>4}")

                results.append({
                    group1_col: g1,
                    group2_col: g2,
                    'n': len(subset),
                    'mean': mean,
                    'positive_pct': pos_pct,
                    'p_value': p_val
                })

        return pd.DataFrame(results)

    def horizon_analysis(self, group_col=None, horizons=[5, 10, 15, 30]):
        """
        不同持仓时间的效果分析
        """
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        print(f"\n{'='*70}")
        print(f"不同Horizon效果分析")
        if group_col:
            print(f"分组: {group_col}")
        print(f"{'='*70}")

        results = []

        if group_col:
            groups = sorted(self.events[group_col].dropna().unique())
        else:
            groups = ['全部']

        for group in groups:
            if group == '全部':
                subset = self.events
            else:
                subset = self.events[self.events[group_col] == group]

            print(f"\n--- {group} ---")
            print(f"{'Horizon':>10} | {'N':>5} | {'均值':>8} | {'正向%':>7} | {'p值':>8} | {'显著':>4}")
            print("-" * 55)

            for hz in horizons:
                col = f'forward_{hz}min'
                if col not in subset.columns:
                    continue

                data = subset[col].dropna()
                if len(data) < 5:
                    continue

                mean = data.mean() * 100
                pos_pct = (data > 0).mean() * 100

                if len(data) >= 10:
                    _, p_val = stats.ttest_1samp(data, 0)
                    p_str = f"{p_val:.4f}"
                    sig = '✓' if p_val < 0.05 else '✗'
                else:
                    p_str = "N/A"
                    sig = '-'
                    p_val = np.nan

                print(f"{hz:>8}min | {len(data):>5} | {mean:>7.3f}% | {pos_pct:>6.1f}% | {p_str:>8} | {sig:>4}")

                results.append({
                    'group': group,
                    'horizon': hz,
                    'n': len(data),
                    'mean': mean,
                    'positive_pct': pos_pct,
                    'p_value': p_val
                })

        return pd.DataFrame(results)

    def mfe_mae_analysis(self, group_col=None):
        """
        MFE/MAE分析 (最大有利/不利偏移)
        用于评估信号的潜在盈利空间和风险
        """
        if self.events is None:
            raise ValueError("请先调用 define_events()")

        print(f"\n{'='*70}")
        print(f"MFE/MAE分析 (未来15分钟内的最大偏移)")
        if group_col:
            print(f"分组: {group_col}")
        print(f"{'='*70}")

        if group_col:
            groups = sorted(self.events[group_col].dropna().unique())
        else:
            groups = ['全部']

        results = []

        for group in groups:
            if group == '全部':
                subset = self.events
            else:
                subset = self.events[self.events[group_col] == group]

            mfe = subset['mfe_15min'].dropna() * 100
            mae = subset['mae_15min'].dropna() * 100

            if len(mfe) < 5:
                continue

            print(f"\n--- {group} (N={len(mfe)}) ---")
            print(f"  MFE (最大有利偏移):")
            print(f"    均值: {mfe.mean():.3f}%, 中位数: {mfe.median():.3f}%")
            print(f"    P25: {mfe.quantile(0.25):.3f}%, P75: {mfe.quantile(0.75):.3f}%")
            print(f"  MAE (最大不利偏移):")
            print(f"    均值: {mae.mean():.3f}%, 中位数: {mae.median():.3f}%")
            print(f"    P25: {mae.quantile(0.25):.3f}%, P75: {mae.quantile(0.75):.3f}%")
            print(f"  MFE/MAE比: {mfe.mean()/mae.mean():.2f}")

            results.append({
                'group': group,
                'n': len(mfe),
                'mfe_mean': mfe.mean(),
                'mfe_median': mfe.median(),
                'mae_mean': mae.mean(),
                'mae_median': mae.median(),
                'mfe_mae_ratio': mfe.mean() / mae.mean() if mae.mean() > 0 else np.nan
            })

        return pd.DataFrame(results)


# ============ 综合分析流程 ============
def run_full_analysis(df, output_file=None):
    """运行完整的统计分析"""

    # 准备分组变量
    df = create_analysis_groups(df)

    all_results = {}

    # ========== 分析1: 下跌后的价格变化 ==========
    print("\n" + "#"*80)
    print("# 分析1: 下跌事件后的价格变化")
    print("#"*80)

    analyzer_drop = EventAnalyzer(df)
    analyzer_drop.define_events(lookback=3, threshold=0.001, direction='drop')

    # 基础统计
    all_results['drop_basic'] = analyzer_drop.basic_stats('forward_15min')

    # 按Gamma分组
    all_results['drop_by_gamma'] = analyzer_drop.analyze_by_group('gamma', 'forward_15min')
    all_results['drop_gamma_compare'] = analyzer_drop.compare_two_groups('gamma', 'forward_15min')

    # 按MR分组
    all_results['drop_by_mr'] = analyzer_drop.analyze_by_group('mr_group', 'forward_15min')
    all_results['drop_mr_compare'] = analyzer_drop.compare_multiple_groups('mr_group', 'forward_15min')

    # MR相关性
    all_results['drop_mr_corr'] = analyzer_drop.correlation_analysis('move_ratio', 'forward_15min')

    # 距离HW分组
    all_results['drop_by_dist_hw'] = analyzer_drop.analyze_by_group('dist_hw_group', 'forward_15min')

    # 时间段分组
    all_results['drop_by_time'] = analyzer_drop.analyze_by_group('time_period', 'forward_15min')

    # 交叉分析: Gamma × MR
    all_results['drop_gamma_mr'] = analyzer_drop.cross_analysis('gamma', 'mr_group', 'forward_15min')

    # 不同Horizon
    all_results['drop_horizon'] = analyzer_drop.horizon_analysis('gamma')

    # MFE/MAE
    all_results['drop_mfe_mae'] = analyzer_drop.mfe_mae_analysis('gamma')

    # ========== 分析2: 上涨后的价格变化 ==========
    print("\n" + "#"*80)
    print("# 分析2: 上涨事件后的价格变化")
    print("#"*80)

    analyzer_rise = EventAnalyzer(df)
    analyzer_rise.define_events(lookback=3, threshold=0.001, direction='rise')

    # 基础统计
    all_results['rise_basic'] = analyzer_rise.basic_stats('forward_15min')

    # 按Gamma分组
    all_results['rise_by_gamma'] = analyzer_rise.analyze_by_group('gamma', 'forward_15min')
    all_results['rise_gamma_compare'] = analyzer_rise.compare_two_groups('gamma', 'forward_15min')

    # 按MR分组
    all_results['rise_by_mr'] = analyzer_rise.analyze_by_group('mr_group', 'forward_15min')

    # 交叉分析
    all_results['rise_gamma_mr'] = analyzer_rise.cross_analysis('gamma', 'mr_group', 'forward_15min')

    # MFE/MAE
    all_results['rise_mfe_mae'] = analyzer_rise.mfe_mae_analysis('gamma')

    # ========== 分析3: 参数敏感性 ==========
    print("\n" + "#"*80)
    print("# 分析3: 参数敏感性分析")
    print("#"*80)

    sensitivity_results = sensitivity_analysis(df)
    all_results['sensitivity'] = sensitivity_results

    # ========== 保存报告 ==========
    if output_file:
        save_report(all_results, output_file, df)

    return all_results


def sensitivity_analysis(df):
    """
    敏感性分析: 测试不同lookback和threshold组合
    """
    print(f"\n{'='*70}")
    print(f"敏感性分析: 不同参数组合的显著性")
    print(f"{'='*70}")

    lookbacks = [1, 2, 3, 6]  # bars (5, 10, 15, 30 min)
    thresholds = [0.0005, 0.001, 0.0015, 0.002]  # 0.05%, 0.1%, 0.15%, 0.2%

    results = []

    print(f"\n{'LB(min)':>8} | {'阈值':>6} | {'事件数':>6} | {'G+均值':>8} | {'G+p':>8} | {'G-均值':>8} | {'G-p':>8} | {'差异p':>8}")
    print("-" * 85)

    for lb in lookbacks:
        for th in thresholds:
            analyzer = EventAnalyzer(df)
            events = analyzer.define_events(lookback=lb, threshold=th, direction='drop')

            if len(events) < 20:
                continue

            # G+ 组
            gp = events[events['gamma'] == 'positive']['forward_15min'].dropna()
            # G- 组
            gn = events[events['gamma'] == 'negative']['forward_15min'].dropna()

            if len(gp) < 5 or len(gn) < 5:
                continue

            gp_mean = gp.mean() * 100
            gn_mean = gn.mean() * 100

            _, gp_p = stats.ttest_1samp(gp, 0) if len(gp) >= 10 else (np.nan, np.nan)
            _, gn_p = stats.ttest_1samp(gn, 0) if len(gn) >= 10 else (np.nan, np.nan)
            _, diff_p = stats.ttest_ind(gp, gn) if len(gp) >= 5 and len(gn) >= 5 else (np.nan, np.nan)

            gp_p_str = f"{gp_p:.4f}" if not np.isnan(gp_p) else "N/A"
            gn_p_str = f"{gn_p:.4f}" if not np.isnan(gn_p) else "N/A"
            diff_p_str = f"{diff_p:.4f}" if not np.isnan(diff_p) else "N/A"

            print(f"{lb*5:>6}min | {th*100:>5.2f}% | {len(events):>6} | {gp_mean:>7.3f}% | {gp_p_str:>8} | {gn_mean:>7.3f}% | {gn_p_str:>8} | {diff_p_str:>8}")

            results.append({
                'lookback_min': lb * 5,
                'threshold': th,
                'n_events': len(events),
                'gp_n': len(gp),
                'gp_mean': gp_mean,
                'gp_p': gp_p,
                'gn_n': len(gn),
                'gn_mean': gn_mean,
                'gn_p': gn_p,
                'diff_p': diff_p,
                'gp_significant': gp_p < 0.05 if not np.isnan(gp_p) else False,
                'gn_significant': gn_p < 0.05 if not np.isnan(gn_p) else False,
                'diff_significant': diff_p < 0.05 if not np.isnan(diff_p) else False,
            })

    results_df = pd.DataFrame(results)

    # 找出显著的组合
    sig_results = results_df[
        results_df['gp_significant'] |
        results_df['gn_significant'] |
        results_df['diff_significant']
    ]

    if len(sig_results) > 0:
        print(f"\n【统计显著的参数组合】")
        for _, r in sig_results.iterrows():
            notes = []
            if r['gp_significant']:
                notes.append(f"G+显著(p={r['gp_p']:.3f})")
            if r['gn_significant']:
                notes.append(f"G-显著(p={r['gn_p']:.3f})")
            if r['diff_significant']:
                notes.append(f"G+/G-差异显著(p={r['diff_p']:.3f})")
            print(f"  LB={r['lookback_min']}min, TH={r['threshold']*100:.2f}%: {', '.join(notes)}")
    else:
        print(f"\n没有找到统计显著的参数组合")

    return results_df


def save_report(results, output_file, df):
    """保存分析报告"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SPX事件驱动统计分析报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据日期: {df['date'].min()} ~ {df['date'].max()} ({df['date'].nunique()} 天)\n")
        f.write(f"总数据点: {len(df)}\n")
        f.write("\n")

        # 核心发现
        f.write("="*80 + "\n")
        f.write("核心发现摘要\n")
        f.write("="*80 + "\n\n")

        # 下跌事件Gamma差异
        if 'drop_gamma_compare' in results and results['drop_gamma_compare']:
            r = results['drop_gamma_compare']
            f.write("【下跌后 G+ vs G- 差异】\n")
            f.write(f"  t检验 p值: {r['t_test']['p_value']:.4f}\n")
            f.write(f"  Mann-Whitney p值: {r['mann_whitney']['p_value']:.4f}\n")
            f.write(f"  Cohen's d: {r['cohens_d']:.3f}\n")
            if r['t_test']['p_value'] < 0.05:
                f.write("  结论: ✓ G+和G-下跌后的反应有显著差异\n")
            else:
                f.write("  结论: ✗ G+和G-下跌后的反应无显著差异\n")
            f.write("\n")

        # MR相关性
        if 'drop_mr_corr' in results and results['drop_mr_corr']:
            r = results['drop_mr_corr']
            f.write("【Move Ratio与未来收益的相关性】\n")
            f.write(f"  Spearman相关: r={r['spearman']['r']:.3f}, p={r['spearman']['p']:.4f}\n")
            if r['spearman']['p'] < 0.05:
                direction = "正" if r['spearman']['r'] > 0 else "负"
                f.write(f"  结论: ✓ MR与未来收益有显著{direction}相关\n")
            else:
                f.write("  结论: ✗ MR与未来收益无显著相关\n")
            f.write("\n")

        # 敏感性分析结果
        if 'sensitivity' in results and isinstance(results['sensitivity'], pd.DataFrame):
            sig = results['sensitivity'][
                results['sensitivity']['gp_significant'] |
                results['sensitivity']['diff_significant']
            ]
            f.write("【显著的参数组合】\n")
            if len(sig) > 0:
                for _, row in sig.iterrows():
                    f.write(f"  LB={row['lookback_min']}min, TH={row['threshold']*100:.2f}%\n")
            else:
                f.write("  没有找到显著的参数组合\n")
            f.write("\n")

        f.write("\n" + "="*80 + "\n")
        f.write("详细分析结果已输出到控制台\n")
        f.write("="*80 + "\n")

    print(f"\n报告已保存到: {output_file}")


# ============ 主程序 ============
def main():
    print("="*80)
    print("SPX 事件驱动统计分析")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    key_levels = load_key_levels('data/key_levels')
    df = load_price_data('data/spx')
    df = convert_to_5min(df)
    df = prepare_data(df, key_levels)
    df = create_analysis_groups(df)

    print(f"  数据日期: {df['date'].min()} ~ {df['date'].max()} ({df['date'].nunique()} 天)")
    print(f"  总数据点: {len(df)}")

    # 检查Gamma分布
    gamma_counts = df.groupby('date')['gamma'].first().value_counts()
    print(f"\n  Gamma分布:")
    for g, c in gamma_counts.items():
        print(f"    {g}: {c} 天")

    # 运行完整分析
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/event_statistical_analysis_{timestamp}.txt'

    results = run_full_analysis(df, output_file)

    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)

    return results


if __name__ == "__main__":
    main()
