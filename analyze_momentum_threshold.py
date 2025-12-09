#!/usr/bin/env python3
"""
分析最优动量阈值
1. 比较固定阈值 vs ATR标准化
2. 分别测试 G+ 和 G- 环境
3. 找出最优阈值
"""
import os
import yaml
import pandas as pd
import numpy as np
from glob import glob

def load_all_data():
    """加载所有数据"""
    yaml_files = glob("data/key_levels/*.yaml")
    dates = sorted([os.path.basename(f).replace('.yaml', '') for f in yaml_files])

    all_data = []

    for date_str in dates:
        # 加载 key levels
        yaml_path = f"data/key_levels/{date_str}.yaml"
        csv_path = f"data/spx/SPX_{date_str}_minute.csv"

        if not os.path.exists(yaml_path) or not os.path.exists(csv_path):
            continue

        with open(yaml_path, 'r') as f:
            levels = yaml.safe_load(f)

        df = pd.read_csv(csv_path)

        hedge_wall = levels.get('hedge_wall')
        if hedge_wall is None:
            continue

        df['date'] = date_str
        df['hedge_wall'] = hedge_wall
        df['daily_open'] = df['open'].iloc[0]
        df['is_g_plus'] = df['close'] > hedge_wall

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def calculate_momentum_stats(df, lookback=15):
    """计算动量统计"""
    df = df.copy()

    # 计算动量
    df['momentum'] = df.groupby('date')['close'].pct_change(lookback) * 100

    # 计算 ATR (简化版: 使用 high-low)
    df['tr'] = (df['high'] - df['low']) / df['close'] * 100  # 百分比
    df['atr'] = df.groupby('date')['tr'].transform(lambda x: x.rolling(lookback).mean())

    # 标准化动量 (用 ATR)
    df['momentum_atr'] = df['momentum'] / df['atr']

    return df

def analyze_threshold_effectiveness(df, threshold_pct, use_atr=False):
    """分析特定阈值的有效性"""
    df = df.dropna(subset=['momentum', 'atr'])

    # G+ 数据
    g_plus = df[df['is_g_plus']].copy()
    # G- 数据
    g_minus = df[~df['is_g_plus']].copy()

    results = {}

    for env_name, env_df in [('G+', g_plus), ('G-', g_minus)]:
        if len(env_df) < 100:
            continue

        if use_atr:
            # ATR 标准化: threshold 是 ATR 的倍数
            after_drop = env_df['momentum_atr'] < -threshold_pct
            after_rise = env_df['momentum_atr'] > threshold_pct
        else:
            # 固定百分比阈值
            after_drop = env_df['momentum'] < -threshold_pct
            after_rise = env_df['momentum'] > threshold_pct

        # 统计触发率
        drop_rate = after_drop.sum() / len(env_df) * 100
        rise_rate = after_rise.sum() / len(env_df) * 100
        trigger_rate = (drop_rate + rise_rate)

        results[env_name] = {
            'total_bars': len(env_df),
            'drop_triggers': after_drop.sum(),
            'rise_triggers': after_rise.sum(),
            'drop_rate': drop_rate,
            'rise_rate': rise_rate,
            'trigger_rate': trigger_rate,
        }

    return results

def main():
    print("加载数据...")
    df = load_all_data()
    print(f"总数据: {len(df)} 行, {df['date'].nunique()} 天")

    print("\n计算动量和ATR...")
    df = calculate_momentum_stats(df, lookback=15)

    # 基础统计
    g_plus = df[df['is_g_plus']].dropna(subset=['momentum'])
    g_minus = df[~df['is_g_plus']].dropna(subset=['momentum'])

    print(f"\n=== 动量分布统计 ===")
    print(f"G+ 样本: {len(g_plus)}")
    print(f"  动量 std: {g_plus['momentum'].std():.4f}%")
    print(f"  动量 abs mean: {g_plus['momentum'].abs().mean():.4f}%")
    print(f"  ATR mean: {g_plus['atr'].mean():.4f}%")

    print(f"\nG- 样本: {len(g_minus)}")
    print(f"  动量 std: {g_minus['momentum'].std():.4f}%")
    print(f"  动量 abs mean: {g_minus['momentum'].abs().mean():.4f}%")
    print(f"  ATR mean: {g_minus['atr'].mean():.4f}%")

    # 计算实际波动比
    g_plus_volatility = g_plus['momentum'].std()
    g_minus_volatility = g_minus['momentum'].std()
    actual_ratio = g_minus_volatility / g_plus_volatility
    print(f"\n实际 G-/G+ 动量标准差比: {actual_ratio:.2f}x")

    # ATR 比较
    g_plus_atr = g_plus['atr'].mean()
    g_minus_atr = g_minus['atr'].mean()
    atr_ratio = g_minus_atr / g_plus_atr
    print(f"实际 G-/G+ ATR比: {atr_ratio:.2f}x")

    # ========== 测试不同阈值 ==========
    print(f"\n{'='*80}")
    print("固定百分比阈值测试 (目标: 触发率 10-20%)")
    print(f"{'='*80}")

    print(f"\n{'阈值':>8} | {'G+ Drop%':>10} | {'G+ Rise%':>10} | {'G+ Total%':>10} | {'G- Drop%':>10} | {'G- Rise%':>10} | {'G- Total%':>10}")
    print("-" * 90)

    for th in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        results = analyze_threshold_effectiveness(df, th, use_atr=False)
        if 'G+' in results and 'G-' in results:
            gp = results['G+']
            gm = results['G-']
            print(f"{th:>7.2f}% | {gp['drop_rate']:>9.1f}% | {gp['rise_rate']:>9.1f}% | {gp['trigger_rate']:>9.1f}% | "
                  f"{gm['drop_rate']:>9.1f}% | {gm['rise_rate']:>9.1f}% | {gm['trigger_rate']:>9.1f}%")

    # ========== ATR 标准化测试 ==========
    print(f"\n{'='*80}")
    print("ATR 标准化阈值测试 (阈值 = N x ATR)")
    print(f"{'='*80}")

    print(f"\n{'ATR倍数':>8} | {'G+ Drop%':>10} | {'G+ Rise%':>10} | {'G+ Total%':>10} | {'G- Drop%':>10} | {'G- Rise%':>10} | {'G- Total%':>10}")
    print("-" * 90)

    for th in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        results = analyze_threshold_effectiveness(df, th, use_atr=True)
        if 'G+' in results and 'G-' in results:
            gp = results['G+']
            gm = results['G-']
            print(f"{th:>7.1f}x | {gp['drop_rate']:>9.1f}% | {gp['rise_rate']:>9.1f}% | {gp['trigger_rate']:>9.1f}% | "
                  f"{gm['drop_rate']:>9.1f}% | {gm['rise_rate']:>9.1f}% | {gm['trigger_rate']:>9.1f}%")

    # ========== 找出等效触发率的阈值 ==========
    print(f"\n{'='*80}")
    print("等效触发率分析 (找出 G+ 和 G- 触发率相近的阈值组合)")
    print(f"{'='*80}")

    target_trigger_rate = 15  # 目标触发率 15%

    # 对于 G+，找到接近 15% 触发率的阈值
    best_gp_th = None
    best_gp_diff = float('inf')
    for th in np.arange(0.05, 0.30, 0.01):
        results = analyze_threshold_effectiveness(df, th, use_atr=False)
        if 'G+' in results:
            diff = abs(results['G+']['trigger_rate'] - target_trigger_rate)
            if diff < best_gp_diff:
                best_gp_diff = diff
                best_gp_th = th
                best_gp_rate = results['G+']['trigger_rate']

    # 对于 G-，找到接近 15% 触发率的阈值
    best_gm_th = None
    best_gm_diff = float('inf')
    for th in np.arange(0.05, 0.40, 0.01):
        results = analyze_threshold_effectiveness(df, th, use_atr=False)
        if 'G-' in results:
            diff = abs(results['G-']['trigger_rate'] - target_trigger_rate)
            if diff < best_gm_diff:
                best_gm_diff = diff
                best_gm_th = th
                best_gm_rate = results['G-']['trigger_rate']

    print(f"\n目标触发率: {target_trigger_rate}%")
    print(f"G+ 最优阈值: {best_gp_th:.2f}% (实际触发率: {best_gp_rate:.1f}%)")
    print(f"G- 最优阈值: {best_gm_th:.2f}% (实际触发率: {best_gm_rate:.1f}%)")
    print(f"实际 G-/G+ 阈值比: {best_gm_th/best_gp_th:.2f}x")

    # ATR 方式
    print(f"\n使用 ATR 标准化 (目标触发率 {target_trigger_rate}%):")
    best_atr_th = None
    best_atr_diff = float('inf')
    for th in np.arange(0.5, 3.0, 0.1):
        results = analyze_threshold_effectiveness(df, th, use_atr=True)
        if 'G+' in results and 'G-' in results:
            # ATR 的优势是 G+ 和 G- 触发率应该相近
            avg_rate = (results['G+']['trigger_rate'] + results['G-']['trigger_rate']) / 2
            rate_diff = abs(results['G+']['trigger_rate'] - results['G-']['trigger_rate'])
            target_diff = abs(avg_rate - target_trigger_rate)

            # 综合评分: 接近目标 + G+/G- 触发率相近
            score = target_diff + rate_diff * 0.5
            if score < best_atr_diff:
                best_atr_diff = score
                best_atr_th = th
                best_atr_gp = results['G+']['trigger_rate']
                best_atr_gm = results['G-']['trigger_rate']

    print(f"最优 ATR 倍数: {best_atr_th:.1f}x")
    print(f"  G+ 触发率: {best_atr_gp:.1f}%")
    print(f"  G- 触发率: {best_atr_gm:.1f}%")

    # ========== 结论 ==========
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")
    print(f"""
1. 固定阈值方式:
   - G+ 推荐: {best_gp_th:.2f}%
   - G- 推荐: {best_gm_th:.2f}%
   - 比例: {best_gm_th/best_gp_th:.2f}x (vs 我们之前用的 1.8x)

2. ATR 标准化方式:
   - 统一使用: {best_atr_th:.1f}x ATR
   - 优点: 自动适应不同波动环境
   - 缺点: 需要额外计算 ATR

3. 建议:
   - 如果追求简单: 使用固定阈值 G+={best_gp_th:.2f}%, G-={best_gm_th:.2f}%
   - 如果追求准确: 使用 ATR 标准化 {best_atr_th:.1f}x
""")

if __name__ == "__main__":
    os.chdir("/Users/toni/trade related/1206version")
    main()
