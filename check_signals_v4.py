#!/usr/bin/env python3
"""
检查所有天的数据 - V4 版本
核心改进:
1. 移除 MR 作为必要条件（因为很多日子波动本身就小）
2. 信号条件: Gamma环境 + 标准化动量
3. MR 作为信号质量的分级指标，不是过滤条件
"""
import os
import yaml
import pandas as pd
import numpy as np
from glob import glob

# ============ 配置参数 ============
CONFIG = {
    # 动量阈值（基于数据分析优化）
    'momentum_th_g_plus': 0.13,    # G+ 环境动量阈值 0.13%
    'momentum_th_g_minus': 0.26,   # G- 环境动量阈值 0.26% (2.0x)
    'momentum_lookback': 15,       # 动量回望期（分钟）
}

def load_key_levels(date_str):
    yaml_path = f"data/key_levels/{date_str}.yaml"
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_price_data(date_str):
    csv_path = f"data/spx/SPX_{date_str}_minute.csv"
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

def analyze_day_v4(date_str, config):
    """分析单天数据 - V4版 (移除MR作为必要条件)"""
    levels = load_key_levels(date_str)
    df = load_price_data(date_str)

    if levels is None or df is None:
        return None, "数据缺失"

    hedge_wall = levels.get('hedge_wall')
    if hedge_wall is None:
        return None, "HW缺失"

    daily_open = df['open'].iloc[0]

    stats = {
        'date': date_str,
        'total_bars': len(df),
        'daily_open': daily_open,
        'hedge_wall': hedge_wall,
        'g_plus_bars': 0,
        'g_minus_bars': 0,
        # V4: 基础信号 (只需 Gamma + 动量)
        'g_plus_long': 0,   # G+ 跌后做多
        'g_plus_short': 0,  # G+ 涨后做空
        'g_minus_long': 0,  # G- 涨后做多
        'g_minus_short': 0, # G- 跌后做空
    }

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        # Gamma 环境
        is_g_plus = price > hedge_wall
        if is_g_plus:
            stats['g_plus_bars'] += 1
        else:
            stats['g_minus_bars'] += 1

        # 动量检测 - 使用标准化阈值
        if i >= config['momentum_lookback']:
            price_ago = df['close'].iloc[i - config['momentum_lookback']]
            momentum_pct = (price - price_ago) / price_ago * 100

            # 根据 Gamma 环境选择阈值
            th = config['momentum_th_g_plus'] if is_g_plus else config['momentum_th_g_minus']

            after_drop = momentum_pct < -th
            after_rise = momentum_pct > th

            # V4: 信号只需要 Gamma + 动量方向
            if is_g_plus:
                if after_drop:
                    stats['g_plus_long'] += 1
                if after_rise:
                    stats['g_plus_short'] += 1
            else:
                if after_rise:
                    stats['g_minus_long'] += 1
                if after_drop:
                    stats['g_minus_short'] += 1

    return stats, None

def main():
    yaml_files = glob("data/key_levels/*.yaml")
    dates = sorted([os.path.basename(f).replace('.yaml', '') for f in yaml_files])

    print(f"=== Gamma策略信号检测 V4 (移除MR必要条件) ===")
    print(f"信号条件: Gamma环境 + 标准化动量 (G+:{CONFIG['momentum_th_g_plus']}%, G-:{CONFIG['momentum_th_g_minus']}%)")
    print(f"共 {len(dates)} 天数据")
    print("=" * 100)

    results = []
    for date_str in dates:
        stats, error = analyze_day_v4(date_str, CONFIG)
        if error:
            print(f"{date_str}: {error}")
            continue
        results.append(stats)

    # 打印每日详情
    print(f"\n{'日期':>12} | {'G+%':>5} | {'G+L':>4} | {'G+S':>4} | {'G-L':>4} | {'G-S':>4} | {'总计':>5} | 状态")
    print("-" * 80)

    total_signals = 0
    days_with_signals = 0

    for s in results:
        g_plus_pct = s['g_plus_bars'] / s['total_bars'] * 100
        total = s['g_plus_long'] + s['g_plus_short'] + s['g_minus_long'] + s['g_minus_short']
        total_signals += total

        if total > 0:
            days_with_signals += 1
            indicator = "✓"
        else:
            indicator = "✗"

        print(f"{s['date']:>12} | {g_plus_pct:>4.0f}% | {s['g_plus_long']:>4} | {s['g_plus_short']:>4} | "
              f"{s['g_minus_long']:>4} | {s['g_minus_short']:>4} | {total:>5} | {indicator}")

    print("=" * 80)

    # 汇总统计
    print(f"\n=== V4 统计汇总 ===")
    print(f"有数据天数: {len(results)}")
    print(f"有信号天数: {days_with_signals} / {len(results)} ({days_with_signals/len(results)*100:.1f}%)")
    print(f"总信号数: {total_signals}")
    print(f"日均信号: {total_signals/len(results):.1f}")

    # 按类型分解
    print(f"\n=== 信号分解 ===")
    g_plus_long = sum(s['g_plus_long'] for s in results)
    g_plus_short = sum(s['g_plus_short'] for s in results)
    g_minus_long = sum(s['g_minus_long'] for s in results)
    g_minus_short = sum(s['g_minus_short'] for s in results)

    print(f"G+ Long (跌后做多-均值回归):  {g_plus_long:>5} ({g_plus_long/total_signals*100:.1f}%)")
    print(f"G+ Short (涨后做空-均值回归): {g_plus_short:>5} ({g_plus_short/total_signals*100:.1f}%)")
    print(f"G- Long (涨后做多-趋势延续):  {g_minus_long:>5} ({g_minus_long/total_signals*100:.1f}%)")
    print(f"G- Short (跌后做空-趋势延续): {g_minus_short:>5} ({g_minus_short/total_signals*100:.1f}%)")

if __name__ == "__main__":
    os.chdir("/Users/toni/trade related/1206version")
    main()
