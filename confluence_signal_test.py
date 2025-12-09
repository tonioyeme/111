"""
Confluence Score Signal Testing Script

This script tests how SPX price moves after confluence signals are generated
during the 9:30-11:00 AM window.

Instead of traditional P&L/win rate backtesting, we measure:
1. Price change 5, 10, 15, 30 minutes after signal
2. Max favorable excursion (MFE) - how far price moved in signal direction
3. Max adverse excursion (MAE) - how far price moved against signal
4. Whether price eventually reached certain targets (0.1%, 0.2%, 0.5%)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SIMPLIFIED SMC/ICT DETECTORS
# ============================================================================

class SimplifiedSMCAnalyzer:
    """Simplified SMC analysis for testing purposes."""

    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback

    def detect_swing_points(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Detect swing highs and lows."""
        swing_highs = []
        swing_lows = []

        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            # Check for swing high
            is_swing_high = True
            is_swing_low = True

            for j in range(1, self.swing_lookback + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j] or \
                   data['high'].iloc[i] <= data['high'].iloc[i + j]:
                    is_swing_high = False
                if data['low'].iloc[i] >= data['low'].iloc[i - j] or \
                   data['low'].iloc[i] >= data['low'].iloc[i + j]:
                    is_swing_low = False

            if is_swing_high:
                swing_highs.append({
                    'timestamp': data.index[i],
                    'price': data['high'].iloc[i]
                })
            if is_swing_low:
                swing_lows.append({
                    'timestamp': data.index[i],
                    'price': data['low'].iloc[i]
                })

        return swing_highs, swing_lows

    def detect_market_structure(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> Dict:
        """Analyze market structure from swing points."""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'ranging', 'strength': 'weak'}

        # Get last 2 swings
        hh = swing_highs[-1]['price'] > swing_highs[-2]['price']  # Higher High
        hl = swing_lows[-1]['price'] > swing_lows[-2]['price']    # Higher Low
        lh = swing_highs[-1]['price'] < swing_highs[-2]['price']  # Lower High
        ll = swing_lows[-1]['price'] < swing_lows[-2]['price']    # Lower Low

        # Check for conflicting signals
        bullish_signals = sum([hh, hl])
        bearish_signals = sum([lh, ll])

        if bullish_signals > 0 and bearish_signals > 0:
            return {'trend': 'ranging', 'strength': 'weak'}
        elif hh and hl:
            return {'trend': 'bullish', 'strength': 'strong'}
        elif hh or hl:
            return {'trend': 'bullish', 'strength': 'weak'}
        elif lh and ll:
            return {'trend': 'bearish', 'strength': 'strong'}
        elif lh or ll:
            return {'trend': 'bearish', 'strength': 'weak'}
        else:
            return {'trend': 'ranging', 'strength': 'weak'}

    def detect_order_blocks(self, data: pd.DataFrame, swing_highs: List[Dict],
                           swing_lows: List[Dict]) -> List[Dict]:
        """Detect order blocks at swing points."""
        order_blocks = []
        current_price = data['close'].iloc[-1]

        # Find OBs at swing lows (bullish)
        for swing in swing_lows[-10:]:
            try:
                idx = data.index.get_loc(swing['timestamp'])
            except:
                continue
            if idx < 2 or idx >= len(data) - 1:
                continue

            # Look for bearish candle before swing low
            for i in range(idx - 1, max(0, idx - 5), -1):
                if data['close'].iloc[i] < data['open'].iloc[i]:  # Bearish candle
                    ob = {
                        'type': 'bullish',
                        'high': data['high'].iloc[i],
                        'low': data['low'].iloc[i],
                        'strength': 0.5,
                        'mitigated': current_price < data['low'].iloc[i] * 0.998
                    }
                    order_blocks.append(ob)
                    break

        # Find OBs at swing highs (bearish)
        for swing in swing_highs[-10:]:
            try:
                idx = data.index.get_loc(swing['timestamp'])
            except:
                continue
            if idx < 2 or idx >= len(data) - 1:
                continue

            # Look for bullish candle before swing high
            for i in range(idx - 1, max(0, idx - 5), -1):
                if data['close'].iloc[i] > data['open'].iloc[i]:  # Bullish candle
                    ob = {
                        'type': 'bearish',
                        'high': data['high'].iloc[i],
                        'low': data['low'].iloc[i],
                        'strength': 0.5,
                        'mitigated': current_price > data['high'].iloc[i] * 1.002
                    }
                    order_blocks.append(ob)
                    break

        return [ob for ob in order_blocks if not ob['mitigated']]

    def detect_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """Detect fair value gaps (imbalances)."""
        fvgs = []
        current_price = data['close'].iloc[-1]

        for i in range(2, len(data) - 1):
            # Bullish FVG
            gap_size = data['low'].iloc[i + 1] - data['high'].iloc[i - 1]
            if gap_size > 0 and gap_size / data['close'].iloc[i] > 0.0005:
                fvg = {
                    'type': 'bullish',
                    'high': data['low'].iloc[i + 1],
                    'low': data['high'].iloc[i - 1],
                    'fill_percentage': 0.0
                }
                # Check if filled
                if current_price <= fvg['low']:
                    fvg['fill_percentage'] = 1.0
                elif current_price < fvg['high']:
                    fvg['fill_percentage'] = (fvg['high'] - current_price) / gap_size
                fvgs.append(fvg)

            # Bearish FVG
            gap_size = data['low'].iloc[i - 1] - data['high'].iloc[i + 1]
            if gap_size > 0 and gap_size / data['close'].iloc[i] > 0.0005:
                fvg = {
                    'type': 'bearish',
                    'high': data['low'].iloc[i - 1],
                    'low': data['high'].iloc[i + 1],
                    'fill_percentage': 0.0
                }
                if current_price >= fvg['high']:
                    fvg['fill_percentage'] = 1.0
                elif current_price > fvg['low']:
                    fvg['fill_percentage'] = (current_price - fvg['low']) / gap_size
                fvgs.append(fvg)

        return [fvg for fvg in fvgs if fvg['fill_percentage'] < 0.5]


class SimplifiedJudasDetector:
    """Simplified Judas Swing detector."""

    def __init__(self, lookback: int = 15, min_move: float = 0.001, reversal_threshold: float = 0.4):
        self.lookback = lookback
        self.min_move = min_move
        self.reversal_threshold = reversal_threshold

    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect Judas Swing pattern."""
        if len(data) < self.lookback + 5:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        lookback_data = data.tail(self.lookback + 5)

        # Try bullish Judas (fake breakdown then reversal up)
        bullish = self._detect_bullish(lookback_data)
        bearish = self._detect_bearish(lookback_data)

        if bullish['detected'] and bearish['detected']:
            return bullish if bullish['confidence'] >= bearish['confidence'] else bearish
        elif bullish['detected']:
            return bullish
        elif bearish['detected']:
            return bearish

        return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

    def _detect_bullish(self, data: pd.DataFrame) -> Dict:
        """Detect bullish Judas (bearish_judas pattern type -> signals bullish move)."""
        scan = data.head(self.lookback)

        # Find swing high then low
        high_idx = scan['high'].idxmax()
        high_price = scan.loc[high_idx, 'high']

        # Look for low after high
        after_high = scan.loc[high_idx:]
        if len(after_high) < 3:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        low_idx = after_high['low'].idxmin()
        low_price = after_high.loc[low_idx, 'low']

        # Check initial move
        initial_move = (high_price - low_price) / high_price
        if initial_move < self.min_move:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        # Check for reversal
        after_low = data.loc[low_idx:]
        if len(after_low) < 2:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        reversal_high = after_low['high'].max()
        retracement = (reversal_high - low_price) / (high_price - low_price) if high_price != low_price else 0

        if retracement >= self.reversal_threshold:
            confidence = min(0.5 + (retracement - 0.4) * 0.5, 0.9)
            return {
                'detected': True,
                'pattern_type': 'bearish_judas',  # Bearish fake -> Bullish reversal
                'confidence': confidence,
                'volume_confirmed': False,
                'clean_pattern': True,
                'sustained_move': retracement > 0.6
            }

        return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

    def _detect_bearish(self, data: pd.DataFrame) -> Dict:
        """Detect bearish Judas (bullish_judas pattern type -> signals bearish move)."""
        scan = data.head(self.lookback)

        # Find swing low then high
        low_idx = scan['low'].idxmin()
        low_price = scan.loc[low_idx, 'low']

        # Look for high after low
        after_low = scan.loc[low_idx:]
        if len(after_low) < 3:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        high_idx = after_low['high'].idxmax()
        high_price = after_low.loc[high_idx, 'high']

        # Check initial move
        initial_move = (high_price - low_price) / low_price
        if initial_move < self.min_move:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        # Check for reversal
        after_high = data.loc[high_idx:]
        if len(after_high) < 2:
            return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}

        reversal_low = after_high['low'].min()
        retracement = (high_price - reversal_low) / (high_price - low_price) if high_price != low_price else 0

        if retracement >= self.reversal_threshold:
            confidence = min(0.5 + (retracement - 0.4) * 0.5, 0.9)
            return {
                'detected': True,
                'pattern_type': 'bullish_judas',  # Bullish fake -> Bearish reversal
                'confidence': confidence,
                'volume_confirmed': False,
                'clean_pattern': True,
                'sustained_move': retracement > 0.6
            }

        return {'detected': False, 'pattern_type': 'none', 'confidence': 0.0}


class SimplifiedRejectionDetector:
    """Simplified rejection pattern detector."""

    def __init__(self, min_wick_ratio: float = 2.0, lookback: int = 5):
        self.min_wick_ratio = min_wick_ratio
        self.lookback = lookback

    def detect(self, data: pd.DataFrame) -> Dict:
        """Detect rejection patterns."""
        if len(data) < self.lookback:
            return {'detected': False, 'rejection_type': 'none', 'confidence': 0.0}

        recent = data.tail(self.lookback)

        # Find candle with best rejection wick
        best_bullish = None
        best_bearish = None

        for idx in recent.index:
            row = recent.loc[idx]
            body_top = max(row['open'], row['close'])
            body_bottom = min(row['open'], row['close'])
            body_size = max(body_top - body_bottom, 0.0001)

            lower_wick = body_bottom - row['low']
            upper_wick = row['high'] - body_top

            lower_ratio = lower_wick / body_size
            upper_ratio = upper_wick / body_size

            if lower_ratio >= self.min_wick_ratio:
                if best_bullish is None or lower_ratio > best_bullish['wick_ratio']:
                    best_bullish = {
                        'rejection_type': 'bullish',
                        'wick_ratio': lower_ratio,
                        'price': row['low'],
                        'confidence': min(lower_ratio / 5.0, 1.0) * 0.6
                    }

            if upper_ratio >= self.min_wick_ratio:
                if best_bearish is None or upper_ratio > best_bearish['wick_ratio']:
                    best_bearish = {
                        'rejection_type': 'bearish',
                        'wick_ratio': upper_ratio,
                        'price': row['high'],
                        'confidence': min(upper_ratio / 5.0, 1.0) * 0.6
                    }

        # Return the stronger rejection
        if best_bullish and best_bearish:
            result = best_bullish if best_bullish['confidence'] >= best_bearish['confidence'] else best_bearish
        elif best_bullish:
            result = best_bullish
        elif best_bearish:
            result = best_bearish
        else:
            return {'detected': False, 'rejection_type': 'none', 'confidence': 0.0}

        return {
            'detected': True,
            'rejection_type': result['rejection_type'],
            'confidence': result['confidence'],
            'at_key_level': False,
            'volume_confirmed': False,
            'triple_rejection': False,
            'multi_candle': False
        }


# ============================================================================
# SIMPLIFIED CONFLUENCE CALCULATOR
# ============================================================================

class SimplifiedConfluenceCalculator:
    """
    Simplified confluence scoring based on the original system.

    Weights (from original):
    - Order Block: 17%
    - Judas Swing: 17%
    - Rejection: 10%
    - Fair Value Gap: 9%
    - Break of Structure: 8%
    """

    WEIGHTS = {
        'order_block': 0.17,
        'judas_swing': 0.17,
        'rejection': 0.10,
        'fair_value_gap': 0.09,
        'break_of_structure': 0.08,
    }

    def __init__(self, min_components: int = 2):
        self.min_components = min_components
        self.trump_threshold = 0.10  # Judas can signal alone if >= 10%

    def calculate(self, signal_data: Dict, direction: str) -> Tuple[float, Dict]:
        """
        Calculate confluence score for given direction.

        Returns:
            (score, breakdown)
        """
        breakdown = {}
        total_score = 0.0
        active_components = 0

        current_price = signal_data.get('current_price', 0)

        # 1. Order Block Score
        ob_score = self._score_order_blocks(
            signal_data.get('order_blocks', []),
            current_price,
            direction
        )
        breakdown['order_block'] = ob_score
        if ob_score > 0:
            total_score += ob_score
            active_components += 1

        # 2. Fair Value Gap Score
        fvg_score = self._score_fvgs(
            signal_data.get('fair_value_gaps', []),
            current_price,
            direction
        )
        breakdown['fair_value_gap'] = fvg_score
        if fvg_score > 0:
            total_score += fvg_score
            active_components += 1

        # 3. Break of Structure Score
        bos_score = self._score_market_structure(
            signal_data.get('market_structure', {}),
            direction
        )
        breakdown['break_of_structure'] = bos_score
        if bos_score > 0:
            total_score += bos_score
            active_components += 1

        # 4. Judas Swing Score
        judas_score = self._score_judas(
            signal_data.get('judas_swing', {}),
            direction
        )
        breakdown['judas_swing'] = judas_score
        if judas_score > 0:
            total_score += judas_score
            active_components += 1

        # 5. Rejection Score
        rejection_score = self._score_rejection(
            signal_data.get('rejection', {}),
            direction
        )
        breakdown['rejection'] = rejection_score
        if rejection_score > 0:
            total_score += rejection_score
            active_components += 1

        breakdown['active_components'] = active_components
        breakdown['total'] = total_score

        # Check trump component
        trump_qualified = judas_score >= self.trump_threshold

        # Min components check
        if not trump_qualified and active_components < self.min_components:
            return 0.0, breakdown

        return min(total_score, 1.0), breakdown

    def _score_order_blocks(self, order_blocks: List[Dict], current_price: float,
                           direction: str) -> float:
        """Score order blocks with distance decay."""
        max_score = 0.0
        target_type = 'bullish' if direction == 'bullish' else 'bearish'

        for ob in order_blocks:
            if ob.get('type') != target_type:
                continue
            if ob.get('mitigated', False):
                continue

            ob_mid = (ob['high'] + ob['low']) / 2
            distance = abs(current_price - ob_mid) / current_price

            # Distance decay
            if distance > 0.03:  # > 3% away
                continue
            elif distance <= 0.005:  # <= 0.5%
                multiplier = 1.0
            else:
                multiplier = max(0, 1.0 - ((distance - 0.005) / 0.025))

            strength = ob.get('strength', 0.5)
            score = self.WEIGHTS['order_block'] * strength * multiplier
            max_score = max(max_score, score)

        return max_score

    def _score_fvgs(self, fvgs: List[Dict], current_price: float, direction: str) -> float:
        """Score fair value gaps."""
        max_score = 0.0
        target_type = 'bullish' if direction == 'bullish' else 'bearish'

        for fvg in fvgs:
            if fvg.get('type') != target_type:
                continue

            fvg_mid = (fvg['high'] + fvg['low']) / 2
            distance = abs(current_price - fvg_mid) / current_price

            if distance > 0.03:
                continue
            elif distance <= 0.005:
                multiplier = 1.0
            else:
                multiplier = max(0, 1.0 - ((distance - 0.005) / 0.025))

            fill_pct = fvg.get('fill_percentage', 0)
            strength = max(0.3, 1.0 - fill_pct)
            score = self.WEIGHTS['fair_value_gap'] * strength * multiplier
            max_score = max(max_score, score)

        return max_score

    def _score_market_structure(self, market_structure: Dict, direction: str) -> float:
        """Score market structure / break of structure."""
        trend = market_structure.get('trend', '').lower()
        strength = market_structure.get('strength', 'weak')

        if direction == 'bullish' and trend == 'bullish':
            multiplier = 1.0 if strength == 'strong' else 0.5
            return self.WEIGHTS['break_of_structure'] * multiplier
        elif direction == 'bearish' and trend == 'bearish':
            multiplier = 1.0 if strength == 'strong' else 0.5
            return self.WEIGHTS['break_of_structure'] * multiplier

        return 0.0

    def _score_judas(self, judas_data: Dict, direction: str) -> float:
        """Score Judas Swing pattern."""
        if not judas_data.get('detected', False):
            return 0.0

        pattern_type = judas_data.get('pattern_type', '')
        confidence = judas_data.get('confidence', 0.5)

        # bearish_judas (fake down) -> bullish signal
        # bullish_judas (fake up) -> bearish signal
        if direction == 'bullish' and pattern_type == 'bearish_judas':
            score = self.WEIGHTS['judas_swing'] * confidence
        elif direction == 'bearish' and pattern_type == 'bullish_judas':
            score = self.WEIGHTS['judas_swing'] * confidence
        else:
            return 0.0

        # Bonuses
        if judas_data.get('sustained_move', False):
            score *= 1.1
        if judas_data.get('clean_pattern', False):
            score *= 1.05

        return min(score, self.WEIGHTS['judas_swing'])

    def _score_rejection(self, rejection_data: Dict, direction: str) -> float:
        """Score rejection pattern."""
        if not rejection_data.get('detected', False):
            return 0.0

        rejection_type = rejection_data.get('rejection_type', '')
        confidence = rejection_data.get('confidence', 0.5)

        if direction == rejection_type:
            score = self.WEIGHTS['rejection'] * confidence

            if rejection_data.get('at_key_level', False):
                score *= 1.2
            if rejection_data.get('triple_rejection', False):
                score *= 1.5

            return min(score, self.WEIGHTS['rejection'])

        return 0.0


# ============================================================================
# SIGNAL OUTCOME TRACKER
# ============================================================================

@dataclass
class SignalOutcome:
    """Tracks the outcome of a confluence signal."""
    timestamp: datetime
    signal_direction: str
    confluence_score: float
    entry_price: float

    # Price changes at different time intervals
    price_5min: Optional[float] = None
    price_10min: Optional[float] = None
    price_15min: Optional[float] = None
    price_30min: Optional[float] = None

    # Maximum favorable/adverse excursion
    max_favorable: float = 0.0
    max_adverse: float = 0.0

    # Target hits
    hit_01pct: bool = False  # 0.1% target
    hit_02pct: bool = False  # 0.2% target
    hit_05pct: bool = False  # 0.5% target

    # Signal breakdown
    breakdown: Dict = None


def track_signal_outcome(data: pd.DataFrame, signal_idx: int,
                        signal: Dict, forward_bars: int = 30) -> SignalOutcome:
    """
    Track price movement after a signal is generated.

    Args:
        data: Full day DataFrame
        signal_idx: Index position of signal in data
        signal: Signal dictionary with direction, score, etc.
        forward_bars: How many bars to track forward
    """
    entry_price = data['close'].iloc[signal_idx]
    signal_time = data.index[signal_idx]
    direction = signal['direction']
    score = signal['score']

    outcome = SignalOutcome(
        timestamp=signal_time,
        signal_direction=direction,
        confluence_score=score,
        entry_price=entry_price,
        breakdown=signal.get('breakdown', {})
    )

    # Track forward prices
    max_bars = min(forward_bars, len(data) - signal_idx - 1)
    if max_bars < 1:
        return outcome

    forward_data = data.iloc[signal_idx + 1:signal_idx + 1 + max_bars]

    # Price at specific intervals
    if len(forward_data) >= 5:
        outcome.price_5min = forward_data['close'].iloc[4]
    if len(forward_data) >= 10:
        outcome.price_10min = forward_data['close'].iloc[9]
    if len(forward_data) >= 15:
        outcome.price_15min = forward_data['close'].iloc[14]
    if len(forward_data) >= 30:
        outcome.price_30min = forward_data['close'].iloc[29]

    # Calculate MFE/MAE
    if direction == 'bullish':
        outcome.max_favorable = (forward_data['high'].max() - entry_price) / entry_price * 100
        outcome.max_adverse = (entry_price - forward_data['low'].min()) / entry_price * 100

        # Target hits
        outcome.hit_01pct = forward_data['high'].max() >= entry_price * 1.001
        outcome.hit_02pct = forward_data['high'].max() >= entry_price * 1.002
        outcome.hit_05pct = forward_data['high'].max() >= entry_price * 1.005
    else:  # bearish
        outcome.max_favorable = (entry_price - forward_data['low'].min()) / entry_price * 100
        outcome.max_adverse = (forward_data['high'].max() - entry_price) / entry_price * 100

        outcome.hit_01pct = forward_data['low'].min() <= entry_price * 0.999
        outcome.hit_02pct = forward_data['low'].min() <= entry_price * 0.998
        outcome.hit_05pct = forward_data['low'].min() <= entry_price * 0.995

    return outcome


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def load_spx_data(file_path: Path) -> pd.DataFrame:
    """Load SPX minute data from CSV."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df


def filter_morning_session(data: pd.DataFrame) -> pd.DataFrame:
    """Filter to 9:30-11:00 AM ET window."""
    # Assuming timestamps are already in ET
    morning_data = data.between_time('09:30', '11:00')
    return morning_data


def generate_signals(data: pd.DataFrame, lookback: int = 30,
                    score_threshold: float = 0.20) -> List[Dict]:
    """
    Generate confluence signals for the data.

    Args:
        data: OHLC DataFrame
        lookback: Bars to look back for analysis
        score_threshold: Minimum confluence score to generate signal
    """
    signals = []

    smc = SimplifiedSMCAnalyzer(swing_lookback=3)
    judas = SimplifiedJudasDetector(lookback=15, min_move=0.0008, reversal_threshold=0.35)
    rejection = SimplifiedRejectionDetector(min_wick_ratio=1.5, lookback=5)
    confluence = SimplifiedConfluenceCalculator(min_components=1)

    # Start after enough lookback data
    start_idx = max(lookback, 30)

    for i in range(start_idx, len(data)):
        analysis_data = data.iloc[max(0, i - lookback):i + 1]

        if len(analysis_data) < 20:
            continue

        current_price = analysis_data['close'].iloc[-1]

        # Run analysis
        swing_highs, swing_lows = smc.detect_swing_points(analysis_data)
        market_structure = smc.detect_market_structure(swing_highs, swing_lows)
        order_blocks = smc.detect_order_blocks(analysis_data, swing_highs, swing_lows)
        fvgs = smc.detect_fair_value_gaps(analysis_data)
        judas_result = judas.detect(analysis_data)
        rejection_result = rejection.detect(analysis_data)

        signal_data = {
            'current_price': current_price,
            'order_blocks': order_blocks,
            'fair_value_gaps': fvgs,
            'market_structure': market_structure,
            'judas_swing': judas_result,
            'rejection': rejection_result
        }

        # Calculate both directions
        bullish_score, bullish_breakdown = confluence.calculate(signal_data, 'bullish')
        bearish_score, bearish_breakdown = confluence.calculate(signal_data, 'bearish')

        # Generate signal if above threshold
        if bullish_score >= score_threshold or bearish_score >= score_threshold:
            if bullish_score > bearish_score:
                signals.append({
                    'index': i,
                    'timestamp': data.index[i],
                    'direction': 'bullish',
                    'score': bullish_score,
                    'breakdown': bullish_breakdown,
                    'opposing_score': bearish_score
                })
            elif bearish_score > bullish_score:
                signals.append({
                    'index': i,
                    'timestamp': data.index[i],
                    'direction': 'bearish',
                    'score': bearish_score,
                    'breakdown': bearish_breakdown,
                    'opposing_score': bullish_score
                })

    return signals


def run_test_on_file(file_path: Path, score_threshold: float = 0.20) -> List[SignalOutcome]:
    """Run test on a single SPX data file."""
    print(f"\nProcessing: {file_path.name}")

    # Load data
    data = load_spx_data(file_path)

    # Filter to morning session
    morning_data = filter_morning_session(data)

    if len(morning_data) < 50:
        print(f"  Insufficient morning data: {len(morning_data)} bars")
        return []

    print(f"  Morning bars: {len(morning_data)}")

    # Generate signals
    signals = generate_signals(morning_data, lookback=30, score_threshold=score_threshold)
    print(f"  Signals generated: {len(signals)}")

    # Track outcomes
    outcomes = []
    for signal in signals:
        outcome = track_signal_outcome(
            morning_data,
            signal['index'],
            signal,
            forward_bars=30
        )
        outcomes.append(outcome)

    return outcomes


def analyze_results(outcomes: List[SignalOutcome]) -> Dict:
    """Analyze signal outcomes and generate statistics."""
    if not outcomes:
        return {}

    results = {
        'total_signals': len(outcomes),
        'bullish_signals': sum(1 for o in outcomes if o.signal_direction == 'bullish'),
        'bearish_signals': sum(1 for o in outcomes if o.signal_direction == 'bearish'),
        'avg_score': np.mean([o.confluence_score for o in outcomes]),
        'avg_mfe': np.mean([o.max_favorable for o in outcomes]),
        'avg_mae': np.mean([o.max_adverse for o in outcomes]),
        'hit_01pct': sum(1 for o in outcomes if o.hit_01pct) / len(outcomes) * 100,
        'hit_02pct': sum(1 for o in outcomes if o.hit_02pct) / len(outcomes) * 100,
        'hit_05pct': sum(1 for o in outcomes if o.hit_05pct) / len(outcomes) * 100,
    }

    # Calculate directional accuracy at different time intervals
    for interval in [5, 10, 15, 30]:
        correct = 0
        total = 0
        for o in outcomes:
            price_at_interval = getattr(o, f'price_{interval}min', None)
            if price_at_interval is None:
                continue
            total += 1
            change = (price_at_interval - o.entry_price) / o.entry_price
            if o.signal_direction == 'bullish' and change > 0:
                correct += 1
            elif o.signal_direction == 'bearish' and change < 0:
                correct += 1

        if total > 0:
            results[f'accuracy_{interval}min'] = correct / total * 100
        else:
            results[f'accuracy_{interval}min'] = 0.0

    # Analyze by score ranges
    for low, high in [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]:
        subset = [o for o in outcomes if low <= o.confluence_score < high]
        if subset:
            results[f'score_{low}_{high}_count'] = len(subset)
            results[f'score_{low}_{high}_mfe'] = np.mean([o.max_favorable for o in subset])
            results[f'score_{low}_{high}_mae'] = np.mean([o.max_adverse for o in subset])
            results[f'score_{low}_{high}_hit01'] = sum(1 for o in subset if o.hit_01pct) / len(subset) * 100

    return results


def main():
    """Main function to run the confluence signal test."""
    print("=" * 80)
    print("CONFLUENCE SCORE SIGNAL TEST")
    print("Testing price movement after confluence signals (9:30-11:00 AM)")
    print("=" * 80)

    # Find all SPX data files
    data_dir = Path("/Users/toni/trade related/1206version/data/spx")
    spx_files = sorted(data_dir.glob("SPX_*.csv"))

    print(f"\nFound {len(spx_files)} SPX data files")

    all_outcomes = []

    for file_path in spx_files:
        outcomes = run_test_on_file(file_path, score_threshold=0.15)
        all_outcomes.extend(outcomes)

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    results = analyze_results(all_outcomes)

    if results:
        print(f"\nTotal Signals: {results['total_signals']}")
        print(f"  Bullish: {results['bullish_signals']}")
        print(f"  Bearish: {results['bearish_signals']}")
        print(f"\nAverage Confluence Score: {results['avg_score']:.2%}")
        print(f"\nMax Favorable Excursion (avg): {results['avg_mfe']:.3f}%")
        print(f"Max Adverse Excursion (avg): {results['avg_mae']:.3f}%")

        print(f"\nTarget Hit Rates:")
        print(f"  0.1% Target: {results['hit_01pct']:.1f}%")
        print(f"  0.2% Target: {results['hit_02pct']:.1f}%")
        print(f"  0.5% Target: {results['hit_05pct']:.1f}%")

        print(f"\nDirectional Accuracy:")
        for interval in [5, 10, 15, 30]:
            key = f'accuracy_{interval}min'
            if key in results:
                print(f"  {interval} min: {results[key]:.1f}%")

        print(f"\nBy Score Range:")
        for low, high in [(0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 1.0)]:
            count_key = f'score_{low}_{high}_count'
            if count_key in results:
                print(f"  {low*100:.0f}-{high*100:.0f}%: {results[count_key]} signals, "
                      f"MFE {results[f'score_{low}_{high}_mfe']:.3f}%, "
                      f"MAE {results[f'score_{low}_{high}_mae']:.3f}%, "
                      f"0.1% hit: {results[f'score_{low}_{high}_hit01']:.1f}%")

    # Print detailed signal log
    print("\n" + "=" * 80)
    print("DETAILED SIGNAL LOG (Sample)")
    print("=" * 80)

    for i, outcome in enumerate(all_outcomes[:20]):  # Show first 20
        direction_symbol = "🔼" if outcome.signal_direction == 'bullish' else "🔽"
        hit_markers = ""
        if outcome.hit_01pct:
            hit_markers += "✓0.1 "
        if outcome.hit_02pct:
            hit_markers += "✓0.2 "
        if outcome.hit_05pct:
            hit_markers += "✓0.5"

        print(f"{outcome.timestamp.strftime('%Y-%m-%d %H:%M')} | "
              f"{direction_symbol} {outcome.signal_direction.upper():7} | "
              f"Score: {outcome.confluence_score:.1%} | "
              f"MFE: {outcome.max_favorable:.2f}% | "
              f"MAE: {outcome.max_adverse:.2f}% | "
              f"{hit_markers}")

    return all_outcomes, results


if __name__ == "__main__":
    outcomes, results = main()
