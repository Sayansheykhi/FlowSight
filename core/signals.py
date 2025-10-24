"""
Trading signal generation module.
Generates LONG/SHORT/WAIT signals based on order flow metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from .metrics import OrderFlowMetrics, OrderFlowCalculator

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    metrics: OrderFlowMetrics
    reasoning: List[str]
    risk_score: float  # 0.0 to 1.0 (higher = more risky)

class SignalGenerator:
    """Generates trading signals based on order flow analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.calculator = OrderFlowCalculator()
        
        # Signal thresholds
        self.cvd_threshold = config.get('cvd_threshold', 0.3)
        self.imbalance_threshold = config.get('imbalance_threshold', 0.2)
        self.absorption_threshold = config.get('absorption_threshold', 0.4)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.min_volume_threshold = config.get('min_volume_threshold', 1000)
        
    def update_calculator(self, calculator: OrderFlowCalculator):
        """Update the metrics calculator."""
        self.calculator = calculator
    
    def generate_signal(self, symbol: str) -> TradingSignal:
        """Generate a trading signal for a symbol."""
        try:
            metrics = self.calculator.calculate_all_metrics(symbol)
            
            # Calculate signal components
            cvd_signal = self._analyze_cvd(metrics)
            imbalance_signal = self._analyze_imbalance(metrics)
            absorption_signal = self._analyze_absorption(metrics)
            momentum_signal = self._analyze_momentum(symbol)
            volume_signal = self._analyze_volume(symbol)
            
            # Combine signals
            signal_type, confidence, reasoning = self._combine_signals(
                cvd_signal, imbalance_signal, absorption_signal, 
                momentum_signal, volume_signal
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(metrics, symbol)
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=metrics.mid_price,
                metrics=metrics,
                reasoning=reasoning,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._create_wait_signal(symbol, f"Error: {str(e)}")
    
    def _analyze_cvd(self, metrics: OrderFlowMetrics) -> Tuple[SignalType, float, str]:
        """Analyze CVD for signal generation."""
        cvd = metrics.cvd
        
        if cvd > self.cvd_threshold:
            return SignalType.LONG, min(abs(cvd) / 2, 1.0), f"Strong buying pressure (CVD: {cvd:.2f})"
        elif cvd < -self.cvd_threshold:
            return SignalType.SHORT, min(abs(cvd) / 2, 1.0), f"Strong selling pressure (CVD: {cvd:.2f})"
        else:
            return SignalType.WAIT, 0.3, f"Neutral CVD (CVD: {cvd:.2f})"
    
    def _analyze_imbalance(self, metrics: OrderFlowMetrics) -> Tuple[SignalType, float, str]:
        """Analyze order book imbalance."""
        imbalance = metrics.imbalance
        
        if imbalance > self.imbalance_threshold:
            return SignalType.LONG, min(abs(imbalance) * 2, 1.0), f"Bid-heavy order book (Imbalance: {imbalance:.2f})"
        elif imbalance < -self.imbalance_threshold:
            return SignalType.SHORT, min(abs(imbalance) * 2, 1.0), f"Ask-heavy order book (Imbalance: {imbalance:.2f})"
        else:
            return SignalType.WAIT, 0.3, f"Balanced order book (Imbalance: {imbalance:.2f})"
    
    def _analyze_absorption(self, metrics: OrderFlowMetrics) -> Tuple[SignalType, float, str]:
        """Analyze absorption ratio."""
        absorption = metrics.absorption
        
        if absorption > self.absorption_threshold:
            return SignalType.WAIT, 0.7, f"High absorption - institutional activity (Absorption: {absorption:.2f})"
        else:
            return SignalType.WAIT, 0.3, f"Low absorption - retail activity (Absorption: {absorption:.2f})"
    
    def _analyze_momentum(self, symbol: str) -> Tuple[SignalType, float, str]:
        """Analyze price momentum."""
        momentum = self.calculator.calculate_momentum(symbol)
        
        if momentum > 0.01:  # 1% positive momentum
            return SignalType.LONG, min(abs(momentum) * 50, 1.0), f"Positive momentum ({momentum:.3f})"
        elif momentum < -0.01:  # 1% negative momentum
            return SignalType.SHORT, min(abs(momentum) * 50, 1.0), f"Negative momentum ({momentum:.3f})"
        else:
            return SignalType.WAIT, 0.3, f"Neutral momentum ({momentum:.3f})"
    
    def _analyze_volume(self, symbol: str) -> Tuple[SignalType, float, str]:
        """Analyze volume characteristics."""
        trades = self.calculator.trade_history.get(symbol, [])
        
        if len(trades) < 10:
            return SignalType.WAIT, 0.1, "Insufficient volume data"
        
        recent_volume = sum(trade.size for trade in trades[-10:])
        
        if recent_volume > self.min_volume_threshold:
            return SignalType.WAIT, 0.8, f"High volume activity ({recent_volume:.0f})"
        else:
            return SignalType.WAIT, 0.2, f"Low volume activity ({recent_volume:.0f})"
    
    def _combine_signals(self, *signals) -> Tuple[SignalType, float, List[str]]:
        """Combine multiple signal components."""
        signal_types = [s[0] for s in signals]
        confidences = [s[1] for s in signals]
        reasonings = [s[2] for s in signals]
        
        # Count signal types
        long_count = signal_types.count(SignalType.LONG)
        short_count = signal_types.count(SignalType.SHORT)
        wait_count = signal_types.count(SignalType.WAIT)
        
        # Determine final signal
        if long_count >= 2 and long_count > short_count:
            final_signal = SignalType.LONG
            confidence = np.mean([c for s, c, r in signals if s == SignalType.LONG])
        elif short_count >= 2 and short_count > long_count:
            final_signal = SignalType.SHORT
            confidence = np.mean([c for s, c, r in signals if s == SignalType.SHORT])
        else:
            final_signal = SignalType.WAIT
            confidence = np.mean(confidences)
        
        # Adjust confidence based on consensus
        consensus_factor = max(long_count, short_count) / len(signals)
        confidence *= consensus_factor
        
        # Ensure minimum confidence for non-WAIT signals
        if final_signal != SignalType.WAIT and confidence < self.confidence_threshold:
            final_signal = SignalType.WAIT
            confidence = 0.3
        
        return final_signal, confidence, reasonings
    
    def _calculate_risk_score(self, metrics: OrderFlowMetrics, symbol: str) -> float:
        """Calculate risk score for the signal."""
        risk_factors = []
        
        # Volatility risk
        volatility = self.calculator.calculate_volatility(symbol)
        risk_factors.append(min(volatility * 10, 1.0))
        
        # Spread risk
        spread_risk = min(metrics.bid_ask_spread / metrics.mid_price * 1000, 1.0)
        risk_factors.append(spread_risk)
        
        # Absorption risk (high absorption = higher risk)
        risk_factors.append(metrics.absorption)
        
        # Volume risk (low volume = higher risk)
        trades = self.calculator.trade_history.get(symbol, [])
        if trades:
            recent_volume = sum(trade.size for trade in trades[-10:])
            volume_risk = max(0, 1 - recent_volume / self.min_volume_threshold)
            risk_factors.append(volume_risk)
        
        return np.mean(risk_factors)
    
    def _create_wait_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a WAIT signal."""
        return TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=SignalType.WAIT,
            confidence=0.1,
            price=0.0,
            metrics=None,
            reasoning=[reason],
            risk_score=1.0
        )
    
    def get_signal_strength(self, signal: TradingSignal) -> str:
        """Get human-readable signal strength."""
        if signal.confidence >= 0.8:
            return "STRONG"
        elif signal.confidence >= 0.6:
            return "MODERATE"
        elif signal.confidence >= 0.4:
            return "WEAK"
        else:
            return "VERY_WEAK"
    
    def should_alert(self, signal: TradingSignal) -> bool:
        """Determine if signal should trigger an alert."""
        return (
            signal.signal_type != SignalType.WAIT and
            signal.confidence >= self.confidence_threshold and
            signal.risk_score <= 0.7
        )
    
    def get_signal_summary(self, signal: TradingSignal) -> Dict:
        """Get a summary of the signal for alerts."""
        return {
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "signal": signal.signal_type.value,
            "confidence": f"{signal.confidence:.2f}",
            "strength": self.get_signal_strength(signal),
            "price": f"{signal.price:.4f}",
            "risk_score": f"{signal.risk_score:.2f}",
            "reasoning": signal.reasoning,
            "metrics": {
                "cvd": f"{signal.metrics.cvd:.2f}" if signal.metrics else "N/A",
                "imbalance": f"{signal.metrics.imbalance:.2f}" if signal.metrics else "N/A",
                "absorption": f"{signal.metrics.absorption:.2f}" if signal.metrics else "N/A",
                "vwap": f"{signal.metrics.vwap:.4f}" if signal.metrics else "N/A"
            }
        }

@dataclass
class SessionState:
    """Session state for tracking trading activity."""
    start_time: datetime
    total_volume: float = 0.0
    trade_count: int = 0
    signal_count: int = 0
    last_signal_time: Optional[datetime] = None
    consecutive_signals: int = 0
    session_volatility: float = 0.0
    avg_trade_size: float = 0.0

def generate_signal(metrics: Dict[str, float], session_state: SessionState) -> Tuple[str, int]:
    """
    Generate trading signal based on order flow metrics and session state.
    
    Rules:
    - LONG when CVD rising, imbalance > 0.15, price >= VWAP
    - SHORT when CVD falling, imbalance < -0.15, price <= VWAP
    - WAIT otherwise
    
    Args:
        metrics: Dictionary containing order flow metrics
        session_state: Current session state
        
    Returns:
        Tuple of (signal_type, confidence_score)
        signal_type: "LONG", "SHORT", or "WAIT"
        confidence_score: 0-100
    """
    try:
        # Extract metrics with defaults
        cvd = metrics.get('cvd', 0.0)
        cvd_trend = metrics.get('cvd_trend', 0.0)  # CVD change over time
        imbalance = metrics.get('imbalance', 0.0)
        price = metrics.get('price', 0.0)
        vwap = metrics.get('vwap', 0.0)
        
        # Calculate confidence factors
        confidence_factors = []
        
        # CVD trend factor
        cvd_strength = abs(cvd_trend)
        cvd_factor = min(cvd_strength * 20, 1.0)  # Scale CVD trend
        confidence_factors.append(cvd_factor)
        
        # Imbalance strength factor
        imbalance_strength = abs(imbalance)
        imbalance_factor = min(imbalance_strength * 3, 1.0)  # Scale imbalance
        confidence_factors.append(imbalance_factor)
        
        # Price vs VWAP alignment factor
        if vwap > 0:
            price_vwap_ratio = price / vwap
            if price_vwap_ratio >= 1.0:
                price_factor = min((price_vwap_ratio - 1.0) * 10, 1.0)
            else:
                price_factor = min((1.0 - price_vwap_ratio) * 10, 1.0)
        else:
            price_factor = 0.0
        confidence_factors.append(price_factor)
        
        # Session activity factor
        session_factor = _calculate_session_factor(session_state)
        confidence_factors.append(session_factor)
        
        # Determine signal based on rules
        signal_type = "WAIT"
        base_confidence = 0
        
        # LONG signal conditions (temporarily removing imbalance requirement)
        if (cvd_trend > 0 and  # CVD rising
            price >= vwap * 0.999):  # Price >= VWAP (with small tolerance)
            
            signal_type = "LONG"
            base_confidence = 60  # Base confidence for LONG
            
            # Boost confidence based on strength
            cvd_boost = min(cvd_trend * 100, 20)
            imbalance_boost = min((imbalance - 0.05) * 200, 15)
            price_boost = min((price / vwap - 1.0) * 100, 10) if vwap > 0 else 0
            
            base_confidence += cvd_boost + imbalance_boost + price_boost
        
        # SHORT signal conditions (temporarily removing imbalance requirement)
        elif (cvd_trend < 0 and  # CVD falling
              price <= vwap * 1.001):  # Price <= VWAP (with small tolerance)
              
            signal_type = "SHORT"
            base_confidence = 60  # Base confidence for SHORT
            
            # Boost confidence based on strength
            cvd_boost = min(abs(cvd_trend) * 100, 20)
            imbalance_boost = min((abs(imbalance) - 0.05) * 200, 15)
            price_boost = min((1.0 - price / vwap) * 100, 10) if vwap > 0 else 0
            
            base_confidence += cvd_boost + imbalance_boost + price_boost
        
        # WAIT signal (default)
        else:
            signal_type = "WAIT"
            base_confidence = 20  # Low confidence for WAIT
            
            # Adjust confidence based on how close we are to signal conditions
            if cvd_trend > 0 and imbalance > 0.05 and price >= vwap * 0.999:
                base_confidence = 40  # Close to LONG
            elif cvd_trend < 0 and imbalance < -0.05 and price <= vwap * 1.001:
                base_confidence = 40  # Close to SHORT
        
        # Calculate final confidence
        avg_factor = np.mean(confidence_factors) if confidence_factors else 0.5
        confidence_multiplier = 0.5 + (avg_factor * 0.5)  # Range: 0.5 to 1.0
        
        final_confidence = int(base_confidence * confidence_multiplier)
        final_confidence = max(0, min(100, final_confidence))  # Clamp to 0-100
        
        # Adjust for session activity
        if signal_type != "WAIT":
            # Reduce confidence if too many recent signals
            if session_state.consecutive_signals > 3:
                final_confidence = int(final_confidence * 0.7)
            
            # Reduce confidence if low session activity
            if session_state.trade_count < 10:
                final_confidence = int(final_confidence * 0.8)
        
        return signal_type, final_confidence
        
    except Exception as e:
        logger.error(f"Error in generate_signal: {e}")
        return "WAIT", 10

def _calculate_session_factor(session_state: SessionState) -> float:
    """Calculate session activity factor for confidence adjustment."""
    try:
        # Time factor (more activity = higher confidence)
        session_duration = (datetime.now() - session_state.start_time).total_seconds() / 3600  # hours
        time_factor = min(session_duration / 2, 1.0)  # Normalize to 2 hours
        
        # Volume factor
        if session_state.trade_count > 0:
            avg_volume_per_trade = session_state.total_volume / session_state.trade_count
            volume_factor = min(avg_volume_per_trade / 10, 1.0)  # Normalize to 10 units
        else:
            volume_factor = 0.0
        
        # Activity factor
        activity_factor = min(session_state.trade_count / 100, 1.0)  # Normalize to 100 trades
        
        # Signal frequency factor (penalize too many signals)
        if session_state.signal_count > 0 and session_duration > 0:
            signals_per_hour = session_state.signal_count / session_duration
            frequency_factor = max(0.5, 1.0 - (signals_per_hour - 5) * 0.1)  # Penalty after 5 signals/hour
        else:
            frequency_factor = 1.0
        
        # Combine factors
        session_factor = (time_factor * 0.3 + 
                         volume_factor * 0.3 + 
                         activity_factor * 0.2 + 
                         frequency_factor * 0.2)
        
        return max(0.0, min(1.0, session_factor))
        
    except Exception as e:
        logger.error(f"Error calculating session factor: {e}")
        return 0.5

def update_session_state(session_state: SessionState, 
                        trade_volume: float = 0.0, 
                        signal_generated: bool = False) -> SessionState:
    """Update session state with new trade or signal data."""
    try:
        # Update trade data
        if trade_volume > 0:
            session_state.total_volume += trade_volume
            session_state.trade_count += 1
            session_state.avg_trade_size = session_state.total_volume / session_state.trade_count
        
        # Update signal data
        if signal_generated:
            current_time = datetime.now()
            
            # Check if this is consecutive signal
            if (session_state.last_signal_time and 
                (current_time - session_state.last_signal_time).total_seconds() < 300):  # 5 minutes
                session_state.consecutive_signals += 1
            else:
                session_state.consecutive_signals = 1
            
            session_state.signal_count += 1
            session_state.last_signal_time = current_time
        
        # Calculate session volatility (simplified)
        if session_state.trade_count > 10:
            # This would typically use price data, simplified here
            session_state.session_volatility = min(session_state.trade_count / 1000, 0.1)
        
        return session_state
        
    except Exception as e:
        logger.error(f"Error updating session state: {e}")
        return session_state
