"""Feature engineering engine - generates 50 technical + 50 quant features."""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from collections import deque

import config


class FeatureEngine:
    """Generates comprehensive feature set for ML models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical data buffers for time-series features
        self.coinbase_history = deque(maxlen=1000)
        self.polymarket_history = deque(maxlen=1000)
        
    async def generate_features(self, coinbase_data: Dict[str, Any],
                               polymarket_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Generate complete feature set from market data."""
        try:
            # Store historical data
            self.coinbase_history.append(coinbase_data)
            self.polymarket_history.append(polymarket_data)
            
            # Need minimum history
            if len(self.coinbase_history) < 100:
                return None
            
            # Convert to DataFrames for pandas_ta
            cb_df = self._create_coinbase_dataframe()
            
            # Generate 50 technical indicators
            technical_features = self._generate_technical_features(cb_df)
            
            # Generate 50 quantitative models/signals
            quant_features = self._generate_quant_features(coinbase_data, polymarket_data, cb_df)
            
            # Combine all features
            features = {**technical_features, **quant_features}
            
            # Add metadata
            features['timestamp'] = datetime.now().timestamp()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}", exc_info=True)
            return None
    
    def _create_coinbase_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from Coinbase historical data."""
        data = []
        for item in self.coinbase_history:
            data.append({
                'timestamp': item['timestamp'],
                'close': item['price'],
                'open': item['price'],
                'high': item['price'],
                'low': item['price'],
                'volume': sum(t['size'] for t in item.get('recent_trades', [])[-10:])
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _generate_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate 50 technical indicators using pandas_ta."""
        features = {}
        
        try:
            # Momentum Indicators (15)
            rsi = ta.rsi(df['close'], length=config.RSI_PERIOD)
            features['rsi_14'] = rsi.iloc[-1] if not rsi.empty else 50
            features['rsi_7'] = ta.rsi(df['close'], length=7).iloc[-1] if len(df) > 7 else 50
            features['rsi_21'] = ta.rsi(df['close'], length=21).iloc[-1] if len(df) > 21 else 50
            
            # MACD
            macd = ta.macd(df['close'], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)
            if macd is not None and not macd.empty:
                features['macd'] = macd[f'MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}'].iloc[-1]
                features['macd_signal'] = macd[f'MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}'].iloc[-1]
                features['macd_hist'] = macd[f'MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}'].iloc[-1]
            else:
                features['macd'] = features['macd_signal'] = features['macd_hist'] = 0
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                features['stoch_k'] = stoch['STOCHk_14_3_3'].iloc[-1]
                features['stoch_d'] = stoch['STOCHd_14_3_3'].iloc[-1]
            else:
                features['stoch_k'] = features['stoch_d'] = 50
            
            # ROC (Rate of Change)
            features['roc_10'] = ta.roc(df['close'], length=10).iloc[-1] if len(df) > 10 else 0
            features['roc_20'] = ta.roc(df['close'], length=20).iloc[-1] if len(df) > 20 else 0
            
            # MOM (Momentum)
            features['mom_10'] = ta.mom(df['close'], length=10).iloc[-1] if len(df) > 10 else 0
            features['mom_20'] = ta.mom(df['close'], length=20).iloc[-1] if len(df) > 20 else 0
            
            # CCI (Commodity Channel Index)
            cci = ta.cci(df['high'], df['low'], df['close'], length=20)
            features['cci_20'] = cci.iloc[-1] if cci is not None and not cci.empty else 0
            
            # Williams %R
            willr = ta.willr(df['high'], df['low'], df['close'])
            features['willr_14'] = willr.iloc[-1] if willr is not None and not willr.empty else -50
            
            # Trend Indicators (15)
            # Moving Averages
            features['sma_10'] = ta.sma(df['close'], length=10).iloc[-1] if len(df) > 10 else df['close'].iloc[-1]
            features['sma_20'] = ta.sma(df['close'], length=20).iloc[-1] if len(df) > 20 else df['close'].iloc[-1]
            features['sma_50'] = ta.sma(df['close'], length=50).iloc[-1] if len(df) > 50 else df['close'].iloc[-1]
            features['ema_10'] = ta.ema(df['close'], length=10).iloc[-1] if len(df) > 10 else df['close'].iloc[-1]
            features['ema_20'] = ta.ema(df['close'], length=20).iloc[-1] if len(df) > 20 else df['close'].iloc[-1]
            features['ema_50'] = ta.ema(df['close'], length=50).iloc[-1] if len(df) > 50 else df['close'].iloc[-1]
            
            # ADX (Trend Strength)
            adx = ta.adx(df['high'], df['low'], df['close'])
            if adx is not None and not adx.empty:
                features['adx_14'] = adx['ADX_14'].iloc[-1]
                features['di_plus'] = adx['DMP_14'].iloc[-1]
                features['di_minus'] = adx['DMN_14'].iloc[-1]
            else:
                features['adx_14'] = features['di_plus'] = features['di_minus'] = 0
            
            # Aroon
            aroon = ta.aroon(df['high'], df['low'])
            if aroon is not None and not aroon.empty:
                features['aroon_up'] = aroon['AROONU_14'].iloc[-1]
                features['aroon_down'] = aroon['AROOND_14'].iloc[-1]
            else:
                features['aroon_up'] = features['aroon_down'] = 50
            
            # PSAR (Parabolic SAR)
            psar = ta.psar(df['high'], df['low'], df['close'])
            if psar is not None and not psar.empty:
                features['psar'] = psar['PSARl_0.02_0.2'].iloc[-1] if 'PSARl_0.02_0.2' in psar.columns else df['close'].iloc[-1]
            else:
                features['psar'] = df['close'].iloc[-1]
            
            # Ichimoku (simplified)
            features['ichimoku_conv'] = ((df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2).iloc[-1] if len(df) > 9 else df['close'].iloc[-1]
            features['ichimoku_base'] = ((df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2).iloc[-1] if len(df) > 26 else df['close'].iloc[-1]
            
            # Volatility Indicators (10)
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=config.BB_PERIOD, std=config.BB_STD)
            if bbands is not None and not bbands.empty:
                features['bb_upper'] = bbands[f'BBU_{config.BB_PERIOD}_{config.BB_STD}'].iloc[-1]
                features['bb_middle'] = bbands[f'BBM_{config.BB_PERIOD}_{config.BB_STD}'].iloc[-1]
                features['bb_lower'] = bbands[f'BBL_{config.BB_PERIOD}_{config.BB_STD}'].iloc[-1]
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            else:
                price = df['close'].iloc[-1]
                features['bb_upper'] = features['bb_middle'] = features['bb_lower'] = price
                features['bb_width'] = 0
            
            # ATR (Average True Range)
            atr = ta.atr(df['high'], df['low'], df['close'], length=config.ATR_PERIOD)
            features['atr_14'] = atr.iloc[-1] if atr is not None and not atr.empty else 0
            
            # Standard Deviation
            features['std_20'] = df['close'].rolling(20).std().iloc[-1] if len(df) > 20 else 0
            features['std_50'] = df['close'].rolling(50).std().iloc[-1] if len(df) > 50 else 0
            
            # Keltner Channels
            kc = ta.kc(df['high'], df['low'], df['close'])
            if kc is not None and not kc.empty:
                features['kc_upper'] = kc['KCUe_20_2'].iloc[-1]
                features['kc_lower'] = kc['KCLe_20_2'].iloc[-1]
            else:
                features['kc_upper'] = features['kc_lower'] = df['close'].iloc[-1]
            
            # Volume Indicators (10)
            # VWAP
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            features['vwap'] = vwap.iloc[-1] if vwap is not None and not vwap.empty else df['close'].iloc[-1]
            
            # Volume SMA
            features['volume_sma_20'] = df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else df['volume'].iloc[-1]
            
            # OBV (On-Balance Volume)
            obv = ta.obv(df['close'], df['volume'])
            features['obv'] = obv.iloc[-1] if obv is not None and not obv.empty else 0
            
            # MFI (Money Flow Index)
            mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            features['mfi_14'] = mfi.iloc[-1] if mfi is not None and not mfi.empty else 50
            
            # AD (Accumulation/Distribution)
            ad = ta.ad(df['high'], df['low'], df['close'], df['volume'])
            features['ad'] = ad.iloc[-1] if ad is not None and not ad.empty else 0
            
            # ADOSC (AD Oscillator)
            adosc = ta.adosc(df['high'], df['low'], df['close'], df['volume'])
            features['adosc'] = adosc.iloc[-1] if adosc is not None and not adosc.empty else 0
            
            # CMF (Chaikin Money Flow)
            cmf = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
            features['cmf_20'] = cmf.iloc[-1] if cmf is not None and not cmf.empty else 0
            
            # EOM (Ease of Movement)
            eom = ta.eom(df['high'], df['low'], df['close'], df['volume'])
            features['eom_14'] = eom.iloc[-1] if eom is not None and not eom.empty else 0
            
            # PVT (Price Volume Trend)
            pvt = ta.pvt(df['close'], df['volume'])
            features['pvt'] = pvt.iloc[-1] if pvt is not None and not pvt.empty else 0
            
            # NVI (Negative Volume Index)
            nvi = ta.nvi(df['close'], df['volume'])
            features['nvi'] = nvi.iloc[-1] if nvi is not None and not nvi.empty else 1000
            
        except Exception as e:
            self.logger.error(f"Error in technical features: {e}")
        
        return features
    
    def _generate_quant_features(self, coinbase_data: Dict, polymarket_data: Dict,
                                 df: pd.DataFrame) -> Dict[str, float]:
        """Generate 50 quantitative model features."""
        features = {}
        
        try:
            current_price = coinbase_data['price']
            poly_price = polymarket_data['price']
            
            # Order Book Imbalance (5 features)
            cb_ob = coinbase_data['orderbook']
            poly_ob = polymarket_data['orderbook']
            
            features['cb_ob_imbalance'] = self._calculate_order_imbalance(cb_ob)
            features['poly_ob_imbalance'] = self._calculate_order_imbalance(poly_ob)
            features['cb_bid_depth'] = sum(b['size'] for b in cb_ob['bids'][:10])
            features['cb_ask_depth'] = sum(a['size'] for a in cb_ob['asks'][:10])
            features['poly_bid_depth'] = sum(b['size'] for b in poly_ob['bids'][:10])
            
            # Price Velocity (5 features)
            if len(self.coinbase_history) >= 10:
                prices = [h['price'] for h in list(self.coinbase_history)[-10:]]
                features['price_velocity_1s'] = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
                features['price_velocity_5s'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
                features['price_velocity_10s'] = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
                features['price_acceleration'] = self._calculate_acceleration(prices)
                features['price_jerk'] = self._calculate_jerk(prices)
            else:
                features['price_velocity_1s'] = features['price_velocity_5s'] = 0
                features['price_velocity_10s'] = features['price_acceleration'] = features['price_jerk'] = 0
            
            # Correlation & Lag (10 features)
            features['spot_pred_correlation'] = self._calculate_correlation_lag()
            features['correlation_strength'] = abs(features['spot_pred_correlation'])
            features['price_divergence'] = (current_price / 100000 - poly_price) if current_price > 0 else 0
            features['price_convergence_rate'] = self._calculate_convergence_rate()
            features['lead_lag_indicator'] = self._calculate_lead_lag()
            features['cointegration_score'] = self._calculate_cointegration()
            features['correlation_rolling_30'] = self._rolling_correlation(30)
            features['correlation_rolling_60'] = self._rolling_correlation(60)
            features['correlation_rolling_100'] = self._rolling_correlation(100)
            features['correlation_trend'] = features['correlation_rolling_30'] - features['correlation_rolling_100']
            
            # Volatility Features (10 features)
            features['realized_volatility'] = df['close'].pct_change().std() * np.sqrt(252) if len(df) > 1 else 0
            features['garman_klass_vol'] = self._garman_klass_volatility(df)
            features['parkinson_vol'] = self._parkinson_volatility(df)
            features['poly_implied_vol'] = self._calculate_implied_volatility(poly_ob)
            features['vol_smile'] = features['poly_implied_vol'] - features['realized_volatility']
            features['vol_skew'] = self._calculate_vol_skew(poly_ob)
            features['vol_term_structure'] = self._calculate_vol_term_structure()
            features['vol_of_vol'] = self._calculate_vol_of_vol(df)
            features['vol_regime'] = 1 if features['realized_volatility'] > 0.5 else 0
            features['vol_percentile'] = self._calculate_vol_percentile(df)
            
            # Microstructure Features (10 features)
            features['effective_spread'] = coinbase_data.get('bid_ask_spread', 0)
            features['poly_spread'] = polymarket_data.get('spread', 0)
            features['spread_ratio'] = features['poly_spread'] / features['effective_spread'] if features['effective_spread'] > 0 else 0
            features['tick_imbalance'] = self._calculate_tick_imbalance()
            features['trade_intensity'] = len(coinbase_data.get('recent_trades', [])) / 60
            features['volume_imbalance'] = self._calculate_volume_imbalance(coinbase_data)
            features['price_impact'] = self._estimate_price_impact(cb_ob)
            features['kyle_lambda'] = self._calculate_kyle_lambda(coinbase_data)
            features['amihud_illiquidity'] = self._calculate_amihud(df)
            features['roll_spread'] = self._calculate_roll_spread(df)
            
            # Market Regime Features (10 features)
            features['trend_strength'] = self._calculate_trend_strength(df)
            features['market_efficiency'] = self._calculate_market_efficiency(df)
            features['hurst_exponent'] = self._calculate_hurst_exponent(df)
            features['fractal_dimension'] = 2 - features['hurst_exponent']
            features['entropy'] = self._calculate_entropy(df)
            features['lyapunov_exponent'] = self._calculate_lyapunov(df)
            features['regime_hmm'] = self._detect_regime_hmm(df)
            features['crisis_indicator'] = 1 if features['realized_volatility'] > 1.0 else 0
            features['liquidity_score'] = self._calculate_liquidity_score(cb_ob)
            features['market_stress'] = self._calculate_market_stress(features)
            
        except Exception as e:
            self.logger.error(f"Error in quant features: {e}")
        
        return features
    
    # Helper methods for quantitative features
    def _calculate_order_imbalance(self, orderbook: Dict) -> float:
        """Calculate order book imbalance ratio."""
        bid_volume = sum(b.get('size', 0) for b in orderbook.get('bids', []))
        ask_volume = sum(a.get('size', 0) for a in orderbook.get('asks', []))
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0
    
    def _calculate_acceleration(self, prices: list) -> float:
        """Calculate price acceleration (2nd derivative)."""
        if len(prices) < 3:
            return 0
        velocities = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if len(velocities) < 2:
            return 0
        return velocities[-1] - velocities[-2]
    
    def _calculate_jerk(self, prices: list) -> float:
        """Calculate price jerk (3rd derivative)."""
        if len(prices) < 4:
            return 0
        accelerations = []
        for i in range(2, len(prices)):
            v1 = (prices[i-1] - prices[i-2]) / prices[i-2]
            v2 = (prices[i] - prices[i-1]) / prices[i-1]
            accelerations.append(v2 - v1)
        if len(accelerations) < 2:
            return 0
        return accelerations[-1] - accelerations[-2]
    
    def _calculate_correlation_lag(self) -> float:
        """Calculate rolling correlation between spot and prediction prices."""
        if len(self.coinbase_history) < config.ROLLING_CORRELATION_WINDOW:
            return 0
        
        spot_prices = [h['price'] for h in list(self.coinbase_history)[-config.ROLLING_CORRELATION_WINDOW:]]
        pred_prices = [h['price'] for h in list(self.polymarket_history)[-config.ROLLING_CORRELATION_WINDOW:]]
        
        return np.corrcoef(spot_prices, pred_prices)[0, 1] if len(spot_prices) == len(pred_prices) else 0
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate rate of convergence between spot and prediction prices."""
        if len(self.coinbase_history) < 10:
            return 0
        
        divergences = []
        for cb, poly in zip(list(self.coinbase_history)[-10:], list(self.polymarket_history)[-10:]):
            div = abs(cb['price'] / 100000 - poly['price'])
            divergences.append(div)
        
        if len(divergences) < 2:
            return 0
        
        return (divergences[0] - divergences[-1]) / len(divergences)
    
    def _calculate_lead_lag(self) -> float:
        """Calculate lead-lag relationship."""
        if len(self.coinbase_history) < 20:
            return 0
        
        spot_returns = pd.Series([h['price'] for h in list(self.coinbase_history)[-20:]]).pct_change()
        pred_returns = pd.Series([h['price'] for h in list(self.polymarket_history)[-20:]]).pct_change()
        
        # Cross-correlation at lag 1
        return spot_returns.corr(pred_returns.shift(1))
    
    def _calculate_cointegration(self) -> float:
        """Simple cointegration score."""
        # Simplified version - in production, use statsmodels
        return self._calculate_correlation_lag()
    
    def _rolling_correlation(self, window: int) -> float:
        """Calculate rolling correlation."""
        if len(self.coinbase_history) < window:
            return 0
        
        spot_prices = [h['price'] for h in list(self.coinbase_history)[-window:]]
        pred_prices = [h['price'] for h in list(self.polymarket_history)[-window:]]
        
        return np.corrcoef(spot_prices, pred_prices)[0, 1] if len(spot_prices) == len(pred_prices) else 0
    
    def _garman_klass_volatility(self, df: pd.DataFrame) -> float:
        """Garman-Klass volatility estimator."""
        if len(df) < 20:
            return 0
        
        log_hl = (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        log_co = (np.log(df['close'] / df['open']) ** 2).rolling(20).mean()
        
        gk_vol = np.sqrt(0.5 * log_hl.iloc[-1] - (2 * np.log(2) - 1) * log_co.iloc[-1])
        return gk_vol if not np.isnan(gk_vol) else 0
    
    def _parkinson_volatility(self, df: pd.DataFrame) -> float:
        """Parkinson volatility estimator."""
        if len(df) < 20:
            return 0
        
        park_vol = np.sqrt((1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(20).mean().iloc[-1])
        return park_vol if not np.isnan(park_vol) else 0
    
    def _calculate_implied_volatility(self, orderbook: Dict) -> float:
        """Calculate implied volatility from spread."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0
        
        spread = asks[0]['price'] - bids[0]['price']
        mid = (asks[0]['price'] + bids[0]['price']) / 2
        
        return spread / mid if mid > 0 else 0
    
    def _calculate_vol_skew(self, orderbook: Dict) -> float:
        """Calculate volatility skew."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if len(bids) < 5 or len(asks) < 5:
            return 0
        
        # Measure asymmetry in order book depth
        bid_depth = sum(b['size'] for b in bids[:5])
        ask_depth = sum(a['size'] for a in asks[:5])
        
        return (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
    
    def _calculate_vol_term_structure(self) -> float:
        """Simplified vol term structure."""
        # In production, would use options with different maturities
        return 0
    
    def _calculate_vol_of_vol(self, df: pd.DataFrame) -> float:
        """Calculate volatility of volatility."""
        if len(df) < 50:
            return 0
        
        rolling_vol = df['close'].pct_change().rolling(20).std()
        vol_of_vol = rolling_vol.std()
        
        return vol_of_vol if not np.isnan(vol_of_vol) else 0
    
    def _calculate_vol_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current vol percentile."""
        if len(df) < 100:
            return 0.5
        
        vol_series = df['close'].pct_change().rolling(20).std()
        current_vol = vol_series.iloc[-1]
        percentile = (vol_series < current_vol).sum() / len(vol_series)
        
        return percentile if not np.isnan(percentile) else 0.5
    
    def _calculate_tick_imbalance(self) -> float:
        """Calculate tick-level buy/sell imbalance."""
        if len(self.coinbase_history) < 10:
            return 0
        
        recent_trades = list(self.coinbase_history)[-10:]
        buy_volume = sum(t.get('recent_trades', [{}])[0].get('size', 0) 
                        for t in recent_trades 
                        if t.get('recent_trades', [{}])[0].get('side') == 'buy')
        sell_volume = sum(t.get('recent_trades', [{}])[0].get('size', 0) 
                         for t in recent_trades 
                         if t.get('recent_trades', [{}])[0].get('side') == 'sell')
        
        total = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total if total > 0 else 0
    
    def _calculate_volume_imbalance(self, coinbase_data: Dict) -> float:
        """Calculate volume-weighted imbalance."""
        recent_trades = coinbase_data.get('recent_trades', [])
        if not recent_trades:
            return 0
        
        buy_vol = sum(t['size'] for t in recent_trades if t.get('side') == 'buy')
        sell_vol = sum(t['size'] for t in recent_trades if t.get('side') == 'sell')
        
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0
    
    def _estimate_price_impact(self, orderbook: Dict) -> float:
        """Estimate price impact of a market order."""
        asks = orderbook.get('asks', [])
        if not asks:
            return 0
        
        # Calculate cost to buy $10k worth
        target = 10000
        cost = 0
        volume = 0
        
        for ask in asks:
            if volume >= target:
                break
            cost += ask['price'] * ask['size']
            volume += ask['price'] * ask['size']
        
        avg_price = cost / volume if volume > 0 else asks[0]['price']
        return (avg_price - asks[0]['price']) / asks[0]['price'] if asks[0]['price'] > 0 else 0
    
    def _calculate_kyle_lambda(self, coinbase_data: Dict) -> float:
        """Kyle's lambda (price impact coefficient)."""
        recent_trades = coinbase_data.get('recent_trades', [])
        if len(recent_trades) < 2:
            return 0
        
        price_changes = [recent_trades[i]['price'] - recent_trades[i-1]['price'] 
                        for i in range(1, len(recent_trades))]
        signed_volumes = [t['size'] if t.get('side') == 'buy' else -t['size'] 
                         for t in recent_trades[1:]]
        
        if not price_changes or not signed_volumes:
            return 0
        
        # Simple linear regression
        corr = np.corrcoef(signed_volumes, price_changes)[0, 1]
        return corr if not np.isnan(corr) else 0
    
    def _calculate_amihud(self, df: pd.DataFrame) -> float:
        """Amihud illiquidity measure."""
        if len(df) < 20:
            return 0
        
        returns = df['close'].pct_change().abs()
        illiquidity = (returns / df['volume']).rolling(20).mean().iloc[-1]
        
        return illiquidity if not np.isnan(illiquidity) else 0
    
    def _calculate_roll_spread(self, df: pd.DataFrame) -> float:
        """Roll's spread estimator."""
        if len(df) < 20:
            return 0
        
        price_changes = df['close'].diff()
        covariance = price_changes.rolling(20).cov(price_changes.shift(1)).iloc[-1]
        
        roll = 2 * np.sqrt(-covariance) if covariance < 0 else 0
        return roll if not np.isnan(roll) else 0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression."""
        if len(df) < 20:
            return 0
        
        prices = df['close'][-20:].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        return slope / prices[-1] if prices[-1] > 0 else 0
    
    def _calculate_market_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate market efficiency ratio."""
        if len(df) < 20:
            return 0.5
        
        prices = df['close'][-20:].values
        net_change = abs(prices[-1] - prices[0])
        total_change = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
        
        return net_change / total_change if total_change > 0 else 0
    
    def _calculate_hurst_exponent(self, df: pd.DataFrame) -> float:
        """Calculate Hurst exponent (simplified)."""
        if len(df) < 50:
            return 0.5
        
        prices = df['close'][-50:].values
        lags = range(2, 20)
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    def _calculate_entropy(self, df: pd.DataFrame) -> float:
        """Calculate Shannon entropy of returns."""
        if len(df) < 50:
            return 0
        
        returns = df['close'].pct_change().dropna()[-50:]
        hist, _ = np.histogram(returns, bins=20)
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy if not np.isnan(entropy) else 0
    
    def _calculate_lyapunov(self, df: pd.DataFrame) -> float:
        """Simplified Lyapunov exponent."""
        # Placeholder - proper calculation is complex
        return 0
    
    def _detect_regime_hmm(self, df: pd.DataFrame) -> float:
        """Simplified regime detection."""
        if len(df) < 50:
            return 0
        
        # Simple volatility-based regime (0 = low vol, 1 = high vol)
        recent_vol = df['close'].pct_change().std()
        historical_vol = df['close'].pct_change()[-50:].std()
        
        return 1 if recent_vol > historical_vol else 0
    
    def _calculate_liquidity_score(self, orderbook: Dict) -> float:
        """Calculate overall liquidity score."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0
        
        # Combine depth and spread
        depth = sum(b['size'] for b in bids[:10]) + sum(a['size'] for a in asks[:10])
        spread = (asks[0]['price'] - bids[0]['price']) / bids[0]['price']
        
        return depth * (1 - spread) if spread < 1 else 0
    
    def _calculate_market_stress(self, features: Dict) -> float:
        """Calculate composite market stress indicator."""
        stress_components = [
            features.get('realized_volatility', 0) / 2,
            abs(features.get('cb_ob_imbalance', 0)),
            features.get('effective_spread', 0) * 100,
            1 - features.get('liquidity_score', 0) / 1000
        ]
        
        return np.mean(stress_components)
