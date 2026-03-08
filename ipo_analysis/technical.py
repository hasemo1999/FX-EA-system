"""テクニカル分析モジュール"""
from typing import Optional
from .models import TechnicalData


class TechnicalAnalyzer:
    """テクニカル指標の計算・評価"""

    @staticmethod
    def evaluate_ma_position(price: float, ma: float) -> str:
        """株価と移動平均線の位置関係を評価"""
        if price > ma:
            return "above"
        return "below"

    @staticmethod
    def detect_golden_cross(ma_short: float, ma_long: float,
                            prev_ma_short: Optional[float] = None,
                            prev_ma_long: Optional[float] = None) -> Optional[bool]:
        """ゴールデンクロス/デッドクロスを検出"""
        if prev_ma_short is None or prev_ma_long is None:
            # 現在のスナップショットのみ: 短期 > 長期 ならGC済み
            return ma_short > ma_long
        # 前回: short < long → 今回: short > long
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            return True
        return False

    @staticmethod
    def calc_distance_pct(price: float, reference: float) -> float:
        """価格と基準値の乖離率 (%)"""
        if reference == 0:
            return 0.0
        return ((price - reference) / reference) * 100

    def generate_entry_signal(self, data: TechnicalData) -> dict:
        """エントリーシグナルの判定"""
        signals = []
        strength = 0

        # ゴールデンクロス確認
        if data.golden_cross_25_75:
            signals.append("25MA/75MA ゴールデンクロス形成済み")
            strength += 2

        # 株価 vs MA位置
        if data.price_vs_ma25 == "above":
            signals.append("株価が25MAの上")
            strength += 1
        if data.price_vs_ma75 == "above":
            signals.append("株価が75MAの上")
            strength += 1

        # レジスタンスブレイク判定
        if (data.current_price is not None and
                data.resistance_price is not None):
            distance = self.calc_distance_pct(
                data.current_price, data.resistance_price
            )
            if distance >= 0:
                signals.append(f"レジスタンス{data.resistance_price}円をブレイク済み")
                strength += 2
            elif distance >= -3:
                signals.append(
                    f"レジスタンス{data.resistance_price}円まで"
                    f"あと{abs(distance):.1f}% (ブレイク目前)"
                )
                strength += 1

        # RSI判定
        if data.rsi_14 is not None:
            if 40 <= data.rsi_14 <= 60:
                signals.append(f"RSI(14)={data.rsi_14:.0f} ニュートラル圏")
            elif data.rsi_14 < 30:
                signals.append(f"RSI(14)={data.rsi_14:.0f} 売られすぎ → 反発期待")
                strength += 1
            elif data.rsi_14 > 70:
                signals.append(f"RSI(14)={data.rsi_14:.0f} 買われすぎ → 注意")
                strength -= 1

        # 出来高
        if data.volume_ratio is not None and data.volume_ratio >= 1.5:
            signals.append(f"出来高倍率 {data.volume_ratio:.1f}x (活況)")
            strength += 1

        # 5段階判定
        if strength >= 5:
            verdict = "強い買いシグナル"
        elif strength >= 3:
            verdict = "買いシグナル"
        elif strength >= 1:
            verdict = "やや買い寄り"
        elif strength >= 0:
            verdict = "中立"
        else:
            verdict = "様子見"

        return {
            "signals": signals,
            "strength": strength,
            "verdict": verdict,
            "entry_trigger": data.entry_trigger,
        }

    def score_technical(self, data: TechnicalData) -> dict:
        """テクニカルスコアの計算 (0-100)"""
        if data.technical_score is not None:
            # 手動入力スコアがある場合はそれを使用
            score = data.technical_score * 20
        else:
            # 自動計算
            signal_result = self.generate_entry_signal(data)
            score = min(100, max(0, signal_result["strength"] * 15 + 20))

        signal_result = self.generate_entry_signal(data)
        return {
            "score": score,
            "signals": signal_result["signals"],
            "verdict": signal_result["verdict"],
            "entry_trigger": data.entry_trigger,
            "note": data.technical_note,
        }
