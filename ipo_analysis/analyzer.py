"""ファンダメンタル・成長性分析エンジン"""
import json
from pathlib import Path
from typing import Optional
from .models import StockEntry, FundamentalData, GrowthData


class FundamentalAnalyzer:
    """ファンダメンタル指標の分析・評価"""

    # 各指標の評価基準 (min_good, max_good, weight)
    CRITERIA = {
        "roe": {"excellent": 20, "good": 15, "fair": 8, "weight": 15},
        "roa": {"excellent": 10, "good": 7, "fair": 3, "weight": 10},
        "equity_ratio": {"excellent": 60, "good": 40, "fair": 20, "weight": 8},
        "revenue_per_employee": {"excellent": 8000, "good": 5000, "fair": 3000, "weight": 7},
    }

    @staticmethod
    def calc_cagr(values: list[float]) -> Optional[float]:
        """年平均成長率 (CAGR) を計算"""
        if len(values) < 2 or values[0] <= 0:
            return None
        n = len(values) - 1
        return ((values[-1] / values[0]) ** (1 / n) - 1) * 100

    @staticmethod
    def calc_growth_rate(old: float, new: float) -> Optional[float]:
        """前期比成長率を計算"""
        if old == 0:
            return None
        return ((new - old) / abs(old)) * 100

    def score_fundamental(self, data: FundamentalData) -> dict:
        """ファンダメンタル指標のスコアリング (0-100)"""
        scores = {}
        total_weight = 0
        weighted_sum = 0

        # ROE評価
        if data.roe is not None:
            c = self.CRITERIA["roe"]
            if data.roe >= c["excellent"]:
                s = 100
            elif data.roe >= c["good"]:
                s = 80
            elif data.roe >= c["fair"]:
                s = 60
            else:
                s = 30
            scores["roe"] = {"value": data.roe, "score": s, "label": f"{data.roe:.1f}%"}
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # ROA評価
        if data.roa is not None:
            c = self.CRITERIA["roa"]
            if data.roa >= c["excellent"]:
                s = 100
            elif data.roa >= c["good"]:
                s = 80
            elif data.roa >= c["fair"]:
                s = 60
            else:
                s = 30
            scores["roa"] = {"value": data.roa, "score": s, "label": f"{data.roa:.1f}%"}
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # 自己資本比率
        if data.equity_ratio is not None:
            c = self.CRITERIA["equity_ratio"]
            if data.equity_ratio >= c["excellent"]:
                s = 100
            elif data.equity_ratio >= c["good"]:
                s = 80
            elif data.equity_ratio >= c["fair"]:
                s = 60
            else:
                s = 30
            scores["equity_ratio"] = {
                "value": data.equity_ratio, "score": s,
                "label": f"{data.equity_ratio:.1f}%"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # 1人当たり売上高
        if data.revenue_per_employee is not None:
            c = self.CRITERIA["revenue_per_employee"]
            if data.revenue_per_employee >= c["excellent"]:
                s = 100
            elif data.revenue_per_employee >= c["good"]:
                s = 80
            elif data.revenue_per_employee >= c["fair"]:
                s = 60
            else:
                s = 30
            scores["revenue_per_employee"] = {
                "value": data.revenue_per_employee, "score": s,
                "label": f"{data.revenue_per_employee:,.0f}万円"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # 売上CAGR
        if len(data.revenue_history) >= 2:
            cagr = self.calc_cagr(data.revenue_history)
            if cagr is not None:
                if cagr >= 50:
                    s = 100
                elif cagr >= 30:
                    s = 85
                elif cagr >= 15:
                    s = 65
                else:
                    s = 40
                scores["revenue_cagr"] = {
                    "value": cagr, "score": s, "label": f"{cagr:.1f}%"
                }
                weighted_sum += s * 12
                total_weight += 12

        # 営業利益CAGR
        if len(data.operating_profit_history) >= 2:
            cagr = self.calc_cagr(data.operating_profit_history)
            if cagr is not None:
                if cagr >= 80:
                    s = 100
                elif cagr >= 40:
                    s = 85
                elif cagr >= 20:
                    s = 65
                else:
                    s = 40
                scores["op_profit_cagr"] = {
                    "value": cagr, "score": s, "label": f"{cagr:.1f}%"
                }
                weighted_sum += s * 13
                total_weight += 13

        composite = weighted_sum / total_weight if total_weight > 0 else 0
        return {"scores": scores, "composite": round(composite, 1)}


class GrowthAnalyzer:
    """成長性指標の分析・評価"""

    CRITERIA = {
        "shikiho_revision_rate": {"excellent": 50, "good": 20, "fair": 5, "weight": 15},
        "operating_profit_growth": {"excellent": 50, "good": 25, "fair": 10, "weight": 15},
        "revenue_growth": {"excellent": 40, "good": 20, "fair": 10, "weight": 12},
        "forward2_per": {"cheap": 15, "fair": 25, "expensive": 40, "weight": 13},
        "peg_ratio": {"cheap": 0.5, "fair": 1.0, "expensive": 2.0, "weight": 10},
    }

    def score_growth(self, data: GrowthData) -> dict:
        """成長性指標のスコアリング (0-100)"""
        scores = {}
        total_weight = 0
        weighted_sum = 0

        # 四季報修正率
        if data.shikiho_revision_rate is not None:
            c = self.CRITERIA["shikiho_revision_rate"]
            v = data.shikiho_revision_rate
            if v >= c["excellent"]:
                s = 100
            elif v >= c["good"]:
                s = 80
            elif v >= c["fair"]:
                s = 60
            elif v >= 0:
                s = 40
            else:
                s = 20  # 下方修正
            scores["shikiho_revision_rate"] = {
                "value": v, "score": s, "label": f"+{v:.0f}%" if v > 0 else f"{v:.0f}%"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # 営業増益率
        if data.operating_profit_growth is not None:
            c = self.CRITERIA["operating_profit_growth"]
            v = data.operating_profit_growth
            if v >= c["excellent"]:
                s = 100
            elif v >= c["good"]:
                s = 80
            elif v >= c["fair"]:
                s = 60
            elif v >= 0:
                s = 40
            else:
                s = 15
            scores["operating_profit_growth"] = {
                "value": v, "score": s, "label": f"+{v:.0f}%" if v > 0 else f"{v:.0f}%"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # 売上成長率
        if data.revenue_growth is not None:
            c = self.CRITERIA["revenue_growth"]
            v = data.revenue_growth
            if v >= c["excellent"]:
                s = 100
            elif v >= c["good"]:
                s = 80
            elif v >= c["fair"]:
                s = 60
            else:
                s = 35
            scores["revenue_growth"] = {
                "value": v, "score": s, "label": f"+{v:.0f}%" if v > 0 else f"{v:.0f}%"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # 来々期PER（低いほど良い）
        if data.forward2_per is not None:
            c = self.CRITERIA["forward2_per"]
            v = data.forward2_per
            if v <= c["cheap"]:
                s = 100
            elif v <= c["fair"]:
                s = 75
            elif v <= c["expensive"]:
                s = 45
            else:
                s = 20
            scores["forward2_per"] = {
                "value": v, "score": s, "label": f"{v:.2f}倍"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        # PEGレシオ（低いほど良い）
        if data.peg_ratio is not None:
            c = self.CRITERIA["peg_ratio"]
            v = data.peg_ratio
            if v <= c["cheap"]:
                s = 100
            elif v <= c["fair"]:
                s = 75
            elif v <= c["expensive"]:
                s = 40
            else:
                s = 15
            scores["peg_ratio"] = {
                "value": v, "score": s, "label": f"{v:.2f}"
            }
            weighted_sum += s * c["weight"]
            total_weight += c["weight"]

        composite = weighted_sum / total_weight if total_weight > 0 else 0
        return {"scores": scores, "composite": round(composite, 1)}


class StockAnalyzer:
    """銘柄総合分析エンジン"""

    CATEGORY_WEIGHTS = {
        "fundamental": 0.35,
        "growth": 0.35,
        "technical": 0.30,
    }

    def __init__(self):
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.growth_analyzer = GrowthAnalyzer()

    def analyze(self, entry: StockEntry) -> dict:
        """銘柄を総合分析してスコア付き結果を返す"""
        fundamental_result = self.fundamental_analyzer.score_fundamental(entry.fundamental)
        growth_result = self.growth_analyzer.score_growth(entry.growth)

        # テクニカルスコア（手動入力ベース、5段階→100点換算）
        tech_score = 0
        if entry.technical.technical_score is not None:
            tech_score = entry.technical.technical_score * 20  # 5段階→100点

        # 総合スコア
        overall = (
            fundamental_result["composite"] * self.CATEGORY_WEIGHTS["fundamental"]
            + growth_result["composite"] * self.CATEGORY_WEIGHTS["growth"]
            + tech_score * self.CATEGORY_WEIGHTS["technical"]
        )

        return {
            "code": entry.code,
            "name": entry.name,
            "fundamental": fundamental_result,
            "growth": growth_result,
            "technical_score": tech_score,
            "overall_score": round(overall, 1),
            "rank_label": entry.rank_label,
            "tagline": entry.tagline,
        }

    def rank_stocks(self, entries: list[StockEntry]) -> list[dict]:
        """複数銘柄を分析してランキング順に返す"""
        results = [self.analyze(e) for e in entries]
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return results
