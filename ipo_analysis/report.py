"""レポート生成モジュール"""
import json
from pathlib import Path
from datetime import datetime
from .models import StockEntry
from .analyzer import StockAnalyzer
from .technical import TechnicalAnalyzer


class ReportGenerator:
    """分析レポートの生成"""

    def __init__(self):
        self.stock_analyzer = StockAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()

    def generate_stock_report(self, entry: StockEntry) -> str:
        """個別銘柄の詳細レポートを生成"""
        analysis = self.stock_analyzer.analyze(entry)
        tech_detail = self.technical_analyzer.score_technical(entry.technical)

        lines = []
        lines.append("=" * 70)
        lines.append(
            f"  {entry.rank_label} {entry.code} {entry.name}"
            f" ― 「{entry.tagline}」"
        )
        lines.append("=" * 70)
        lines.append(f"  分析日: {entry.analysis_date}")
        lines.append(
            f"  総合スコア: {analysis['overall_score']}/100"
        )
        lines.append("")

        # ── ファンダメンタル ──
        lines.append("─── ファンダメンタル分析 ───")
        fa = analysis["fundamental"]
        lines.append(f"  カテゴリスコア: {fa['composite']}/100")
        lines.append("")

        # 売上推移
        if entry.fundamental.revenue_history:
            rev_str = " → ".join(
                f"{v:,.0f}M" for v in entry.fundamental.revenue_history
            )
            lbl_str = " → ".join(entry.fundamental.revenue_labels)
            lines.append(f"  売上高推移: {rev_str}")
            lines.append(f"              ({lbl_str})")

        # 営業利益推移
        if entry.fundamental.operating_profit_history:
            op_str = " → ".join(
                f"{v:,.0f}M" for v in entry.fundamental.operating_profit_history
            )
            lbl_str = " → ".join(entry.fundamental.operating_profit_labels)
            lines.append(f"  営業利益推移: {op_str}")
            lines.append(f"                ({lbl_str})")

        lines.append("")

        # 各スコア
        for key, detail in fa["scores"].items():
            label_map = {
                "roe": "ROE",
                "roa": "ROA",
                "equity_ratio": "自己資本比率",
                "revenue_per_employee": "1人当たり売上",
                "revenue_cagr": "売上CAGR",
                "op_profit_cagr": "営業利益CAGR",
            }
            name = label_map.get(key, key)
            bar = "█" * (detail["score"] // 10) + "░" * (10 - detail["score"] // 10)
            lines.append(
                f"  {name:<16} {detail['label']:>12}  "
                f"[{bar}] {detail['score']}/100"
            )

        lines.append("")

        # ── 成長性 ──
        lines.append("─── 成長性分析 ───")
        ga = analysis["growth"]
        lines.append(f"  カテゴリスコア: {ga['composite']}/100")
        lines.append("")

        for key, detail in ga["scores"].items():
            label_map = {
                "shikiho_revision_rate": "四季報修正率",
                "operating_profit_growth": "営業増益率",
                "revenue_growth": "売上成長率",
                "forward2_per": "来々期PER",
                "peg_ratio": "PEGレシオ",
            }
            name = label_map.get(key, key)
            bar = "█" * (detail["score"] // 10) + "░" * (10 - detail["score"] // 10)
            lines.append(
                f"  {name:<16} {detail['label']:>12}  "
                f"[{bar}] {detail['score']}/100"
            )

        lines.append("")

        # ── テクニカル ──
        lines.append("─── テクニカル分析 ───")
        lines.append(f"  カテゴリスコア: {tech_detail['score']}/100")
        lines.append(f"  判定: {tech_detail['verdict']}")
        lines.append("")

        if entry.technical.current_price:
            lines.append(f"  現在株価: {entry.technical.current_price:,.0f}円")
        if entry.technical.ma25:
            lines.append(f"  25MA: {entry.technical.ma25:,.0f}円")
        if entry.technical.ma75:
            lines.append(f"  75MA: {entry.technical.ma75:,.0f}円")
        lines.append("")

        for sig in tech_detail["signals"]:
            lines.append(f"  ● {sig}")

        if tech_detail["entry_trigger"]:
            lines.append("")
            lines.append(f"  >>> エントリーシグナル: {tech_detail['entry_trigger']}")

        if tech_detail["note"]:
            lines.append(f"  メモ: {tech_detail['note']}")

        lines.append("")

        # ── 投資判断 ──
        if entry.investment_thesis or entry.risk_factors or entry.catalysts:
            lines.append("─── 投資判断 ───")
            if entry.investment_thesis:
                lines.append(f"  投資テーマ: {entry.investment_thesis}")
            if entry.catalysts:
                lines.append("  カタリスト:")
                for c in entry.catalysts:
                    lines.append(f"    + {c}")
            if entry.risk_factors:
                lines.append("  リスク要因:")
                for r in entry.risk_factors:
                    lines.append(f"    - {r}")
            lines.append("")

        # ── IPO情報 ──
        ipo = entry.ipo_info
        if ipo.listing_date or ipo.offer_price:
            lines.append("─── IPO情報 ───")
            if ipo.listing_date:
                lines.append(f"  上場日: {ipo.listing_date}")
            if ipo.listing_market:
                lines.append(f"  市場: {ipo.listing_market}")
            if ipo.offer_price:
                lines.append(f"  公開価格: {ipo.offer_price:,.0f}円")
            if ipo.initial_price:
                lines.append(f"  初値: {ipo.initial_price:,.0f}円")
            if ipo.initial_return is not None:
                lines.append(f"  初値騰落率: {ipo.initial_return:+.1f}%")
            if ipo.lockup_expiry:
                lines.append(f"  ロックアップ解除: {ipo.lockup_expiry}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_ranking_report(self, entries: list[StockEntry]) -> str:
        """複数銘柄のランキングレポートを生成"""
        rankings = self.stock_analyzer.rank_stocks(entries)

        lines = []
        lines.append("=" * 70)
        lines.append("  IPO銘柄 総合ランキング")
        lines.append(f"  生成日: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        lines.append("")

        # サマリテーブル
        lines.append(
            f"  {'順位':<4} {'コード':<6} {'銘柄名':<12} "
            f"{'ファンダ':>8} {'成長性':>8} {'テクニカル':>8} {'総合':>8}"
        )
        lines.append("  " + "-" * 64)

        for r in rankings:
            lines.append(
                f"  {r['rank']:<4} {r['code']:<6} {r['name']:<12} "
                f"{r['fundamental']['composite']:>7.1f} "
                f"{r['growth']['composite']:>7.1f} "
                f"{r['technical_score']:>7.1f} "
                f"{r['overall_score']:>7.1f}"
            )

        lines.append("")

        # 各銘柄の詳細
        for entry in entries:
            lines.append(self.generate_stock_report(entry))
            lines.append("")

        return "\n".join(lines)

    def export_json(self, entries: list[StockEntry], output_path: str):
        """分析結果をJSONエクスポート"""
        rankings = self.stock_analyzer.rank_stocks(entries)
        output = {
            "generated_at": datetime.now().isoformat(),
            "total_stocks": len(entries),
            "rankings": rankings,
        }
        Path(output_path).write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path
