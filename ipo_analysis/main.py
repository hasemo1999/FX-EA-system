#!/usr/bin/env python3
"""IPO株式投資 データ分析・収集システム - メインエントリーポイント

使い方:
  python -m ipo_analysis.main                    # 全銘柄レポート表示
  python -m ipo_analysis.main --code 418A        # 特定銘柄の詳細レポート
  python -m ipo_analysis.main --export result.json  # JSON出力
  python -m ipo_analysis.main --ranking          # ランキング表示
"""
import argparse
import json
import sys
from pathlib import Path
from .models import (
    StockEntry, FundamentalData, GrowthData,
    TechnicalData, IPOInfo,
)
from .analyzer import StockAnalyzer
from .report import ReportGenerator


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATA_FILE = DATA_DIR / "stocks.json"


def load_stocks(data_file: Path = DEFAULT_DATA_FILE) -> list[StockEntry]:
    """JSONファイルから銘柄データを読み込み"""
    if not data_file.exists():
        print(f"データファイルが見つかりません: {data_file}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(data_file.read_text(encoding="utf-8"))
    entries = []

    for s in raw.get("stocks", []):
        fund_raw = s.get("fundamental", {})
        fundamental = FundamentalData(
            revenue_history=fund_raw.get("revenue_history", []),
            revenue_labels=fund_raw.get("revenue_labels", []),
            operating_profit_history=fund_raw.get("operating_profit_history", []),
            operating_profit_labels=fund_raw.get("operating_profit_labels", []),
            net_profit_history=fund_raw.get("net_profit_history", []),
            net_profit_labels=fund_raw.get("net_profit_labels", []),
            roe=fund_raw.get("roe"),
            roa=fund_raw.get("roa"),
            equity_ratio=fund_raw.get("equity_ratio"),
            employees=fund_raw.get("employees"),
            avg_age=fund_raw.get("avg_age"),
            avg_tenure_years=fund_raw.get("avg_tenure_years"),
            avg_salary=fund_raw.get("avg_salary"),
            revenue_per_employee=fund_raw.get("revenue_per_employee"),
            total_assets=fund_raw.get("total_assets"),
            eps_actual=fund_raw.get("eps_actual"),
            bps=fund_raw.get("bps"),
            current_ratio=fund_raw.get("current_ratio"),
            debt_equity_ratio=fund_raw.get("debt_equity_ratio"),
            interest_bearing_debt=fund_raw.get("interest_bearing_debt"),
            operating_cf=fund_raw.get("operating_cf"),
            investing_cf=fund_raw.get("investing_cf"),
            free_cf=fund_raw.get("free_cf"),
        )

        growth_raw = s.get("growth", {})
        growth = GrowthData(
            shikiho_revision_rate=growth_raw.get("shikiho_revision_rate"),
            operating_profit_growth=growth_raw.get("operating_profit_growth"),
            revenue_growth=growth_raw.get("revenue_growth"),
            forward_per=growth_raw.get("forward_per"),
            forward2_per=growth_raw.get("forward2_per"),
            peg_ratio=growth_raw.get("peg_ratio"),
            eps_growth=growth_raw.get("eps_growth"),
            market_cap=growth_raw.get("market_cap"),
            psr=growth_raw.get("psr"),
        )

        tech_raw = s.get("technical", {})
        technical = TechnicalData(
            current_price=tech_raw.get("current_price"),
            ma25=tech_raw.get("ma25"),
            ma75=tech_raw.get("ma75"),
            ma200=tech_raw.get("ma200"),
            golden_cross_25_75=tech_raw.get("golden_cross_25_75"),
            price_vs_ma25=tech_raw.get("price_vs_ma25"),
            price_vs_ma75=tech_raw.get("price_vs_ma75"),
            avg_volume_20d=tech_raw.get("avg_volume_20d"),
            volume_ratio=tech_raw.get("volume_ratio"),
            rsi_14=tech_raw.get("rsi_14"),
            macd=tech_raw.get("macd"),
            macd_signal=tech_raw.get("macd_signal"),
            support_price=tech_raw.get("support_price"),
            resistance_price=tech_raw.get("resistance_price"),
            entry_trigger=tech_raw.get("entry_trigger"),
            technical_score=tech_raw.get("technical_score"),
            technical_note=tech_raw.get("technical_note"),
        )

        ipo_raw = s.get("ipo_info", {})
        ipo_info = IPOInfo(
            listing_date=ipo_raw.get("listing_date"),
            listing_market=ipo_raw.get("listing_market"),
            offer_price=ipo_raw.get("offer_price"),
            initial_price=ipo_raw.get("initial_price"),
            initial_return=ipo_raw.get("initial_return"),
            lockup_expiry=ipo_raw.get("lockup_expiry"),
            underwriter=ipo_raw.get("underwriter"),
            shares_outstanding=ipo_raw.get("shares_outstanding"),
            ipo_market_cap=ipo_raw.get("ipo_market_cap"),
            founder=ipo_raw.get("founder"),
            founder_ownership_pct=ipo_raw.get("founder_ownership_pct"),
            founder_note=ipo_raw.get("founder_note"),
        )

        entry = StockEntry(
            code=s.get("code", ""),
            name=s.get("name", ""),
            sector=s.get("sector", ""),
            business_description=s.get("business_description", ""),
            analysis_date=s.get("analysis_date", ""),
            overall_rank=s.get("overall_rank"),
            rank_label=s.get("rank_label", ""),
            tagline=s.get("tagline", ""),
            fundamental=fundamental,
            growth=growth,
            technical=technical,
            ipo_info=ipo_info,
            investment_thesis=s.get("investment_thesis", ""),
            risk_factors=s.get("risk_factors", []),
            catalysts=s.get("catalysts", []),
        )
        entries.append(entry)

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="IPO株式投資 データ分析システム"
    )
    parser.add_argument(
        "--code", type=str, default=None,
        help="特定銘柄コードの詳細レポート表示",
    )
    parser.add_argument(
        "--ranking", action="store_true",
        help="全銘柄ランキングレポート表示",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="分析結果をJSONファイルにエクスポート",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="銘柄データJSONファイルパス (デフォルト: data/stocks.json)",
    )
    args = parser.parse_args()

    data_file = Path(args.data) if args.data else DEFAULT_DATA_FILE
    entries = load_stocks(data_file)

    if not entries:
        print("銘柄データが空です。data/stocks.json にデータを追加してください。")
        sys.exit(1)

    reporter = ReportGenerator()

    if args.code:
        target = [e for e in entries if e.code == args.code]
        if not target:
            print(f"銘柄コード '{args.code}' が見つかりません。")
            sys.exit(1)
        print(reporter.generate_stock_report(target[0]))

    elif args.ranking or len(entries) > 1:
        print(reporter.generate_ranking_report(entries))

    else:
        # 1銘柄のみの場合は詳細レポート
        print(reporter.generate_stock_report(entries[0]))

    if args.export:
        path = reporter.export_json(entries, args.export)
        print(f"\nJSON出力: {path}")


if __name__ == "__main__":
    main()
