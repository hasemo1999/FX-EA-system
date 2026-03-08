"""IPO銘柄データモデル定義"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FundamentalData:
    """ファンダメンタル指標"""
    # 売上高（百万円）: 過去～予想の推移リスト
    revenue_history: list[float] = field(default_factory=list)
    revenue_labels: list[str] = field(default_factory=list)

    # 営業利益（百万円）
    operating_profit_history: list[float] = field(default_factory=list)
    operating_profit_labels: list[str] = field(default_factory=list)

    # 効率性指標
    roe: Optional[float] = None          # 自己資本利益率 (%)
    roa: Optional[float] = None          # 総資産利益率 (%)
    equity_ratio: Optional[float] = None  # 自己資本比率 (%)

    # 純利益（百万円）
    net_profit_history: list[float] = field(default_factory=list)
    net_profit_labels: list[str] = field(default_factory=list)

    # 企業効率
    employees: Optional[int] = None           # 従業員数
    avg_age: Optional[float] = None           # 平均年齢
    avg_tenure_years: Optional[float] = None  # 平均勤続年数
    avg_salary: Optional[float] = None        # 平均年収（万円）
    revenue_per_employee: Optional[float] = None  # 1人当たり売上高（万円）

    # 1株指標
    total_assets: Optional[float] = None      # 総資産（百万円）
    eps_actual: Optional[float] = None        # EPS実績（円）
    bps: Optional[float] = None               # BPS（円）

    # 財務安全性
    current_ratio: Optional[float] = None     # 流動比率 (%)
    debt_equity_ratio: Optional[float] = None # 負債資本倍率
    interest_bearing_debt: Optional[float] = None  # 有利子負債（百万円）

    # キャッシュフロー
    operating_cf: Optional[float] = None   # 営業CF（百万円）
    investing_cf: Optional[float] = None   # 投資CF（百万円）
    free_cf: Optional[float] = None        # フリーCF（百万円）


@dataclass
class GrowthData:
    """成長性指標"""
    shikiho_revision_rate: Optional[float] = None   # 四季報修正率 (%)
    operating_profit_growth: Optional[float] = None  # 営業増益率 (%)
    revenue_growth: Optional[float] = None           # 売上成長率 (%)
    forward_per: Optional[float] = None              # 来期予想PER (倍)
    forward2_per: Optional[float] = None             # 来々期予想PER (倍)
    peg_ratio: Optional[float] = None                # PEGレシオ
    eps_growth: Optional[float] = None               # EPS成長率 (%)
    market_cap: Optional[float] = None               # 時価総額（百万円）
    psr: Optional[float] = None                      # PSR (倍)


@dataclass
class TechnicalData:
    """テクニカル指標"""
    current_price: Optional[float] = None       # 現在株価
    ma25: Optional[float] = None                # 25日移動平均
    ma75: Optional[float] = None                # 75日移動平均
    ma200: Optional[float] = None               # 200日移動平均

    golden_cross_25_75: Optional[bool] = None   # 25MA/75MAゴールデンクロス
    price_vs_ma25: Optional[str] = None         # 株価 vs 25MA位置 ("above"/"below")
    price_vs_ma75: Optional[str] = None         # 株価 vs 75MA位置

    # ボリューム
    avg_volume_20d: Optional[float] = None      # 20日平均出来高
    volume_ratio: Optional[float] = None        # 出来高倍率

    # オシレータ
    rsi_14: Optional[float] = None              # RSI(14)
    macd: Optional[float] = None                # MACD
    macd_signal: Optional[float] = None         # MACDシグナル

    # サポレジ
    support_price: Optional[float] = None       # サポートライン
    resistance_price: Optional[float] = None    # レジスタンスライン
    entry_trigger: Optional[str] = None         # エントリーシグナル条件

    # 評価 (1-5)
    technical_score: Optional[int] = None       # テクニカル評価スコア
    technical_note: Optional[str] = None        # テクニカルメモ


@dataclass
class IPOInfo:
    """IPO固有情報"""
    listing_date: Optional[str] = None          # 上場日
    listing_market: Optional[str] = None        # 上場市場
    offer_price: Optional[float] = None         # 公開価格
    initial_price: Optional[float] = None       # 初値
    initial_return: Optional[float] = None      # 初値騰落率 (%)
    lockup_expiry: Optional[str] = None         # ロックアップ解除日
    underwriter: Optional[str] = None           # 主幹事
    shares_outstanding: Optional[int] = None    # 発行済株式数
    ipo_market_cap: Optional[float] = None      # IPO時時価総額（百万円）
    founder: Optional[str] = None               # 創業者
    founder_ownership_pct: Optional[float] = None  # 創業者持株比率 (%)
    founder_note: Optional[str] = None          # 創業者メモ


@dataclass
class StockEntry:
    """銘柄エントリー（全データ統合）"""
    # 基本情報
    code: str = ""                # 銘柄コード
    name: str = ""                # 銘柄名
    sector: str = ""              # セクター/業種
    business_description: str = ""  # 事業内容
    analysis_date: str = ""       # 分析日

    # 総合ランク
    overall_rank: Optional[int] = None    # 総合順位
    overall_score: Optional[float] = None  # 総合スコア (0-100)
    rank_label: str = ""                   # ランクラベル (例: "🥇 1位")
    tagline: str = ""                      # キャッチフレーズ

    # 各カテゴリデータ
    fundamental: FundamentalData = field(default_factory=FundamentalData)
    growth: GrowthData = field(default_factory=GrowthData)
    technical: TechnicalData = field(default_factory=TechnicalData)
    ipo_info: IPOInfo = field(default_factory=IPOInfo)

    # 投資判断
    investment_thesis: str = ""       # 投資テーマ/仮説
    risk_factors: list[str] = field(default_factory=list)  # リスク要因
    catalysts: list[str] = field(default_factory=list)     # カタリスト
