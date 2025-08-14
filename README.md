# EA投資プロジェクト - バックテストシステム

USD/JPYの自動売買戦略のバックテストシステムです。EMA9/EMA21のゴールデンクロス/デッドクロス + ボリューム比率フィルタ + MTF（Multi-Timeframe）分析による戦略を実装しています。

## 🚀 機能

### 基本機能
- **OHLCVデータ解析**: CSVファイルからの自動読み込み
- **EMA指標計算**: 9期間/21期間の指数移動平均
- **MTF分析**: 上位足（60分）での方向性フィルタ
- **Walk-Forward分析**: 7分割での戦略安定性検証
- **コストモデル**: スプレッド、スリッページ、手数料考慮

### 改善機能
- **時間帯フィルタ**: 金曜日特別処理（15:30まで延長）
- **利確ロジック**: 固定TP（2.2）またはATRベース可変TP
- **方向性制御**: ロング/ショート/両方向対応
- **期間フィルタ**: 開始日/終了日指定
- **結果整理**: タイムスタンプ付きフォルダ自動作成

## 📁 ファイル構成

```
EA投資プロジェクト/
├── rough_backtest.py              # メインバックテストスクリプト
├── backtest_config_*.json         # 設定ファイル群
├── requirements.txt               # 依存関係
└── README.md                      # このファイル
```

## 🛠️ セットアップ

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. データファイルの配置
CSVファイルを以下のディレクトリに配置：
```
C:\Users\bnr39\OneDrive\EAデータ　CSV\バックテスト素材\
```

### 3. 設定ファイルの編集
`backtest_config_improved.json`を編集してパラメータを調整

## 📊 使用方法

### 基本実行
```bash
python rough_backtest.py --cfg backtest_config_improved.json
```

### 設定変更例
```bash
# MTF無効
python rough_backtest.py --cfg backtest_config.json --use_mtf false

# 上位足CSV使用
python rough_backtest.py --cfg backtest_config.json --higher_tf_mode csv
```



## ⚙️ 設定パラメータ

| パラメータ | 説明 | デフォルト |
|------------|------|------------|
| `ema_fast` | 短期EMA期間 | 9 |
| `ema_slow` | 長期EMA期間 | 21 |
| `vol_ratio_th` | ボリューム比率閾値 | 1.3 |
| `stop_pts` | 損切りポイント | 1.0 |
| `tp_pts` | 利確ポイント | 2.2 |
| `spread_pts` | スプレッド | 0.02 |
| `slippage_pts` | スリッページ | 0.01 |
| `direction` | 取引方向 | "long" |
| `session_start` | 取引開始時間 | "00:00" |
| `session_end` | 取引終了時間 | "23:59" |

## 📈 最新結果（2023-2024年）

| 項目 | 値 |
|------|-----|
| **取引数** | 155回 |
| **Profit Factor** | 1.151 |
| **勝率** | 49.0% |
| **純利益** | 10.36ポイント |
| **最大ドローダウン** | -6.87ポイント |
| **Walk-Forward合格率** | 83.33% |

## 🎯 目標KPI

| 目標 | 現在値 | 達成状況 |
|------|--------|----------|
| **PF ≥ 1.30** | 1.151 | ❌ 未達成 |
| **WF合格率 ≥ 70%** | 83.33% | ✅ 達成 |
| **MaxDD ≤ 10%** | 6.87% | ✅ 達成 |

## 🔧 技術仕様

- **Python**: 3.11+
- **主要ライブラリ**: pandas, numpy, matplotlib
- **データ形式**: CSV（OHLCV）
- **タイムゾーン**: UTC/Asia/Tokyo対応
- **エラーハンドリング**: try-except + ログ出力
- **型ヒント**: 全関数に型アノテーション

## 📝 ライセンス

MIT License

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

## ⚠️ 免責事項

このシステムは教育・研究目的で作成されています。実際の投資には十分な検証が必要です。

