# -*- coding: utf-8 -*-
"""
slack_daily_summary.py

日次サマリ（PF / MaxDD / WF / Trades / atrq_reject_rate）をSlackへ投稿する最小スクリプト。

使い方:
  python slack_daily_summary.py \
    --results-root "C:\\Users\\bnr39\\OneDrive\\EAデータ　CSV\\バックテスト結果" \
    [--webhook "https://hooks.slack..."] [--dry-run]

備考:
- Webhookは引数 --webhook または環境変数 SLACK_WEBHOOK_URL から取得。
- Webhook未指定時は自動でドライラン（投稿せず内容を表示）に切替。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_result_dir(results_root: Path) -> Optional[Path]:
    """バックテスト結果フォルダ（metrics.json を含む）で最も新しいものを返す。"""
    try:
        if not results_root.exists():
            return None
        candidates = []
        for child in results_root.iterdir():
            try:
                if not child.is_dir():
                    continue
                metrics = child / "metrics.json"
                if metrics.exists():
                    candidates.append((child.stat().st_mtime, child))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    except Exception as e:
        logger.error(f"結果フォルダ探索エラー: {e}")
        return None


def load_metrics(metrics_path: Path) -> Dict[str, object]:
    """metrics.json を読み込む。"""
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"metrics読込エラー: {e}")
        return {}


def format_summary_text(dir_path: Path, metrics: Dict[str, object]) -> str:
    """Slack投稿用テキストを整形。"""
    try:
        trades = int(metrics.get("Trades", 0) or 0)
        pf = float(metrics.get("PF", 0.0) or 0.0)
        maxdd = float(metrics.get("MaxDD", 0.0) or 0.0)
        wf = float(metrics.get("WF_PassRate", 0.0) or 0.0)
        atrq_rate = float(metrics.get("atrq_reject_rate", 0.0) or 0.0)
        run_name = dir_path.name
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        text = (
            f"[Daily Summary] {ts}\n"
            f"Run: {run_name}\n"
            f"PF: {pf:.3f} | MaxDD: {maxdd:.2f} | WF: {wf*100:.2f}% | Trades: {trades} | atrq_reject_rate: {atrq_rate:.3f}"
        )
        return text
    except Exception as e:
        logger.error(f"サマリ整形エラー: {e}")
        return ""


def post_to_slack(webhook_url: str, message: str) -> Tuple[bool, Optional[str]]:
    """WebhookにPOST。requestsが無い環境でも動作するようurllibで実装。"""
    try:
        import urllib.request
        import urllib.error
        import json as _json

        data = _json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            if 200 <= status < 300:
                return True, None
            return False, f"Slack HTTP {status}"
    except Exception as e:
        return False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="日次サマリをSlackへ投稿")
    ap.add_argument("--results-root", required=False, default=r"C:\\Users\\bnr39\\OneDrive\\EAデータ　CSV\\バックテスト結果")
    ap.add_argument("--webhook", required=False, default=None, help="Slack Incoming Webhook URL")
    ap.add_argument("--dry-run", action="store_true", help="実際には投稿せず内容のみ出力")
    args = ap.parse_args()

    try:
        webhook_url = args.webhook or os.environ.get("SLACK_WEBHOOK_URL")
        if webhook_url is None:
            logger.info("Webhook未設定のためドライランに切り替えます（--webhook または SLACK_WEBHOOK_URL を指定してください）")
            args.dry_run = True

        results_root = Path(args.results_root)
        latest_dir = find_latest_result_dir(results_root)
        if latest_dir is None:
            logger.error("最新の結果フォルダが見つかりません。")
            return

        metrics_path = latest_dir / "metrics.json"
        metrics = load_metrics(metrics_path)
        if not metrics:
            logger.error("metrics.json の読み込みに失敗しました。")
            return

        text = format_summary_text(latest_dir, metrics)
        if not text:
            logger.error("投稿メッセージの生成に失敗しました。")
            return

        if args.dry_run:
            logger.info("[DRY-RUN] Slack投稿内容:\n" + text)
            return

        ok, err = post_to_slack(webhook_url, text)
        if not ok:
            logger.error(f"Slack投稿失敗: {err}")
            return
        logger.info("Slack投稿完了")
    except Exception as e:
        logger.error(f"処理失敗: {e}")


if __name__ == "__main__":
    main()


