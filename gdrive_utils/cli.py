# -*- coding: utf-8 -*-
"""CLI エントリーポイント — python -m gdrive_utils <command>"""

import argparse
import logging
import sys

import pandas as pd

from gdrive_utils.config import GDriveConfig
from gdrive_utils.drive import DriveClient
from gdrive_utils.sheets import SheetsClient
from gdrive_utils.sync import SyncEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cmd_upload(args):
    cfg = GDriveConfig.load(args.config)
    dc = DriveClient(cfg)
    fid = dc.upload_file(args.file, folder_id=args.folder_id)
    print(f"Uploaded: {args.file} -> {fid}")


def cmd_download(args):
    cfg = GDriveConfig.load(args.config)
    dc = DriveClient(cfg)
    if args.name:
        dc.download_by_name(args.name, args.output, folder_id=args.folder_id)
    else:
        dc.download_file(args.file_id, args.output)
    print(f"Downloaded -> {args.output}")


def cmd_list(args):
    cfg = GDriveConfig.load(args.config)
    dc = DriveClient(cfg)
    files = dc.list_files(folder_id=args.folder_id, name_pattern=args.pattern)
    for f in files:
        size = f.get("size", "?")
        print(f"{f['id']}  {f['name']:40s}  {size:>10s}  {f.get('modifiedTime', '')}")


def cmd_delete(args):
    cfg = GDriveConfig.load(args.config)
    dc = DriveClient(cfg)
    dc.delete_file(args.file_id)
    print(f"Deleted: {args.file_id}")


def cmd_sync(args):
    cfg = GDriveConfig.load(args.config)
    if args.folder_id:
        cfg.folder_id = args.folder_id
    if args.local_dir:
        cfg.sync_root = args.local_dir
    dc = DriveClient(cfg)
    engine = SyncEngine(dc, cfg)
    result = engine.sync(direction=args.direction)
    print(f"Sync complete: {result}")


def cmd_sheets_read(args):
    cfg = GDriveConfig.load(args.config)
    sc = SheetsClient(cfg)
    df = sc.read_sheet(args.spreadsheet_id, args.worksheet)
    print(df.to_string())


def cmd_sheets_write(args):
    cfg = GDriveConfig.load(args.config)
    sc = SheetsClient(cfg)
    df = pd.read_csv(args.csv_file)
    sc.write_sheet(df, args.spreadsheet_id, args.worksheet, mode=args.mode)
    print(f"Wrote {len(df)} rows to {args.spreadsheet_id}/{args.worksheet}")


def main():
    parser = argparse.ArgumentParser(prog="gdrive_utils", description="Google Drive/Sheets CLI")
    parser.add_argument("--config", "-c", default=None, help="gdrive_config.json パス")
    sub = parser.add_subparsers(dest="command")

    # upload
    p_up = sub.add_parser("upload", help="ファイルをDriveにアップロード")
    p_up.add_argument("file", help="アップロードするローカルファイル")
    p_up.add_argument("--folder-id", default=None)
    p_up.set_defaults(func=cmd_upload)

    # download
    p_dl = sub.add_parser("download", help="Driveからファイルをダウンロード")
    p_dl.add_argument("--file-id", default=None, help="Drive file ID")
    p_dl.add_argument("--name", default=None, help="ファイル名で検索してダウンロード")
    p_dl.add_argument("--output", "-o", required=True, help="出力先パス")
    p_dl.add_argument("--folder-id", default=None)
    p_dl.set_defaults(func=cmd_download)

    # list
    p_ls = sub.add_parser("list", help="Driveフォルダ内のファイル一覧")
    p_ls.add_argument("--folder-id", default=None)
    p_ls.add_argument("--pattern", default=None, help="名前フィルタ")
    p_ls.set_defaults(func=cmd_list)

    # delete
    p_rm = sub.add_parser("delete", help="Driveファイルを削除")
    p_rm.add_argument("file_id", help="削除するファイルのID")
    p_rm.set_defaults(func=cmd_delete)

    # sync
    p_sync = sub.add_parser("sync", help="ローカル⇔Drive双方向同期")
    p_sync.add_argument("local_dir", nargs="?", default=None, help="ローカルディレクトリ")
    p_sync.add_argument("--folder-id", default=None)
    p_sync.add_argument("--direction", choices=["both", "upload", "download"], default=None)
    p_sync.set_defaults(func=cmd_sync)

    # sheets read
    p_sr = sub.add_parser("sheets-read", help="Google Sheetsを読み取り")
    p_sr.add_argument("spreadsheet_id")
    p_sr.add_argument("--worksheet", "-w", default="Sheet1")
    p_sr.set_defaults(func=cmd_sheets_read)

    # sheets write
    p_sw = sub.add_parser("sheets-write", help="CSVをGoogle Sheetsに書き込み")
    p_sw.add_argument("spreadsheet_id")
    p_sw.add_argument("csv_file", help="書き込むCSVファイル")
    p_sw.add_argument("--worksheet", "-w", default="Sheet1")
    p_sw.add_argument("--mode", choices=["replace", "append"], default="replace")
    p_sw.set_defaults(func=cmd_sheets_write)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
