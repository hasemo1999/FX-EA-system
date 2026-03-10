# -*- coding: utf-8 -*-
"""指数バックオフデコレータ"""

import functools
import logging
import time

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


def retry_on_api_error(max_retries: int = 3, base_delay: float = 1.0):
    """Google API の一時的エラーに対して指数バックオフでリトライするデコレータ。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from googleapiclient.errors import HttpError
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except HttpError as e:
                    last_error = e
                    if e.resp.status not in RETRYABLE_STATUS_CODES:
                        raise
                    if attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "API error %d (attempt %d/%d), retrying in %.1fs: %s",
                        e.resp.status, attempt + 1, max_retries, delay, e
                    )
                    time.sleep(delay)
            raise last_error  # unreachable but satisfies type checker
        return wrapper
    return decorator
