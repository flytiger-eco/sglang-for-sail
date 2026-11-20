"""Overlap image-only prompt tokenization with independent Encoder work."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


@lru_cache(maxsize=1)
def _executor():
    logging.getLogger(__name__).info("Independent Encoder prompt tokenization overlap enabled (2 workers)")
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="epd-text")


def prepare_text(processor, prompt, items):
    """Return raw prompt IDs; visual expansion still uses the original processor."""
    if os.environ.get("SGLANG_ENCODER_TEXT_OVERLAP", "0") != "1":
        return None
    if not isinstance(prompt, str) or type(processor).__name__ != "QwenVLImageProcessor":
        return None
    if not items or not all(
        str(item.get("modality")).lower().split(".")[-1] == "image" for item in items
    ):
        return None
    return asyncio.get_running_loop().run_in_executor(
        _executor(), processor._tokenizer.encode, prompt
    )
