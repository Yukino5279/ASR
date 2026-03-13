# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:36:32 2026

@author: 86177
"""

import os
import wave
import tempfile 
from typing import Any
import numpy as np
import whisper
from config import MODEL_SIZE

model = whisper.load_model(MODEL_SIZE)

def _format_timestamp(seconds: float) -> str:
    """将秒数格式化为字幕友好的 mm:ss.mmm 字符串。"""
    if seconds < 0:
        seconds = 0
    milliseconds = int(round(seconds * 1000))
    minutes, ms_rest = divmod(milliseconds, 60_000)
    secs, ms = divmod(ms_rest, 1_000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _build_subtitle_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """从 whisper 结果中提取带时间戳的字幕段。"""
    subtitles: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        subtitles.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "start_label": _format_timestamp(start),
                "end_label": _format_timestamp(end),
                "text": text,
            }
        )
    return subtitles


def transcribe_audio(file_path: str) -> str:
    result = model.transcribe(file_path, language="zh", task="transcribe", fp16=False)
    """
        向后兼容：仍然返回纯文本识别结果。
    """
    return result.get("text", "").strip()

def transcribe_pcm16_subtitles(audio_bytes: bytes, sample_rate: int = 16000) -> dict[str, Any]:
    """
    将 PCM16LE 单声道字节流转换为实时字幕段。
    优先走内存 numpy，避免频繁写临时文件，提升吞吐。
    """
    if not audio_bytes:
        return {"text": "", "subtitles": []}

    pcm_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm_data.size == 0:
        return {"text": "", "subtitles": []}

    result = model.transcribe(
        pcm_data,
        language="zh",
        task="transcribe",
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
    )

    subtitles = _build_subtitle_segments(result)
    text = result.get("text", "").strip()
    return {"text": text, "subtitles": subtitles}