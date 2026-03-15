# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:36:32 2026

@author: 86177
"""


from typing import Any
import numpy as np
import whisper
from config import MODEL_SIZE

model = whisper.load_model(MODEL_SIZE)

MIN_RMS_FOR_SPEECH = 0.006  # 经验阈值：过滤静音和背景噪声，抑制幻觉文本

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

def _audio_rms(pcm_data: np.ndarray) -> float:
    if pcm_data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm_data), dtype=np.float32)))


def _extract_text(result: dict[str, Any]) -> str:
    text = str(result.get("text", "")).strip()
    if text:
        return text
    segment_texts = [str(seg.get("text", "")).strip() for seg in result.get("segments", [])]
    merged = " ".join([t for t in segment_texts if t]).strip()
    return merged


def transcribe_audio(file_path: str) -> str:
    """
    文件转写：优先自动语言检测，若结果为空则回退到中文提示再试一次。
    避免 /asr 因语言参数或边界片段导致空文本。
    """
    base_kwargs = {
        "task": "transcribe",
        "fp16": False,
        "temperature": 0,
        "condition_on_previous_text": False,
    }

    result_auto = model.transcribe(file_path, **base_kwargs)
    text_auto = _extract_text(result_auto)
    if text_auto:
        return text_auto

    result_zh = model.transcribe(file_path, **base_kwargs)
    return _extract_text(result_zh)


def transcribe_pcm16_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    向后兼容：仍然返回纯文本识别结果。
    """
    return transcribe_pcm16_subtitles(audio_bytes, sample_rate=sample_rate)["text"]



def transcribe_pcm16_subtitles(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    initial_prompt: str | None = None,  #提示词可以为：字符串或空
) -> dict[str, Any]:
    """
    将 PCM16LE 单声道字节流转换为实时字幕段。
    使用静音过滤+上下文提示降低重复幻觉概率。
    """
    if not audio_bytes:
        return {"text": "", "subtitles": []}

    pcm_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm_data.size == 0:
        return {"text": "", "subtitles": []}

    if _audio_rms(pcm_data) < MIN_RMS_FOR_SPEECH:
        return {"text": "", "subtitles": []}

    result = model.transcribe(
        pcm_data,
        # language="zh",
        task="transcribe",          #(translate)翻译成英文或者(transcribe)原本语言
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,   #当前识别不依赖之前的文本
        initial_prompt=initial_prompt,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    subtitles = _build_subtitle_segments(result)
    text = result.get("text", "").strip()
    return {"text": text, "subtitles": subtitles}