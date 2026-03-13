# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:36:32 2026

@author: 86177
"""

import os
import wave
import tempfile
import whisper
from config import MODEL_SIZE

model = whisper.load_model(MODEL_SIZE)

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def transcribe_pcm16_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    将 PCM16LE 单声道字节流落成临时 wav 文件后交给 whisper 识别。
    """
    if not audio_bytes:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        result = model.transcribe(tmp_path)
        return result.get("text", "")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)