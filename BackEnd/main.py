# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:38:27 2026

@author: 86177
"""

from fastapi import FastAPI, UploadFile, File, WebSocket,WebSocketDisconnect
import os
from asr_service import transcribe_audio, transcribe_pcm16_subtitles
from config import UPLOAD_FOLDER
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500"
    ]  ,# 允许前端访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法
    allow_headers=["*"],  # 允许所有header
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    milliseconds = int(round(seconds * 1000))
    minutes, ms_rest = divmod(milliseconds, 60_000)
    secs, ms = divmod(ms_rest, 1_000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"

@app.get("/")
def root():
    return {"message": "TEST"}

@app.post("/asr")
async def speech_to_text(file: UploadFile = File(...)):     #声明异步函数，接收上传文件
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())     #等待异步任务完成

    text = transcribe_audio(file_path)

    return {
        "filename": file.filename,
        "text": text
    }
@app.websocket("/asr/stream")
async def stream_speech_to_subtitle(websocket: WebSocket):
    """真正增量字幕 WebSocket 接口。"""
    await websocket.accept()

   
    sample_rate = 16000
    bytes_per_second = sample_rate * 2
    min_bytes_for_partial = int(1.5 * bytes_per_second)
    overlap_bytes = int(0.4 * bytes_per_second)  # 小重叠窗口用于保留上下文

    audio_buffer = bytearray()
    committed_subtitles: list[dict] = []
    processed_bytes = 0
    last_emitted_end = 0.0

    def transcribe_incremental(force: bool = False) -> dict | None:
        nonlocal processed_bytes, last_emitted_end

        unread_bytes = len(audio_buffer) - processed_bytes
        if unread_bytes <= 0:
            return None
        if not force and unread_bytes < min_bytes_for_partial:
            return None

        chunk_start = max(0, processed_bytes - overlap_bytes)
        chunk_bytes = bytes(audio_buffer[chunk_start:])
        result = transcribe_pcm16_subtitles(chunk_bytes, sample_rate=sample_rate)

        for subtitle in result.get("subtitles", []):
            abs_start = round(float(subtitle.get("start", 0.0)) + chunk_start / bytes_per_second, 3)
            abs_end = round(float(subtitle.get("end", abs_start)) + chunk_start / bytes_per_second, 3)
            text = str(subtitle.get("text", "")).strip()
            if not text:
                continue

            # 去重：只接收时间上晚于已提交末尾的字幕
            if abs_end <= last_emitted_end + 0.05:
                continue

            subtitle_item = {
                "start": abs_start,
                "end": abs_end,
                "start_label": _format_timestamp(abs_start),
                "end_label": _format_timestamp(abs_end),
                "text": text,
            }
            committed_subtitles.append(subtitle_item)
            last_emitted_end = max(last_emitted_end, abs_end)

        processed_bytes = len(audio_buffer)
        merged_text = "".join([s["text"] for s in committed_subtitles]).strip()
        return {
            "text": merged_text,
            "subtitles": committed_subtitles,
            "latest_subtitle": committed_subtitles[-1] if committed_subtitles else None,
        }

    def snapshot_payload() -> dict:
        merged_text = "".join([s["text"] for s in committed_subtitles]).strip()
        return {
            "text": merged_text,
            "subtitles": committed_subtitles,
            "latest_subtitle": committed_subtitles[-1] if committed_subtitles else None,
        }


    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                audio_buffer.extend(message["bytes"])

                partial = transcribe_incremental(force=False)
                if partial is not None:
                    await websocket.send_json({"type": "partial", **partial})

            elif "text" in message and message["text"] is not None:
                command = message["text"].strip().lower()

                if command == "flush":
                    partial = transcribe_incremental(force=True) or snapshot_payload()
                    await websocket.send_json({"type": "partial", **partial})
                elif command == "end":
                    final_result = transcribe_incremental(force=True) or snapshot_payload()
                    await websocket.send_json({"type": "final", **final_result})
                    await websocket.close()
                    break
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown command. Use 'flush' or 'end'."
                    })

    except WebSocketDisconnect: # 客户端主动断开
        return
