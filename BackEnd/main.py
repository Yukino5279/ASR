# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:38:27 2026

@author: 86177
"""

from fastapi import FastAPI, UploadFile, File, WebSocket,WebSocketDisconnect
import os
from asr_service import transcribe_audio, transcribe_pcm16_bytes
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
async def stream_speech_to_text(websocket: WebSocket):
    """
    实时语音识别 WebSocket 接口（简化版）

    客户端约定:
    1) 发送二进制音频分片（二进制消息，PCM16LE, mono, 16kHz）。
    2) 发送文本消息 "flush": 立即返回当前累计音频的中间识别结果。
    3) 发送文本消息 "end": 返回最终识别结果并关闭连接。

    返回消息(JSON):
    - {"type": "partial", "text": "..."}
    - {"type": "final", "text": "..."}
    - {"type": "error", "message": "..."}
    """
    await websocket.accept()

    audio_buffer = bytearray()
    sample_rate = 16000
    bytes_per_second = sample_rate * 2  # PCM16 单声道 -> 每秒 16000 * 2 字节
    min_bytes_for_partial = 2 * bytes_per_second  # 每累计约 2 秒语音做一次中间结果

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                audio_buffer.extend(message["bytes"])

                if len(audio_buffer) >= min_bytes_for_partial:
                    partial_text = transcribe_pcm16_bytes(bytes(audio_buffer), sample_rate=sample_rate)
                    await websocket.send_json({"type": "partial", "text": partial_text})

            elif "text" in message and message["text"] is not None:
                command = message["text"].strip().lower()

                if command == "flush":
                    partial_text = transcribe_pcm16_bytes(bytes(audio_buffer), sample_rate=sample_rate)
                    await websocket.send_json({"type": "partial", "text": partial_text})
                elif command == "end":
                    final_text = transcribe_pcm16_bytes(bytes(audio_buffer), sample_rate=sample_rate)
                    await websocket.send_json({"type": "final", "text": final_text})
                    await websocket.close()
                    break
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown command. Use 'flush' or 'end'."
                    })

    except WebSocketDisconnect:
        # 客户端主动断开
        return
