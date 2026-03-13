const apiBase = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadResult = document.getElementById("uploadResult");

const startBtn = document.getElementById("startBtn");
const flushBtn = document.getElementById("flushBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const partialResult = document.getElementById("partialResult");
const finalResult = document.getElementById("finalResult");
const subtitlePreview = document.getElementById("subtitlePreview");

let ws = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;

function setStatus(text) {
  statusEl.textContent = `状态：${text}`;
}

function renderSubtitles(subtitles = []) {
  subtitlePreview.value = subtitles.map((line) => `[${line.start_label} - ${line.end_label}] ${line.text}`).join("\n");
}


uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    uploadResult.textContent = "请先选择音频文件。";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadResult.textContent = "识别中...";
  try {
    const resp = await fetch(`${apiBase}/asr`, { method: "POST", body: formData });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    uploadResult.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    uploadResult.textContent = `上传识别失败: ${err.message}`;
  }
});

startBtn.addEventListener("click", async () => {
  try {
    const wsUrl = apiBase.replace("http", "ws") + "/asr/stream";
    ws = new WebSocket(wsUrl);

    ws.onopen = () => setStatus("WebSocket 已连接，正在采集音频...");
    ws.onerror = () => setStatus("WebSocket 出错");
    ws.onclose = () => setStatus("WebSocket 已关闭");

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "partial") {
          partialResult.value = msg.text || "";
          renderSubtitles(msg.subtitles || []);
        }
        if (msg.type === "final") {
          finalResult.value = msg.text || "";
          renderSubtitles(msg.subtitles || []);
        }
      } catch {
        // ignore non-json message
      }
    };

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        noiseSuppression: true,
        echoCancellation: true,
        autoGainControl: true,
      },
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // 这里使用 ScriptProcessorNode 实现简洁的 demo 采样
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    processorNode.onaudioprocess = (event) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const inputData = event.inputBuffer.getChannelData(0);
      const pcm16 = floatTo16BitPCM(downsampleBuffer(inputData, audioContext.sampleRate, 16000));
      ws.send(pcm16);
    };

    startBtn.disabled = true;
    flushBtn.disabled = false;
    stopBtn.disabled = false;
    partialResult.value = "";
    finalResult.value = "";
    subtitlePreview.value = "";
  } catch (err) {
    setStatus(`启动失败: ${err.message}`);
    cleanupAudio();
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  }
});

flushBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("flush");
  }
});

stopBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("end");
  }
  cleanupAudio();
  startBtn.disabled = false;
  flushBtn.disabled = true;
  stopBtn.disabled = true;
});

function cleanupAudio() {
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate >= inputRate) {
    return buffer;
  }
  const ratio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output.buffer;
}
