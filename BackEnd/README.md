# ASR
导入所需的包。
必要なパケットを導入する。
# 1. 创建虚拟环境 (在 ASR\BackEnd 目录下)

python -m venv venv

# 2. 激活环境
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. 一次性安装所有核心包
pip install openai-whisper numpy fastapi uvicorn python-multipart websockets