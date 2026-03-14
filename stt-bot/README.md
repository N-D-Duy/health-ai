# stt-bot (Prototype)

AI medical scribe prototype cho tieng Viet:

`audio -> whisper STT -> qwen2.5:7b extraction -> SOAP draft -> doctor review -> finalize`

## 1) Setup

```bash
cd stt-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Yeu cau he thong:
- Da chay Ollama va da pull model `qwen2.5:7b`
- Co `ffmpeg` trong PATH
- GPU CUDA (khuyen nghi cho `faster-whisper`)

## 2) Run API

```bash
uvicorn main:app --reload --port 8010
```

## 3) Test nhanh

- Mo file `test.html` tren browser
- Dat Base URL: `http://localhost:8010`
- Chay theo thu tu:
  1. Start Session
  2. Stop Session
  3. Upload Audio
  4. Transcribe
  5. Analyze (Extraction + SOAP)
  6. Save Review
  7. Finalize

## 3.1) Tao audio test tu text dinh nghia san

```bash
cd stt-bot
source .venv/bin/activate

# Tao 1 file theo preset (mac dinh ho_hap)
python3 scripts/generate_test_audio.py

# Tao toan bo preset trong scripts/defined_texts_vi.json
python3 scripts/generate_test_audio.py --all

# Tao tu text tuy chinh
python3 scripts/generate_test_audio.py --text "Bac si: Anh met moi tu bao gio? Benh nhan: Em met 2 ngay nay." --output-name custom_case_01
```

File output mac dinh: `data/uploads/samples/*.mp3`

## 4) Endpoints chinh

- `GET /health`
- `POST /sessions/start`
- `POST /sessions/{id}/stop`
- `POST /sessions/{id}/audio`
- `POST /sessions/{id}/transcribe`
- `GET /sessions/{id}/transcript`
- `POST /sessions/{id}/analyze`
- `GET /sessions/{id}/clinical-note`
- `GET /sessions/{id}/review`
- `POST /sessions/{id}/review`
- `POST /sessions/{id}/finalize`
