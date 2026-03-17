# stt-bot (Medical Scribe Prototype)

`stt-bot` là prototype backend hỗ trợ bác sĩ ghi chép bệnh án từ hội thoại khám bệnh tiếng Việt.

Mục tiêu của project:
- Nhận file audio khám bệnh.
- Chuyển giọng nói thành transcript (STT).
- Trích xuất thông tin lâm sàng quan trọng.
- Tạo bản nháp SOAP note.
- Hỗ trợ bác sĩ review và finalize.

Pipeline tổng quát:
```text
audio
  -> transcribe (faster-whisper)
  -> extraction JSON (Ollama)
  -> SOAP draft JSON (Ollama)
  -> doctor review
  -> finalize
```

Tài liệu này giải thích chi tiết để người mới cũng hiểu được từng thành phần, dữ liệu vào/ra và lý do thiết kế.

---

## 1. Bài toán hệ thống đang giải

Khi khám bệnh, bác sĩ thường mất nhiều thời gian cho việc:
- Nghe lại audio/ghi nhớ nội dung.
- Trích xuất triệu chứng, thời điểm khởi phát, điểm phụ.
- Viết lại SOAP note.

`stt-bot` tự động hóa phần nháp ban đầu, nhưng vẫn giữ `human-in-the-loop`:
- Model tạo nháp.
- Bác sĩ review, sửa.
- Chỉ sau review mới được finalize.

---

## 2. Thành phần sử dụng, vai trò và tại sao dùng

### 2.1 API layer
- Công nghệ: `FastAPI` + `uvicorn`
- Vai trò:
  - Quản lý lifecycle session (start/stop/upload/transcribe/analyze/review/finalize).
  - Cung cấp endpoint để frontend/test harness gọi.
- Lý do dùng:
  - Build REST API nhanh, rõ ràng schema, dễ mở rộng.

### 2.2 STT engine
- Công nghệ: `faster-whisper`
- Vai trò:
  - Chuyển file audio thành transcript segment + full_text.
- Lý do dùng:
  - Hiệu năng tốt hơn whisper thường trong setup local.
  - Hỗ trợ GPU (`cuda`) và fallback CPU nếu CUDA lỗi.

### 2.3 LLM processing
- Công nghệ: `Ollama` + model `qwen2.5:7b`
- Vai trò:
  - Trích xuất thông tin lâm sàng có cấu trúc JSON (`ClinicalExtraction`).
  - Tạo nháp SOAP JSON (`SoapDraft`).
- Lý do dùng:
  - Chạy local để bảo mật dữ liệu và giảm phụ thuộc cloud.
  - Dễ ép model output JSON cho downstream processing.

### 2.4 Data persistence
- Công nghệ: JSON store file (`data/store/sessions.json`) qua `SessionStore`.
- Vai trò:
  - Lưu sessions, transcript, extraction, soap, review, finalization.
- Lý do dùng:
  - Đơn giản cho prototype.
  - Dễ inspect dữ liệu bằng tay trong quá trình dev.

### 2.5 Upload storage
- Thư mục mặc định: `data/uploads/`
- Vai trò:
  - Lưu file audio đã upload theo tên `session_id + ext`.

---

## 3. Kiến trúc module trong code

```text
main.py
  -> SessionStore (src/services/session_store.py)
  -> SttService (src/services/stt_service.py)
  -> ClinicalNlpService (src/services/clinical_nlp_service.py)
  -> SoapService (src/services/soap_service.py)
  -> OllamaClient (src/services/ollama_client.py)
```

Vai trò từng service:
- `SttService`:
  - Lazy-load Whisper model.
  - Transcribe audio -> `TranscriptResult`.
- `ClinicalNlpService`:
  - Build prompt extraction.
  - Gọi Ollama và parse JSON.
- `SoapService`:
  - Build prompt SOAP từ transcript + extraction.
  - Gọi Ollama và parse JSON.
- `SessionStore`:
  - Read/write an toàn bằng lock thread.
  - Cập nhật status theo từng giai đoạn.
- `OllamaClient`:
  - Wrapper gọi `ollama.generate`.
  - Có fallback tách JSON từ text nếu model trả thêm wrapper markdown.

---

## 4. Luồng hoạt động chi tiết (end-to-end)

### Bước 1 - Start session
Endpoint: `POST /sessions/start`

Input:
```json
{
  "patient_id": "P001",
  "encounter_id": "E001",
  "doctor_id": "D001"
}
```

Hệ thống:
- Tạo `session_id` (UUID).
- Lưu session vào store với `status = "recording"`.

Output:
- Trả object `session`.

### Bước 2 - Stop session (optional theo UX)
Endpoint: `POST /sessions/{id}/stop`

Hệ thống:
- Set `ended_at`.
- Nếu đang `recording` thì chuyển `status -> created`.

### Bước 3 - Upload audio
Endpoint: `POST /sessions/{id}/audio` (multipart form-data)

Validation:
- Bắt buộc có filename.
- Kiểm tra `content_type` audio hoặc ext hợp lệ (`.wav`, `.mp3`, `.m4a`, `.webm`, `.ogg`, `.flac`).
- Không chấp nhận file rỗng.

Hệ thống:
- Ghi file vào `UPLOAD_DIR` với tên `<session_id>.<ext>`.
- Cập nhật session: `audio_path`, `status = uploaded`.

Output:
- Session cập nhật + kích thước file + tên file.

### Bước 4 - Transcribe
Endpoint: `POST /sessions/{id}/transcribe`

Hệ thống:
- Lấy `audio_path` từ session.
- Gọi `SttService.transcribe_file()` trong thread (không block event loop).
- Whisper config:
  - `language=vi`
  - `vad_filter=True`
  - `beam_size=5`
- Chuẩn hóa text qua `normalize_text()` để giảm noise spacing.
- Tạo `TranscriptResult`:
  - `segments` (start/end ms, speaker, text, confidence)
  - `full_text` (gộp các line)
  - `language`, `created_at`, `transcript_id`
- Lưu transcript vào store, cập nhật session `status = transcribed`.

Output:
- JSON transcript đầy đủ.

### Bước 5 - Analyze (Extraction + SOAP)
Endpoint: `POST /sessions/{id}/analyze`

Điều kiện:
- Session phải có transcript.
- `full_text` không được rỗng.

Hệ thống:
1. `ClinicalNlpService.extract(transcript_text)`:
   - Prompt yêu cầu JSON schema:
     - `chief_complaint`
     - `onset`
     - `associated_symptoms[]`
     - `pertinent_negatives[]`
     - `short_summary`
     - `evidence_snippets[]`
   - Ràng buộc "không bổ sung thông tin không có trong transcript".
2. `SoapService.generate(transcript_text, extraction)`:
   - Prompt tạo JSON `subjective/objective/assessment/plan`.
   - Nếu không rõ thì ghi "Khong ro".
3. Lưu extraction + soap vào store.
4. Cập nhật session `status = processed`.

Output:
```json
{
  "extraction": {...},
  "soap_note": {...}
}
```

### Bước 6 - Review
Endpoint đọc tổng hợp: `GET /sessions/{id}/review`
- Trả full bundle:
  - `session`
  - `transcript`
  - `extraction`
  - `soap_note`
  - `review`
  - `finalization`

Endpoint ghi review: `POST /sessions/{id}/review`

Input:
```json
{
  "doctor_id": "D001",
  "extraction": {
    "chief_complaint": "...",
    "onset": "...",
    "associated_symptoms": [],
    "pertinent_negatives": [],
    "short_summary": "...",
    "evidence_snippets": []
  },
  "soap_note": {
    "subjective": "...",
    "objective": "...",
    "assessment": "...",
    "plan": "...",
    "generated_at": "2026-03-17T00:00:00Z",
    "model_name": "qwen2.5:7b"
  },
  "comment": "Da sua ten thuoc"
}
```

Hệ thống:
- Lưu review, cập nhật session `status = reviewed`.

### Bước 7 - Finalize
Endpoint: `POST /sessions/{id}/finalize`

Input:
```json
{
  "doctor_id": "D001",
  "note": "Dong y ban review cuoi"
}
```

Hệ thống:
- Kiểm tra session đã có review chưa.
- Nếu chưa review -> lỗi `400`.
- Nếu hợp lệ -> tạo finalization record, `status = finalized`.

---

## 5. Session state machine

Trạng thái được cập nhật trong `SessionStore`:

```text
recording
  -> created        (sau /stop)
  -> uploaded       (sau /audio)
  -> transcribed    (sau /transcribe)
  -> processed      (sau /analyze)
  -> reviewed       (sau /review)
  -> finalized      (sau /finalize)
```

Lưu ý:
- `processed` được set khi lưu extraction/soap.
- `finalize` bắt buộc sau `review`.

---

## 6. Đầu vào - đầu ra và data flow

### Đầu vào chính
- Multipart audio file.
- Metadata session (`patient_id`, `encounter_id`, `doctor_id`).
- Các payload review/finalize.

### Đầu ra chính
- Transcript segment-level + full text.
- Clinical extraction JSON.
- SOAP draft JSON.
- Review bundle để bác sĩ duyệt.
- Finalization record.

### Data flow theo storage
```text
Upload audio        -> data/uploads/<session_id>.<ext>
Session metadata    -> sessions.json:sessions
Transcript          -> sessions.json:transcripts
Extraction          -> sessions.json:extractions
SOAP                -> sessions.json:soap_notes
Review              -> sessions.json:reviews
Finalize            -> sessions.json:finalizations
```

---

## 7. API reference

### Health
- `GET /health`
- Trả:
  - `status`
  - `ollama_ok`
  - `whisper_ready`
  - `whisper_device`

### Session lifecycle
- `POST /sessions/start`
- `POST /sessions/{id}/stop`
- `GET /sessions/{id}`

### Audio / STT
- `POST /sessions/{id}/audio`
- `POST /sessions/{id}/transcribe`
- `GET /sessions/{id}/transcript`

### NLP / SOAP
- `POST /sessions/{id}/analyze`
- `GET /sessions/{id}/clinical-note`

### Review / Finalization
- `GET /sessions/{id}/review`
- `POST /sessions/{id}/review`
- `POST /sessions/{id}/finalize`

### Debug helper
- `POST /debug/upload-temp`
  - Dùng để debug upload nhanh trong giai đoạn prototype.

---

## 8. Cấu hình env vars

Từ `src/config.py` và `.env.example`:

- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:7b`)
- `WHISPER_MODEL_SIZE` (default: `medium`)
- `WHISPER_DEVICE` (default: `cuda`)
- `WHISPER_COMPUTE_TYPE` (default: `float16`)
- `WHISPER_LANGUAGE` (default: `vi`)
- `UPLOAD_DIR` (default: `data/uploads`)
- `STORE_FILE` (default: `data/store/sessions.json`)
- `MAX_INPUT_CHARS` (default: `12000`)

Ý nghĩa quan trọng:
- `MAX_INPUT_CHARS` cắt transcript khi đưa vào prompt để tránh prompt quá dài.
- `WHISPER_DEVICE=cuda` sẽ thử dùng GPU trước, nếu thất bại sẽ fallback CPU/int8.

---

## 9. Cài đặt và chạy

### 9.1 Yêu cầu hệ thống
- Python 3.10+
- Ollama đang chạy và đã pull model `qwen2.5:7b`
- `ffmpeg` có trong PATH
- (Khuyến nghị) GPU CUDA cho faster-whisper

### 9.2 Setup
```bash
cd stt-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 9.3 Chạy server
Lựa chọn 1:
```bash
uvicorn main:app --reload --port 8010
```

Lựa chọn 2 (ưu tiên khi dùng CUDA lib trong venv):
```bash
bash run.sh
```

Mặc định API: `http://localhost:8010`

### 9.4 Test nhanh
- Mở `test.html`
- Đặt Base URL: `http://localhost:8010`
- Chạy theo thứ tự:
  1. Start Session
  2. Stop Session
  3. Upload Audio
  4. Transcribe
  5. Analyze
  6. Save Review
  7. Finalize

---

## 10. Cấu trúc thư mục

```text
stt-bot/
  main.py                              # FastAPI endpoints
  run.sh                               # Chạy uvicorn với LD_LIBRARY_PATH cho CUDA libs trong venv
  src/
    config.py                          # Đọc env vars + tạo upload/store dirs
    models.py                          # Pydantic schemas cho request/response
    services/
      stt_service.py                   # faster-whisper transcribe
      clinical_nlp_service.py          # Trích xuất thông tin lâm sàng JSON
      soap_service.py                  # Tạo SOAP draft JSON
      ollama_client.py                 # Wrapper gọi Ollama + parse JSON
      session_store.py                 # Lưu/truy vấn dữ liệu sessions.json
      transcript_normalizer.py         # Chuẩn hóa text transcript
  data/
    uploads/                           # Audio đã upload
    store/sessions.json                # Persistent JSON store
  test.html                            # Giao diện test API nhanh
```

---

## 11. Giới hạn hiện tại và hướng phát triển

Giới hạn:
- Chưa có speaker diarization thực sự (`speaker` hiện là `UNKNOWN`).
- Store đang là 1 file JSON, chưa tối ưu cho concurrent cao.
- Chưa có cơ chế auth/audit log cho mọi thao tác review/finalize.

Hướng phát triển:
- Thêm speaker diarization và timestamp-to-note linking.
- Chuyển store sang Postgres/Redis + object storage cho audio.
- Thêm validation y khoa và bộ test chất lượng output SOAP.
