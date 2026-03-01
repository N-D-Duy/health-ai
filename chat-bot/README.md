# chat-bot-yte

Chatbot tư vấn sức khỏe tiếng Việt, chạy hoàn toàn **offline** (local LLM + local vector DB). Kiến trúc RAG (Retrieval-Augmented Generation) kết hợp với bộ lọc chủ đề bằng LLM, đảm bảo bot chỉ trả lời đúng phạm vi y tế.

---

## Kiến trúc tổng quan

```
                        ┌──────────────────────────────────────────────┐
                        │                  FastAPI Server               │
                        │               (main.py · port 8000)           │
                        └──────────────────┬───────────────────────────┘
                                           │ POST /chat
                                           ▼
                        ┌──────────────────────────────────────────────┐
                        │               chat_once()  (chat.py)          │
                        │                                              │
                        │  [1] LLM Classifier                          │
                        │      Câu hỏi có phải y tế không?             │
                        │      → KHÔNG  ──► Từ chối cố định            │
                        │      → CÓ     ──►  [2]                       │
                        │                                              │
                        │  [2] RAG – ChromaDB + BAAI/bge-m3            │
                        │      Tìm top-K đoạn văn liên quan             │
                        │      cosine distance ≤ 0.75?                 │
                        │      → CÓ  context  ──► đưa vào system prompt│
                        │      → KHÔNG context ──► dùng kiến thức chung│
                        │                                              │
                        │  [3] LLM Chat – Ollama (Qwen2.5 7B)          │
                        │      Sinh câu trả lời + history session      │
                        └──────────────────────────────────────────────┘
```

---

## Thành phần

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| **API Server** | FastAPI + Uvicorn | Nhận request HTTP, quản lý session lịch sử hội thoại |
| **LLM (Inference)** | Ollama · Qwen2.5 7B | Classifier chủ đề + sinh câu trả lời |
| **Embedding Model** | `BAAI/bge-m3` (HuggingFace) | Encode câu hỏi và tài liệu thành vector |
| **Vector Database** | ChromaDB (persistent, local) | Lưu và tìm kiếm đoạn văn y tế bằng cosine similarity |
| **Dữ liệu** | MedQuAD (dịch tiếng Việt) | Bộ câu hỏi–đáp y tế được index sẵn |

---

## Luồng hoạt động chi tiết

### 1. Khởi động server

```bash
./scripts/run_server.sh
```

Khi server khởi động, `lifespan` trong `main.py` chạy **warm-up**:
- Gọi `query_chromadb("sức khỏe", 1)` để load embedding model và mở kết nối ChromaDB vào bộ nhớ
- Mục đích: request đầu tiên không bị chậm do lazy-load model weights

### 2. Nhận request

```
POST /chat
{ "message": "đau đầu kéo dài có nguy hiểm không?", "session_id": "user_123" }
```

- `session_id` tùy chọn — nếu không truyền sẽ dùng session mặc định `"default"`
- Server lấy lịch sử hội thoại của session đó từ bộ nhớ (tối đa `MAX_HISTORY_MESSAGES = 12` tin nhắn gần nhất)

### 3. Bước 1 — LLM Classifier (off-topic guard)

Trước khi làm bất cứ điều gì, `chat.py` gọi `_is_medical_query()`:

```
Prompt gửi LLM:
  "Nhiệm vụ: Xác định xem câu hỏi sau có thuộc lĩnh vực y tế, sức khỏe,
   bệnh lý, triệu chứng, thuốc, dinh dưỡng, vệ sinh cá nhân... không.
   Chỉ trả lời đúng một từ: CÓ hoặc KHÔNG.
   Câu hỏi: {query}"
```

- `temperature=0`, `num_predict=10` → deterministic, cực nhanh
- Trả về `"CÓ"` → đi tiếp; trả về `"KHÔNG"` → trả ngay câu từ chối cố định, **không gọi RAG hay LLM nữa**

> Đây là cơ chế chính để lọc off-topic. Không dùng keyword cứng nên không bị bỏ sót.

### 4. Bước 2 — RAG (Retrieval-Augmented Generation)

Nếu câu hỏi là y tế, `retrieval.py` thực hiện semantic search trong ChromaDB:

1. **Encode câu hỏi** bằng `BAAI/bge-m3` thành vector 1024 chiều
2. **Tìm top-K = 5** đoạn văn gần nhất theo **cosine distance** trong collection `medquad_vi`
3. **Kiểm tra ngưỡng**: nếu `best_distance ≤ 0.75` → có context liên quan; ngược lại → không có context

> Cosine distance: càng nhỏ càng giống. `0.75` là ngưỡng "đủ gần để tin dùng".

### 5. Bước 3 — Sinh câu trả lời (LLM Chat)

`chat.py` xây dựng `messages` cho Ollama:

```
[system]   SYSTEM_GUARDRAIL_VI
           + (nếu có context) các đoạn RAG trích dẫn
           + (nếu không có context) hướng dẫn dùng kiến thức chung

[user]     [lịch sử hội thoại gần nhất]
[assistant]

[user]     câu hỏi hiện tại
```

- Gọi `client.chat()` tới Ollama (`qwen2.5:7b`)
- Mọi câu trả lời đều được `_ensure_disclaimer()` đảm bảo có dòng: *"Thông tin chỉ mang tính tham khảo; bạn nên gặp bác sĩ để được chẩn đoán và điều trị."*

### 6. Trả kết quả

```json
{
  "response": "Đau đầu kéo dài có thể do nhiều nguyên nhân...\n\nThông tin chỉ mang tính tham khảo...",
  "session_id": "user_123"
}
```

Lịch sử hội thoại được lưu lại trong RAM cho session đó, phục vụ các lượt hỏi tiếp theo.

---

## Cài đặt & Chạy

### Yêu cầu

- Python 3.10+
- [Ollama](https://ollama.com) đang chạy locally
- (Tùy chọn) HuggingFace token để tải embedding model nhanh hơn

### 1. Cài dependencies

```bash
cd chat-bot
pip install -r requirements.txt
```

### 2. Tạo Ollama model

```bash
ollama create chat-bot-yte -f Modelfile
```

Model dựa trên `qwen2.5:7b` với `temperature=0.5`, `num_ctx=4096`.

### 3. (Lần đầu) Build dữ liệu & Index vào ChromaDB

Nếu chưa có `chroma_db/` hoặc muốn rebuild:

```bash
# Bước 1: Tải và dịch dataset MedQuAD sang tiếng Việt
python scripts/build_medquad_vi.py

# Bước 2: Index CSV vào ChromaDB
python scripts/index_to_chromadb.py
```

> `chroma_db/` đã có sẵn trong repo — bỏ qua bước này nếu không cần rebuild.

### 4. Chạy server

```bash
# Tùy chọn: set HF token để tránh rate limit khi tải model
export HF_TOKEN=hf_xxx

./scripts/run_server.sh
# → http://localhost:8000
```

---

## API

### `GET /health`

Kiểm tra trạng thái server và kết nối Ollama.

```json
{ "status": "ok", "ollama_ok": true }
```

### `POST /chat`

| Field | Type | Mô tả |
|---|---|---|
| `message` | string (required) | Câu hỏi của người dùng |
| `session_id` | string (optional) | ID phiên hội thoại, mặc định `"default"` |

---

## Cấu hình (Environment Variables)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `OLLAMA_MODEL` | `chat-bot-yte` | Tên model Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Địa chỉ Ollama |
| `EMBED_MODEL_NAME` | `BAAI/bge-m3` | Model embedding HuggingFace |
| `CHROMA_PATH` | `chroma_db` | Thư mục lưu ChromaDB |
| `CHROMA_COLLECTION` | `medquad_vi` | Tên collection |
| `TOP_K` | `5` | Số đoạn văn RAG tìm mỗi lượt |
| `RAG_DISTANCE_OFF_TOPIC_THRESHOLD` | `0.75` | Ngưỡng cosine distance để coi context là liên quan |
| `MAX_HISTORY_MESSAGES` | `12` | Số tin nhắn lịch sử giữ per session |
| `HF_TOKEN` | _(trống)_ | HuggingFace API token |

---

## Cấu trúc thư mục

```
chat-bot/
├── main.py                  # FastAPI app, endpoints, session management
├── Modelfile                # Định nghĩa Ollama model (base: qwen2.5:7b)
├── requirements.txt
├── src/
│   ├── chat.py              # Logic chính: classifier → RAG → LLM
│   ├── retrieval.py         # ChromaDB query, embedding
│   └── config.py            # Tất cả hằng số & env vars
├── scripts/
│   ├── build_medquad_vi.py  # Tải và dịch dataset MedQuAD
│   ├── index_to_chromadb.py # Index CSV vào ChromaDB
│   ├── run_server.sh        # Script chạy server
│   └── test_chat.py         # Test nhanh API
└── data/
    └── medquad_vi.csv       # Dataset y tế tiếng Việt
```
