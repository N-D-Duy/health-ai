# chat-bot-yte

Chatbot tư vấn sức khỏe tiếng Việt chạy local theo kiến trúc RAG (Retrieval-Augmented Generation). Mục tiêu là:
- Chỉ trả lời trong phạm vi y tế/sức khỏe.
- Tận dụng tri thức từ bộ dữ liệu MedQuAD đã index vào ChromaDB.
- Vẫn hoạt động được offline ở tầng model suy luận (Ollama local).

Tài liệu này viết theo hướng onboarding: người chưa biết gì về project vẫn có thể hiểu được thành phần nào đang chạy, vì sao dùng nó, dữ liệu đi như thế nào và API trả gì.

---

## 1. Project này giải bài toán gì?

Trong hệ thống tư vấn y tế, nếu cho LLM trả lời trực tiếp thì dễ bị:
- Hallucination (tự suy diễn sai).
- Lệch chủ đề (người dùng hỏi ngoài y tế).
- Câu trả lời không ổn định khi câu hỏi mơ hồ.

`chat-bot` giải bài toán này bằng 3 lớp bảo vệ:
- Lớp 1: LLM classifier xác định câu hỏi có thuộc y tế không.
- Lớp 2: Retrieval + rerank để đưa context liên quan vào prompt.
- Lớp 3: Guardrail + disclaimer bắt buộc trong câu trả lời.

---

## 2. Kiến trúc tổng quan

```text
Client
  -> POST /chat
FastAPI (main.py)
  -> chat_once() (src/chat.py)
     1) Medical classifier (Ollama)
     2) Query expansion (Ollama)
     3) Hybrid retrieval (Chroma semantic + BM25)
     4) Reciprocal Rank Fusion (RRF)
     5) Cross-encoder rerank
     6) LLM answer generation (Ollama)
  -> Return {response, session_id}
```

Dữ liệu tri thức:
```text
MedQuAD (EN)
  -> scripts/build_medquad_vi.py (lọc + dịch)
  -> data/medquad_vi.csv
  -> scripts/index_to_chromadb.py (chunk + embedding)
  -> chroma_db/ (persistent local vector store)
```

---

## 3. Thành phần được sử dụng, vai trò và lý do lựa chọn

### 3.1 API layer
- Công nghệ: `FastAPI` + `uvicorn`
- Vai trò:
  - Cung cấp endpoint `GET /health`, `POST /chat`.
  - Quản lý session history trong RAM theo `session_id`.
  - Warm-up các thành phần nặng khi startup.
- Lý do dùng:
  - Nhẹ, dễ mở rộng REST API.
  - Tích hợp validation request/response rõ ràng.

### 3.2 LLM suy luận
- Công nghệ: `Ollama` + model `chat-bot-yte` (base `qwen2.5:7b` trong `Modelfile`).
- Vai trò:
  - Phân loại off-topic (medical classifier).
  - Query expansion.
  - Sinh câu trả lời cuối.
- Lý do dùng:
  - Chạy local, không phụ thuộc cloud API.
  - Qwen 7B cho chat tiếng Việt khá ổn trong prototype.

### 3.3 Vector retrieval
- Công nghệ: `ChromaDB` + embedding `BAAI/bge-m3`.
- Vai trò:
  - Lưu các chunk hỏi-đáp y tế.
  - Semantic search theo cosine distance.
- Lý do dùng:
  - ChromaDB dễ setup local, có persistent storage.
  - bge-m3 là model embedding đa ngôn ngữ tốt cho retrieval đa ngôn ngữ.

### 3.4 Keyword retrieval
- Công nghệ: `rank-bm25`.
- Vai trò:
  - Bắt các trường hợp cần khớp từ khóa chính xác (tên thuốc, tên bệnh, triệu chứng cụ thể).
- Lý do dùng:
  - Semantic search mạnh về ý nghĩa, BM25 mạnh về exact lexical match.
  - Kết hợp 2 kênh giúp giảm miss retrieval.

### 3.5 Rank fusion và rerank
- Công nghệ:
  - Reciprocal Rank Fusion (RRF) trong `src/retrieval.py`.
  - CrossEncoder `BAAI/bge-reranker-base`.
- Vai trò:
  - RRF gộp nhiều danh sách kết quả (từ query gốc + query mở rộng, semantic + BM25).
  - Reranker score lại cặp `(query, chunk)` để chọn top chunk chất lượng nhất.
- Lý do dùng:
  - Kết hợp retrieval candidates rộng và precision cao ở bước cuối.

### 3.6 Dữ liệu nguồn
- Nguồn: MedQuAD (script `build_medquad_vi.py` đang lọc `qtype = symptoms`).
- Vai trò:
  - Tạo kho tri thức ban đầu cho domain y tế.
- Lý do dùng:
  - Dữ liệu Q&A y tế có cấu trúc rõ ràng, phù hợp cho RAG prototype.

---

## 4. Luồng hoạt động chi tiết (runtime)

### 4.1 Startup
Khi chạy server (`scripts/run_server.sh`):
- Warm-up semantic retrieval: gọi `query_chromadb("suc khoe", 1)` để load embedding + kết nối collection.
- Nếu bật `HYBRID_SEARCH_ENABLED`: build BM25 index từ toàn bộ corpus trong Chroma 1 lần.
- Nếu bật `RERANK_ENABLED`: preload reranker weights.

Ý nghĩa: request đầu tiên không bị trễ do lazy load model.

### 4.2 Nhận request chat
Endpoint:
```http
POST /chat
Content-Type: application/json
```

Body:
```json
{
  "message": "Dau dau keo dai co nguy hiem khong?",
  "session_id": "user_123"
}
```

Server:
- Chuẩn hóa `session_id` (mặc định `default` nếu không có).
- Lấy lịch sử hội thoại theo session trong RAM.

### 4.3 Bước 1 - Kiểm tra chủ đề y tế
Hàm: `_is_medical_query()` trong `src/chat.py`.
- Gửi prompt classifier cho Ollama.
- Kỳ vọng model trả `CO`/`KHONG`.
- Nếu `KHONG`: trả ngay `REFUSAL_OFF_TOPIC_VI`, không retrieve, không generate thêm.

Tại sao cần bước này?
- Giữ phạm vi hệ thống.
- Giảm nguy cơ model "trò chuyện linh tinh" ngoài y tế.

### 4.4 Bước 2 - Query expansion
Hàm: `_expand_query()`.
- Từ 1 câu hỏi, sinh thêm `QUERY_EXPANSION_N` cách diễn đạt (mặc định 2).
- Tổng thành 3 truy vấn: `[query_goc, query_1, query_2]`.

Tại sao cần?
- Người dùng viết câu hỏi ngắn/không chuẩn văn phong.
- Expansion tăng khả năng bắt đúng document liên quan.

### 4.5 Bước 3 - Hybrid retrieval
Với mỗi query:
- Semantic search: `query_chromadb()`.
- Keyword search: `bm25_search()` (nếu hybrid đang bật).

Sau đó:
- Dùng `reciprocal_rank_fusion()` để gộp kết quả.
- RRF ưu tiên chunk xuất hiện ở vị trí cao trong nhiều danh sách.

### 4.6 Bước 4 - Reranking
Hàm: `rerank_chunks()`.
- Input: pool candidates.
- CrossEncoder score lại theo cặp `(user_message, chunk_text)`.
- Lấy `RERANK_TOP_N` chunk (mặc định 3) đưa vào prompt.

### 4.7 Bước 5 - Build prompt và generate
`chat_once()` xây `messages` theo thứ tự:
- `system`: guardrail + context chunks (nếu có).
- `history`: tối đa `MAX_HISTORY_MESSAGES` gần nhất.
- `user`: câu hỏi hiện tại.

Sau đó gọi:
- `client.chat(model=OLLAMA_MODEL, messages=...)`.

Kết quả được:
- Ép bổ sung disclaimer nếu model quên (`_ensure_disclaimer()`).
- Lưu vào history session (`user`, `assistant`).
- Trả lại API.

---

## 5. Đầu vào - đầu ra và data flow

### 5.1 Đầu vào runtime
- Từ người dùng: `message`, `session_id`.
- Từ hệ thống: collection ChromaDB, cấu hình env, model Ollama.

### 5.2 Đầu ra runtime
`POST /chat` trả:
```json
{
  "response": "...noi dung tu van...\n\nThong tin chi mang tinh tham khao...",
  "session_id": "user_123"
}
```

### 5.3 Dữ liệu đi như thế nào?
```text
User question
  -> classifier
  -> (optional) expansion
  -> retrieval (semantic + bm25)
  -> fusion + rerank
  -> LLM answer with context + history
  -> response + save session in RAM
```

Lưu ý quan trọng:
- Lịch sử session hiện tại lưu trong RAM (`_sessions`), restart server sẽ mất.
- Tri thức RAG lưu bền vững trong `chroma_db/`.

---

## 6. Pipeline dữ liệu xây kho tri thức (offline indexing)

### Bước A - Tạo CSV tiếng Việt
Script: `scripts/build_medquad_vi.py`
- Load dataset `keivalya/MedQuad-MedicalQnADataset`.
- Lọc `qtype` theo danh sách cho phép (hiện tại là `symptoms`).
- Có thể dịch bằng provider (`azure`, `libretranslate`) hoặc giữ nguyên (`none`).
- Output: `data/medquad_vi.csv`.

### Bước B - Index vào ChromaDB
Script: `scripts/index_to_chromadb.py`
- Đọc CSV.
- Tạo document dạng `Hoi: ...\nDap: ...`.
- Chunk text theo `max_chars/overlap`.
- Embed bằng `BAAI/bge-m3`.
- Add vào collection Chroma persistent.

---

## 7. API reference

### 7.1 `GET /health`
Mục đích: kiểm tra server và Ollama.

Response mẫu:
```json
{
  "status": "ok",
  "ollama_ok": true
}
```

### 7.2 `POST /chat`
Request:
- `message` (`string`, required): câu hỏi.
- `session_id` (`string`, optional): id phiên chat.

Response:
- `response`: câu trả lời đã qua guardrail.
- `session_id`: id phiên được sử dụng.

---

## 8. Cấu hình environment variables

### Core
- `OLLAMA_MODEL` (default: `chat-bot-yte`): tên model Ollama được gọi.
- `OLLAMA_HOST` (default: `http://localhost:11434`): endpoint Ollama.

### Retrieval
- `CHROMA_PATH` (default: `chroma_db`): thư mục Chroma persistent.
- `CHROMA_COLLECTION` (default: `medquad_vi`): tên collection.
- `EMBED_MODEL_NAME` (default: `BAAI/bge-m3`): embedding model.
- `TOP_K` (default: `5`): số chunk trả về khi không rerank.
- `RAG_DISTANCE_OFF_TOPIC_THRESHOLD` (default: `0.75`): ngưỡng fallback khi không có rerank.

### Rerank / Hybrid / Expansion
- `RERANK_ENABLED` (`true|false`, default `true`).
- `RERANKER_MODEL_NAME` (default `BAAI/bge-reranker-base`).
- `RERANK_TOP_N` (default `3`).
- `RETRIEVE_CANDIDATES` (default `10`).
- `HYBRID_SEARCH_ENABLED` (`true|false`, default `true`).
- `QUERY_EXPANSION_ENABLED` (`true|false`, default `true`).
- `QUERY_EXPANSION_N` (default `2`).

### Conversation behavior
- `MAX_HISTORY_MESSAGES` (default `12`): số message history đưa vào prompt.
- `REFUSAL_OFF_TOPIC_VI`: câu từ chối cố định khi off-topic.
- `SYSTEM_GUARDRAIL_VI`: system instruction tổng.

### Optional
- `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`: token Hugging Face để tải model ổn định hơn.

---

## 9. Cài đặt và chạy

### 9.1 Yêu cầu
- Python 3.10+
- Ollama đã cài và đang chạy.
- Model được tạo từ `Modelfile`.

### 9.2 Cài dependencies
```bash
cd chat-bot
pip install -r requirements.txt
```

### 9.3 Tạo model Ollama cho app
```bash
ollama create chat-bot-yte -f Modelfile
```

### 9.4 (Nếu cần) rebuild tri thức
```bash
python scripts/build_medquad_vi.py --limit 800 --translate-provider none
python scripts/index_to_chromadb.py --in data/medquad_vi.csv
```

### 9.5 Chạy API
```bash
./scripts/run_server.sh
```
Mặc định: `http://localhost:8000`

---

## 10. Cấu trúc thư mục và vai trò

```text
chat-bot/
  main.py                    # FastAPI app, session memory RAM, endpoints
  Modelfile                  # Định nghĩa model chat-bot-yte (base qwen2.5:7b)
  src/
    config.py                # Toàn bộ env config và guardrail strings
    chat.py                  # Pipeline chat_once: classifier -> retrieve -> answer
    retrieval.py             # Chroma, BM25, RRF, reranker
  scripts/
    build_medquad_vi.py      # Tạo data CSV từ MedQuAD
    index_to_chromadb.py     # Chunk + embedding + index vào Chroma
    run_server.sh            # Chạy uvicorn
  data/medquad_vi.csv        # Data đầu vào cho indexing
  chroma_db/                 # Vector DB local persistent
```

---

## 11. Giới hạn hiện tại và hướng mở rộng

Giới hạn:
- Session history đang lưu RAM, chưa có Redis/DB cho production.
- Off-topic classifier phụ thuộc LLM (có thể cần eval thêm với bộ test).
- Nguồn tri thức hiện tại chủ yếu từ MedQuAD subset.

Hướng mở rộng:
- Thêm bộ test regression cho classifier và retrieval.
- Bổ sung citation (id chunk/nguồn) trong response.
- Lưu session persistent và telemetry cho monitoring.
