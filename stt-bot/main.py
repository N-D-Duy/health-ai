from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import OLLAMA_HOST, OLLAMA_MODEL, UPLOAD_DIR
from src.models import FinalizePayload, HealthResponse, ReviewPayload, StartSessionRequest
from src.services.clinical_nlp_service import ClinicalNlpService
from src.services.ollama_client import OllamaClient
from src.services.session_store import SessionStore
from src.services.soap_service import SoapService
from src.services.stt_service import SttService

load_dotenv(override=False)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: ping Ollama để đảm bảo model đang loaded vào VRAM trước request đầu tiên.
    # stt-bot dùng chung model với chat-bot (OLLAMA_MODEL), Ollama cache trong 5m (keep_alive).
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
        await asyncio.to_thread(
            client.generate,
            model=OLLAMA_MODEL,
            prompt="ok",
            options={"num_predict": 1},
        )
        logger.info("Ollama warm-up done (model=%s)", OLLAMA_MODEL)
    except Exception as e:
        logger.warning("Ollama warm-up failed (non-fatal): %s", e)
    yield


app = FastAPI(title="stt-bot-medical-scribe", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()
stt_service = SttService()
ollama_client = OllamaClient()
clinical_nlp_service = ClinicalNlpService(ollama_client)
soap_service = SoapService(ollama_client)


def _session_or_404(session_id: str) -> dict:
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=1.5)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return HealthResponse(
        status="ok",
        ollama_ok=ollama_ok,
        whisper_ready=stt_service.is_ready(),
        whisper_device=stt_service.device_used if stt_service.is_ready() else None,
    )


@app.post("/sessions/start")
def start_session(req: StartSessionRequest) -> dict:
    session = store.create_session(req)
    return {"session": session}


@app.post("/sessions/{session_id}/stop")
def stop_session(session_id: str) -> dict:
    _session_or_404(session_id)
    try:
        session = store.stop_session(session_id)
        return {"session": session}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = _session_or_404(session_id)
    return {"session": session}


@app.post("/sessions/{session_id}/audio")
async def upload_audio(session_id: str, file: UploadFile = File(...)) -> dict:
    session = _session_or_404(session_id)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file")

    is_audio = (file.content_type or "").startswith("audio/")
    allowed_ext = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
    suffix = Path(file.filename).suffix.lower()

    if not is_audio and suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail="File không phải audio hợp lệ")

    out_name = f"{session_id}{suffix or '.wav'}"
    out_path = UPLOAD_DIR / out_name

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File audio rỗng")

    with out_path.open("wb") as f:
        f.write(data)

    updated = store.attach_audio(session_id, str(out_path))
    return {"session": updated, "file_size": len(data), "file": out_name}


@app.post("/sessions/{session_id}/transcribe")
async def transcribe_session(session_id: str) -> dict:
    session = _session_or_404(session_id)

    audio_path = session.get("audio_path")
    if not audio_path:
        raise HTTPException(status_code=400, detail="Session chưa có audio")

    path = Path(audio_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail="Không tìm thấy file audio")

    try:
        transcript = await asyncio.to_thread(stt_service.transcribe_file, session_id, path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    payload = transcript.model_dump(mode="json")
    store.save_transcript(session_id, payload)
    return {"transcript": payload}


@app.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str) -> dict:
    _session_or_404(session_id)
    transcript = store.get_transcript(session_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript chưa sẵn sàng")
    return {"transcript": transcript}


@app.post("/sessions/{session_id}/analyze")
async def analyze_session(session_id: str) -> dict:
    _session_or_404(session_id)
    transcript = store.get_transcript(session_id)
    if not transcript:
        raise HTTPException(status_code=400, detail="Cần transcribe trước")

    transcript_text = transcript.get("full_text") or ""
    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript rỗng")

    try:
        extraction = await asyncio.to_thread(clinical_nlp_service.extract, transcript_text)
        soap = await asyncio.to_thread(soap_service.generate, transcript_text, extraction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze thất bại: {e}")

    extraction_payload = extraction.model_dump(mode="json")
    soap_payload = soap.model_dump(mode="json")
    store.save_extraction(session_id, extraction_payload, ollama_client.model)
    store.save_soap(session_id, soap_payload, ollama_client.model)

    return {"extraction": extraction_payload, "soap_note": soap_payload}


@app.get("/sessions/{session_id}/clinical-note")
def get_clinical_note(session_id: str) -> dict:
    _session_or_404(session_id)
    soap = store.get_soap(session_id)
    if not soap:
        raise HTTPException(status_code=404, detail="SOAP note chưa sẵn sàng")
    return {"soap_note": soap}


@app.get("/sessions/{session_id}/review")
def get_review_bundle(session_id: str) -> dict:
    try:
        bundle = store.get_full_review_bundle(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return bundle


@app.post("/sessions/{session_id}/review")
def review_session(session_id: str, payload: ReviewPayload) -> dict:
    _session_or_404(session_id)
    data = payload.model_dump(mode="json")
    saved = store.save_review(session_id, data)
    return {"review": saved}


@app.post("/sessions/{session_id}/finalize")
def finalize_session(session_id: str, payload: FinalizePayload) -> dict:
    _session_or_404(session_id)
    data = payload.model_dump(mode="json")
    try:
        final = store.finalize(session_id, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"finalization": final}


@app.post("/debug/upload-temp")
async def debug_upload(file: UploadFile = File(...)) -> dict:
    # Convenience endpoint to quickly inspect upload issues during prototype testing.
    suffix = Path(file.filename or "sample.wav").suffix or ".wav"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    data = await file.read()
    temp.write(data)
    temp.flush()
    temp.close()
    return {"path": temp.name, "size": len(data), "content_type": file.content_type}
