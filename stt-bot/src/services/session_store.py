from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import STORE_FILE
from src.models import StartSessionRequest


class SessionStore:
    def __init__(self, file_path: Path = STORE_FILE):
        self.file_path = file_path
        self._lock = threading.Lock()
        if not self.file_path.exists():
            self._write(
                {
                    "sessions": {},
                    "transcripts": {},
                    "extractions": {},
                    "soap_notes": {},
                    "reviews": {},
                    "finalizations": {},
                }
            )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict:
        with self.file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_session(self, req: StartSessionRequest) -> dict:
        with self._lock:
            data = self._read()
            now = self._now_iso()
            session_id = str(uuid4())
            session = {
                "session_id": session_id,
                "patient_id": req.patient_id,
                "encounter_id": req.encounter_id,
                "doctor_id": req.doctor_id,
                "started_at": now,
                "ended_at": None,
                "status": "recording",
                "audio_path": None,
                "transcript_id": None,
                "created_at": now,
                "updated_at": now,
            }
            data["sessions"][session_id] = session
            self._write(data)
            return session

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            data = self._read()
            return data["sessions"].get(session_id)

    def stop_session(self, session_id: str) -> dict:
        with self._lock:
            data = self._read()
            session = data["sessions"].get(session_id)
            if not session:
                raise KeyError("Session not found")
            now = self._now_iso()
            session["ended_at"] = now
            session["updated_at"] = now
            if session["status"] == "recording":
                session["status"] = "created"
            self._write(data)
            return session

    def attach_audio(self, session_id: str, audio_path: str) -> dict:
        with self._lock:
            data = self._read()
            session = data["sessions"].get(session_id)
            if not session:
                raise KeyError("Session not found")
            session["audio_path"] = audio_path
            session["status"] = "uploaded"
            session["updated_at"] = self._now_iso()
            self._write(data)
            return session

    def save_transcript(self, session_id: str, transcript: dict) -> dict:
        with self._lock:
            data = self._read()
            session = data["sessions"].get(session_id)
            if not session:
                raise KeyError("Session not found")
            transcript_id = transcript["transcript_id"]
            data["transcripts"][transcript_id] = transcript
            session["transcript_id"] = transcript_id
            session["status"] = "transcribed"
            session["updated_at"] = self._now_iso()
            self._write(data)
            return transcript

    def get_transcript(self, session_id: str) -> dict | None:
        with self._lock:
            data = self._read()
            session = data["sessions"].get(session_id)
            if not session or not session.get("transcript_id"):
                return None
            return data["transcripts"].get(session["transcript_id"])

    def save_extraction(self, session_id: str, extraction: dict, model_name: str) -> dict:
        with self._lock:
            data = self._read()
            if session_id not in data["sessions"]:
                raise KeyError("Session not found")
            payload = {
                "session_id": session_id,
                "model_name": model_name,
                "generated_at": self._now_iso(),
                "data": extraction,
            }
            data["extractions"][session_id] = payload
            data["sessions"][session_id]["status"] = "processed"
            data["sessions"][session_id]["updated_at"] = self._now_iso()
            self._write(data)
            return payload

    def get_extraction(self, session_id: str) -> dict | None:
        with self._lock:
            data = self._read()
            return data["extractions"].get(session_id)

    def save_soap(self, session_id: str, soap: dict, model_name: str) -> dict:
        with self._lock:
            data = self._read()
            if session_id not in data["sessions"]:
                raise KeyError("Session not found")
            payload = {
                "session_id": session_id,
                "model_name": model_name,
                "generated_at": self._now_iso(),
                "data": soap,
            }
            data["soap_notes"][session_id] = payload
            data["sessions"][session_id]["status"] = "processed"
            data["sessions"][session_id]["updated_at"] = self._now_iso()
            self._write(data)
            return payload

    def get_soap(self, session_id: str) -> dict | None:
        with self._lock:
            data = self._read()
            return data["soap_notes"].get(session_id)

    def save_review(self, session_id: str, review: dict) -> dict:
        with self._lock:
            data = self._read()
            if session_id not in data["sessions"]:
                raise KeyError("Session not found")
            payload = {
                "session_id": session_id,
                "reviewed_at": self._now_iso(),
                "data": review,
            }
            data["reviews"][session_id] = payload
            data["sessions"][session_id]["status"] = "reviewed"
            data["sessions"][session_id]["updated_at"] = self._now_iso()
            self._write(data)
            return payload

    def get_review(self, session_id: str) -> dict | None:
        with self._lock:
            data = self._read()
            return data["reviews"].get(session_id)

    def finalize(self, session_id: str, payload: dict) -> dict:
        with self._lock:
            data = self._read()
            if session_id not in data["sessions"]:
                raise KeyError("Session not found")
            if session_id not in data["reviews"]:
                raise ValueError("Session chưa được review")
            final = {
                "session_id": session_id,
                "finalized_at": self._now_iso(),
                "data": payload,
            }
            data["finalizations"][session_id] = final
            data["sessions"][session_id]["status"] = "finalized"
            data["sessions"][session_id]["updated_at"] = self._now_iso()
            self._write(data)
            return final

    def get_full_review_bundle(self, session_id: str) -> dict:
        with self._lock:
            data = self._read()
            session = data["sessions"].get(session_id)
            if not session:
                raise KeyError("Session not found")
            transcript = None
            if session.get("transcript_id"):
                transcript = data["transcripts"].get(session["transcript_id"])
            return {
                "session": session,
                "transcript": transcript,
                "extraction": data["extractions"].get(session_id),
                "soap_note": data["soap_notes"].get(session_id),
                "review": data["reviews"].get(session_id),
                "finalization": data["finalizations"].get(session_id),
            }
