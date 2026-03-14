from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    encounter_id: str = Field(..., min_length=1)
    doctor_id: str = Field(..., min_length=1)


class RecordingSession(BaseModel):
    session_id: str
    patient_id: str
    encounter_id: str
    doctor_id: str
    started_at: datetime
    ended_at: datetime | None = None
    status: str
    audio_path: str | None = None
    transcript_id: str | None = None
    created_at: datetime
    updated_at: datetime


class TranscriptSegment(BaseModel):
    start_ms: int
    end_ms: int
    speaker: str = "UNKNOWN"
    text: str
    confidence: float | None = None


class TranscriptResult(BaseModel):
    transcript_id: str
    session_id: str
    language: str
    full_text: str
    segments: list[TranscriptSegment]
    created_at: datetime


class ClinicalExtraction(BaseModel):
    chief_complaint: str | None = None
    onset: str | None = None
    associated_symptoms: list[str] = Field(default_factory=list)
    pertinent_negatives: list[str] = Field(default_factory=list)
    short_summary: str | None = None
    evidence_snippets: list[str] = Field(default_factory=list)


class SoapDraft(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    generated_at: datetime
    model_name: str


class ReviewPayload(BaseModel):
    doctor_id: str = Field(..., min_length=1)
    extraction: ClinicalExtraction
    soap_note: SoapDraft
    comment: str | None = None


class FinalizePayload(BaseModel):
    doctor_id: str = Field(..., min_length=1)
    note: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama_ok: bool
    whisper_ready: bool
    whisper_device: str | None = None


class ApiMessage(BaseModel):
    message: str
    details: dict[str, Any] | None = None
