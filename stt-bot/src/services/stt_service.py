from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)
from src.models import TranscriptResult, TranscriptSegment
from src.services.transcript_normalizer import normalize_text


class SttService:
    def __init__(self) -> None:
        self._model = None
        self._device_used: str = WHISPER_DEVICE

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError(
                "Không import được faster-whisper. Hãy cài dependencies trong requirements.txt"
            ) from e

        # Try requested device first; fall back to CPU if CUDA libs are missing.
        for device, compute in [(WHISPER_DEVICE, WHISPER_COMPUTE_TYPE), ("cpu", "int8")]:
            try:
                self._model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute)
                self._device_used = device
                break
            except Exception as e:
                if device == "cpu":
                    raise RuntimeError(f"Không khởi tạo được Whisper model: {e}") from e
                # CUDA failed – warn and retry with CPU
                import logging
                logging.getLogger(__name__).warning(
                    "CUDA init failed (%s), retrying with CPU/int8", e
                )

        return self._model

    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device_used(self) -> str:
        return self._device_used

    def transcribe_file(self, session_id: str, audio_path: Path) -> TranscriptResult:
        model = self._get_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            vad_filter=True,
            beam_size=5,
        )

        parsed_segments: list[TranscriptSegment] = []
        full_lines: list[str] = []

        for seg in segments:
            text = normalize_text(seg.text)
            if not text:
                continue
            parsed = TranscriptSegment(
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                speaker="UNKNOWN",
                text=text,
                confidence=getattr(seg, "avg_logprob", None),
            )
            parsed_segments.append(parsed)
            full_lines.append(text)

        transcript = TranscriptResult(
            transcript_id=str(uuid4()),
            session_id=session_id,
            language=getattr(info, "language", WHISPER_LANGUAGE) or WHISPER_LANGUAGE,
            full_text="\n".join(full_lines),
            segments=parsed_segments,
            created_at=datetime.now(timezone.utc),
        )
        return transcript
