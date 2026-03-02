from __future__ import annotations

import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import io

from configs import GENERATION_CONFIG, MODEL_NAME, SAFETY_SETTINGS, SYSTEM_PROMPT

load_dotenv(override=False)

genai.configure(api_key=os.getenv("YOUR_GOOGLE_GEMINI_API", ""))

app = FastAPI(title="xray-analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    analysis: str
    model: str
    finish_reason: str | None = None
    usage: dict | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (png/jpg/jpeg).")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG,
        system_instruction=SYSTEM_PROMPT,
    )

    try:
        response = model.generate_content(["Analyze this image.", image])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    candidate = response.candidates[0] if response.candidates else None
    analysis = response.text or ""
    finish_reason = str(candidate.finish_reason) if candidate else None

    usage = None
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        m = response.usage_metadata
        usage = {
            "prompt_token_count": getattr(m, "prompt_token_count", None),
            "candidates_token_count": getattr(m, "candidates_token_count", None),
            "total_token_count": getattr(m, "total_token_count", None),
        }

    return AnalyzeResponse(
        analysis=analysis,
        model=MODEL_NAME,
        finish_reason=finish_reason,
        usage=usage,
    )
