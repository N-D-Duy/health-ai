from __future__ import annotations

import base64
import io
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from PIL import Image
from pydantic import BaseModel

from configs import MAX_TOKENS, MODEL_NAME, SYSTEM_PROMPT, TEMPERATURE

load_dotenv(override=False)

_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", MODEL_NAME)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

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
    return {"status": "ok", "model": _deployment}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (png/jpg/jpeg).")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    # Resize về tối đa 1024x1024 trước khi encode
    image.thumbnail((1024, 1024), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format=image.format or "JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = file.content_type or "image/jpeg"

    try:
        response = client.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this x-ray image."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
            max_completion_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI error: {e}")

    choice = response.choices[0]
    analysis = choice.message.content or ""
    finish_reason = str(choice.finish_reason) if choice.finish_reason else None

    usage = None
    if response.usage:
        usage = {
            "prompt_token_count": response.usage.prompt_tokens,
            "candidates_token_count": response.usage.completion_tokens,
            "total_token_count": response.usage.total_tokens,
        }

    return AnalyzeResponse(
        analysis=analysis,
        model=_deployment,
        finish_reason=finish_reason,
        usage=usage,
    )
