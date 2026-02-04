from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid

from app.ocr_service import run_ocr
from app.rate_limit import rate_limiter
from app.security import validate_image

app = FastAPI(
    title="Secure Devanagari OCR API",
    version="0.1.0"
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- RESPONSE MODELS ----------------
class Candidate(BaseModel):
    char: str
    confidence: float


class CharacterResult(BaseModel):
    position: int
    predicted: str
    candidates: list[Candidate]


class OCRResponse(BaseModel):
    recognized_text: str
    characters: list[CharacterResult]


# ---------------- OCR ENDPOINT ----------------
@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    request: Request,
    file: UploadFile = File(...)
):
    # Rate limit (light)
    rate_limiter(request)

    # Validate image
    await validate_image(file)

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    temp_filename = f"{uuid.uuid4()}.png"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_ocr(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)