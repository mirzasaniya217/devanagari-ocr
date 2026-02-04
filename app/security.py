from fastapi import UploadFile, HTTPException

ALLOWED_TYPES = ["image/png", "image/jpeg"]
MAX_SIZE_MB = 10

async def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images allowed"
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail="Image too large (max 10MB)"
        )

    # reset pointer so OCR can read again
    await file.seek(0)