from __future__ import annotations

from pathlib import Path

import aiofiles
from fastapi import UploadFile, status

from app.core.config import BACKEND_DIR, Settings
from app.core.exceptions import AppException


class StorageService:
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, settings: Settings):
        self.settings = settings

    async def persist_upload(self, task_id: str, upload_file: UploadFile) -> str:
        filename = upload_file.filename or "upload.bin"
        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed_suffixes:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=40011,
                message="unsupported image format",
            )

        if upload_file.content_type and not upload_file.content_type.startswith("image/"):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=40012,
                message="invalid content type for image upload",
            )

        upload_root = Path(self.settings.upload_dir)
        if not upload_root.is_absolute():
            upload_root = (BACKEND_DIR.parent / upload_root).resolve()
        upload_root.mkdir(parents=True, exist_ok=True)

        destination = upload_root / f"{task_id}{suffix}"
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        total_bytes = 0

        try:
            async with aiofiles.open(destination, "wb") as output_stream:
                while chunk := await upload_file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise AppException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            error_code=41311,
                            message="uploaded image exceeds size limit",
                        )
                    await output_stream.write(chunk)
        finally:
            await upload_file.close()

        return str(destination)
