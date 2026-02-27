"""Document download service for Google Cloud Storage sources."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import GCSDownloadError


class GCSDownloader:
    """Download a GCS object to a local temporary file."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        gcs_client: Any | None = None,
        max_download_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_download_bytes = max_download_bytes
        self._gcs_client = gcs_client

    def download(self, bucket: str, object_key: str) -> Path:
        bucket_name = bucket.strip()
        blob_name = object_key.strip()
        if not bucket_name:
            raise GCSDownloadError("GCS bucket is required")
        if not blob_name:
            raise GCSDownloadError("GCS object key is required")

        suffix = self._resolve_suffix(blob_name)
        blob = self._resolve_blob(bucket_name, blob_name)
        blob_size = blob.size
        if blob_size is None:
            raise GCSDownloadError(f"Unable to determine size for gs://{bucket_name}/{blob_name}")
        if blob_size > self.max_download_bytes:
            raise GCSDownloadError(
                f"GCS object exceeds limit of {self.max_download_bytes} bytes"
            )

        temp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            blob.download_to_filename(str(temp_path))
            return temp_path
        except Exception as exc:
            self._safe_unlink(temp_path)
            raise GCSDownloadError(f"Failed to download gs://{bucket_name}/{blob_name}: {exc}") from exc

    def _resolve_blob(self, bucket_name: str, blob_name: str) -> Any:
        client = self._get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        try:
            blob.reload(client=client)
        except Exception as exc:
            raise GCSDownloadError(f"Unable to read GCS object metadata: {exc}") from exc

        return blob

    def _get_client(self) -> Any:
        if self._gcs_client is not None:
            return self._gcs_client

        encoded_credentials = self.settings.gcs_credentials
        if not encoded_credentials:
            raise GCSDownloadError("GCS credentials are not configured")

        try:
            from google.cloud import storage
            from google.oauth2 import service_account
        except Exception as exc:
            raise GCSDownloadError(f"Unable to import GCS client libraries: {exc}") from exc

        try:
            credentials_info = self._decode_credentials_info(encoded_credentials)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            self._gcs_client = storage.Client(
                project=credentials.project_id or credentials_info.get("project_id"),
                credentials=credentials,
            )
            return self._gcs_client
        except Exception as exc:
            raise GCSDownloadError(f"Unable to initialize GCS client: {exc}") from exc

    def _decode_credentials_info(self, encoded_credentials: str) -> dict[str, Any]:
        try:
            payload = encoded_credentials.strip()
            padding = "=" * (-len(payload) % 4)
            decoded_bytes = base64.b64decode(payload + padding)
            parsed = json.loads(decoded_bytes.decode("utf-8"))
        except Exception as exc:
            raise GCSDownloadError("GCS credentials must be valid base64-encoded JSON") from exc

        if not isinstance(parsed, dict):
            raise GCSDownloadError("GCS credentials JSON must be an object")

        return parsed

    def _resolve_suffix(self, object_key: str) -> str:
        suffix = Path(object_key).suffix.lower()
        if suffix in self.settings.allowed_extensions:
            return suffix

        allowed = ", ".join(self.settings.allowed_extensions)
        raise GCSDownloadError(
            f"Unsupported extension '{suffix}'. Allowed extensions: {allowed}"
        )

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
