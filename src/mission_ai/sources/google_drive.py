import json
import logging
import os
import re
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlencode

MASKED = "***MASKED***"

logger = logging.getLogger("mission_ai.sources.google_drive")


class GoogleDriveFolderResolver:
    """Resolve mission-provided Google Drive folder links into local images.

    Authentication modes, in priority order:
    1. Saved user OAuth credentials (private/shared folders the user can access)
    2. GOOGLE_DRIVE_ACCESS_TOKEN (already-issued user token)
    3. GOOGLE_DRIVE_API_KEY (public/"Anyone with the link" folders)
    4. GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE (optional automation identity)

    The mission only supplies a Drive URL. The caller never needs to move or
    re-share the mission folder to the Mission AI service account.
    """

    DRIVE_API_URL = "https://www.googleapis.com/drive/v3"
    DRIVE_SCOPE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        download_dir: Path = Path("./input"),
        credentials_path: Optional[str] = None,
        access_token: Optional[str] = None,
        api_key: Optional[str] = None,
        oauth_token_path: Optional[str] = None,
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.supported_exts = self.SUPPORTED_EXTS

        self.credentials_path = credentials_path or os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")
        self.access_token = access_token or os.getenv("GOOGLE_DRIVE_ACCESS_TOKEN")
        self.api_key = api_key or os.getenv("GOOGLE_DRIVE_API_KEY")
        self.oauth_token_path = oauth_token_path or os.getenv(
            "GOOGLE_DRIVE_OAUTH_TOKEN_FILE",
            str(Path.home() / ".config" / "mission-ai" / "google-drive-token.json"),
        )

        self._access_token: Optional[str] = None
        self._credentials: Optional[Any] = None
        self._auth_mode: Optional[str] = None

        # User OAuth is the primary path because mission folders may be private
        # but already shared with the operator's Google account.
        if self._load_user_oauth_credentials():
            self._auth_mode = "oauth"
        elif self.access_token:
            self._access_token = self.access_token
            self._auth_mode = "access_token"
        elif self.api_key:
            self._auth_mode = "api_key"
        elif self.credentials_path:
            self._load_service_account_credentials()
            self._auth_mode = "service_account"

    def _load_user_oauth_credentials(self) -> bool:
        """Load a previously-authorized Google user credential file, if present."""
        token_path = Path(self.oauth_token_path).expanduser()
        if not token_path.exists():
            return False

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request as GoogleAuthRequest

            credentials = Credentials.from_authorized_user_file(
                str(token_path),
                scopes=[self.DRIVE_SCOPE_READONLY],
            )

            if credentials.valid:
                self._credentials = credentials
                self._access_token = credentials.token
                logger.debug("Using saved Google user OAuth credentials")
                return True

            if credentials.expired and credentials.refresh_token:
                credentials.refresh(GoogleAuthRequest())
                self._credentials = credentials
                self._access_token = credentials.token
                self._write_secure_token_file(token_path, credentials.to_json())
                logger.debug("Refreshed Google user OAuth credentials")
                return True
        except Exception as exc:
            logger.warning("Saved Google OAuth credentials unavailable: %s", type(exc).__name__)

        return False

    @staticmethod
    def _write_secure_token_file(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def _load_service_account_credentials(self) -> None:
        """Load optional service-account credentials."""
        if not self.credentials_path:
            return

        creds_path = Path(self.credentials_path).expanduser()
        if not creds_path.exists():
            raise FileNotFoundError(f"Service account file not found: {creds_path}")

        try:
            with open(creds_path, "r") as f:
                creds_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Invalid service account file at {creds_path}: {e}") from e

        try:
            from google.oauth2 import service_account

            self._credentials = service_account.Credentials.from_service_account_info(
                creds_data,
                scopes=[self.DRIVE_SCOPE_READONLY],
            )
            self._refresh_access_token()
        except ImportError as e:
            raise ImportError(
                "Service account authentication requires 'google-auth'. "
                "Install with: pip install -e '.[drive]'"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to load service account from {creds_path}: {type(e).__name__}"
            ) from e

    def _refresh_access_token(self) -> None:
        """Refresh a Google credential using the standard requests transport."""
        if not self._credentials:
            raise ValueError("No credentials loaded to refresh")

        try:
            from google.auth.transport.requests import Request as GoogleAuthRequest

            self._credentials.refresh(GoogleAuthRequest())
            self._access_token = self._credentials.token
            logger.debug("Access token refreshed")
        except ImportError as e:
            raise ImportError(
                "Google authentication requires 'google-auth'. "
                "Install with: pip install -e '.[drive]'"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to refresh access token: {type(e).__name__}") from e

    def _extract_folder_id(self, url: str) -> str:
        """Extract a folder ID from supported Google Drive folder URLs."""
        match = re.search(
            r"drive[./](?:googleusercontent\.com|google\.com)/drive/(?:[uU]/[0-9]+/)?folders/([a-zA-Z0-9_-]+)",
            url,
        )
        if not match:
            raise ValueError(
                "Invalid Google Drive folder URL. "
                "Expected format: https://drive.google.com/drive/folders/FOLDER_ID"
            )
        return match.group(1)

    def _get_auth_headers(self) -> dict:
        """Return bearer headers when using OAuth/access-token/service-account auth."""
        if self._access_token:
            return {"Authorization": f"Bearer {self._access_token}"}

        if self._credentials:
            try:
                self._refresh_access_token()
            except Exception:
                pass
            if self._access_token:
                return {"Authorization": f"Bearer {self._access_token}"}

        # API-key mode intentionally has no Authorization header.
        if self.api_key and self._auth_mode == "api_key":
            return {}

        raise ValueError(
            "Google Drive access is not configured. "
            "Use user OAuth for private/shared mission folders, "
            "GOOGLE_DRIVE_API_KEY for public folders, or "
            "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE for folders explicitly shared to the service account."
        )

    def _api_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated or API-key Drive API request."""
        url = f"{self.DRIVE_API_URL}/{endpoint}"
        query_params = dict(params or {})

        if self._auth_mode == "api_key":
            query_params["key"] = self.api_key

        if query_params:
            url += "?" + urlencode(query_params, doseq=True)

        headers = self._get_auth_headers()
        headers["Accept"] = "application/json"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                error_data = json.loads(e.read().decode("utf-8")) if e.fp else {}
            except Exception:
                error_data = {}
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            safe_msg = error_msg.replace(str(self._access_token or ""), MASKED)
            raise RuntimeError(f"Google Drive API error: {safe_msg} ({e.code})") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Google Drive API connection error: {e.reason}") from e

    def _list_files_in_folder(self, folder_id: str) -> List[dict]:
        """List all files in a mission folder, including public shared-drive folders."""
        files: List[dict] = []
        page_token = None

        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken,files(id,name,mimeType,size,webContentLink,capabilities/canDownload)",
                "pageSize": 100,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token

            result = self._api_request("files", params)
            files.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return files

    def _is_supported_image(self, file: dict) -> bool:
        name = file.get("name", "").lower()
        mime_type = file.get("mimeType", "").lower()
        return Path(name).suffix.lower() in self.supported_exts or mime_type in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }

    def _download_file(self, file: dict) -> Path:
        """Download an image using the same auth mode used for folder discovery."""
        file_id = file.get("id")
        file_name = file.get("name")
        if not file_id or not file_name:
            raise ValueError("Invalid file metadata")

        local_path = self.download_dir / file_name
        if local_path.exists():
            return local_path

        # webContentLink is browser-oriented and can work without another API call
        # for public files, but it is not reliable for private mission folders.
        if self._auth_mode == "api_key":
            download_url = f"{self.DRIVE_API_URL}/files/{file_id}"
            query = {"alt": "media", "key": self.api_key}
        else:
            download_url = f"{self.DRIVE_API_URL}/files/{file_id}"
            query = {"alt": "media", "supportsAllDrives": "true"}

        req = urllib.request.Request(
            download_url + "?" + urlencode(query),
            headers=self._get_auth_headers(),
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                with open(local_path, "wb") as f:
                    f.write(response.read())
            return local_path
        except urllib.error.HTTPError as e:
            safe_name = Path(file_name).name
            raise RuntimeError(f"Failed to download file {safe_name}: HTTP {e.code}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to download file {Path(file_name).name}: {type(e).__name__}") from e

    def resolve(self, url: str) -> List[Path]:
        """Resolve a mission-provided Google Drive folder URL to local images."""
        folder_id = self._extract_folder_id(url)
        files = self._list_files_in_folder(folder_id)

        resolved_paths: List[Path] = []
        for file in files:
            if not self._is_supported_image(file):
                continue
            try:
                resolved_paths.append(self._download_file(file))
            except Exception as e:
                logger.warning(
                    "Failed to download %s: %s",
                    file.get("name", "unknown"),
                    type(e).__name__,
                )

        if not resolved_paths:
            raise RuntimeError(
                f"No supported images found in Google Drive folder {folder_id}. "
                f"Found {len(files)} files total."
            )

        return resolved_paths
