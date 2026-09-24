import json
import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Any

# Security: mask sensitive values in error messages
MASKED = "***MASKED***"

logger = logging.getLogger("mission_ai.sources.google_drive")


class GoogleDriveFolderResolver:
    """Google Drive folder resolver using Google Drive API v3.
    
    Supports two authentication methods (in priority order):
    1. Access Token: GOOGLE_DRIVE_ACCESS_TOKEN (for user contexts)
    2. Service Account: GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE (for automation)
    
    Folder URL format: https://drive.google.com/drive/folders/FOLDER_ID
    
    Required scopes for service account:
    - https://www.googleapis.com/auth/drive.readonly (recommended)
    - https://www.googleapis.com/auth/drive (for write access)
    """

    DRIVE_API_URL = "https://www.googleapis.com/drive/v3"
    DRIVE_SCOPE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        download_dir: Path = Path("./input"),
        credentials_path: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.supported_exts = self.SUPPORTED_EXTS
        
        # Get credentials from environment or parameters
        self.credentials_path = credentials_path or os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")
        self.access_token = access_token or os.getenv("GOOGLE_DRIVE_ACCESS_TOKEN")
        self._access_token: Optional[str] = None
        self._credentials: Optional[Any] = None
        
        # Priority: 1. Access token, 2. Service account
        if self.access_token:
            self._access_token = self.access_token
            logger.debug("Using access token authentication")
        elif self.credentials_path:
            self._load_service_account_credentials()
            logger.debug("Using service account authentication")

    def _load_service_account_credentials(self) -> None:
        """Load service account credentials and obtain access token."""
        if not self.credentials_path:
            return
        
        creds_path = Path(self.credentials_path)
        if not creds_path.exists():
            raise FileNotFoundError(f"Service account file not found: {creds_path}")
        
        try:
            with open(creds_path, "r") as f:
                creds_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Invalid service account file at {creds_path}: {e}") from e
        
        try:
            from google.oauth2 import service_account
            
            # Create credentials with Drive scope
            self._credentials = service_account.Credentials.from_service_account_info(
                creds_data,
                scopes=[self.DRIVE_SCOPE_READONLY]
            )
            
            # Get access token
            if self._credentials.token:
                self._access_token = self._credentials.token
            else:
                # Request new token
                self._refresh_access_token()
                
        except ImportError:
            raise ImportError(
                "Service account authentication requires 'google-auth' library. "
                "Install with: pip install google-auth"
            )
        except Exception as e:
            # Mask any credential details in error
            safe_path = str(creds_path)
            raise ValueError(f"Failed to load service account from {safe_path}: {type(e).__name__}") from e

    def _refresh_access_token(self) -> None:
        """Refresh the access token for service account credentials."""
        if not self._credentials:
            raise ValueError("No credentials loaded to refresh")
        
        try:
            # Request new token
            self._credentials.refresh(self._get_refresh_request)
            self._access_token = self._credentials.token
            logger.debug("Access token refreshed")
        except Exception as e:
            raise RuntimeError(f"Failed to refresh access token: {type(e).__name__}")

    def _get_refresh_request(self, token_url: str) -> Any:
        """Create a refresh request using urllib (no external dependencies)."""
        import urllib.request
        
        # Build the request
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        # Service account uses JWT assertion, not refresh token
        # For service account, we need to sign a JWT
        if hasattr(self._credentials, '_sign_jwt'):
            # Use google-auth's internal JWT signing
            # This will be called automatically by refresh()
            pass
        
        # For standard OAuth2 refresh
        data = f"grant_type=refresh_token&refresh_token={self._credentials.refresh_token}".encode()
        req = urllib.request.Request(token_url, data=data, headers=headers)
        return req

    def _get_auth_headers(self) -> dict:
        """Get authorization headers for API requests."""
        if not self._access_token:
            # Try to refresh if we have credentials
            if self._credentials:
                try:
                    self._refresh_access_token()
                except Exception:
                    pass
            
            if not self._access_token:
                raise ValueError(
                    "Google Drive authentication required. "
                    "Set GOOGLE_DRIVE_ACCESS_TOKEN or GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE."
                )
        
        return {"Authorization": f"Bearer {self._access_token}"}

    def _extract_folder_id(self, url: str) -> str:
        """Extract folder ID from Google Drive URL."""
        if "drive.google.com/drive/folders/" not in url:
            raise ValueError(
                f"Invalid Google Drive folder URL. "
                "Expected format: https://drive.google.com/drive/folders/FOLDER_ID"
            )
        # Extract from URL path
        parts = url.split("/drive/folders/")
        if len(parts) < 2:
            raise ValueError(f"Cannot extract folder ID from URL")
        folder_id = parts[1].split("/")[0].split("?")[0].split("&")[0]
        return folder_id

    def _api_request(self, endpoint: str, params: dict = None) -> dict:
        """Make authenticated API request to Google Drive."""
        url = f"{self.DRIVE_API_URL}/{endpoint}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"
        
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_data = json.loads(e.read().decode("utf-8")) if e.fp else {}
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            # Mask token in error messages
            safe_msg = error_msg.replace(str(self._access_token or ""), MASKED)
            raise RuntimeError(f"Google Drive API error: {safe_msg} ({e.code})")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Google Drive API connection error: {e.reason}")

    def _list_files_in_folder(self, folder_id: str) -> List[dict]:
        """List all files in a Google Drive folder."""
        files = []
        page_token = None
        
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "files(id, name, mimeType, size, webContentLink)",
                "pageSize": 100,
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
        """Check if file is a supported image."""
        name = file.get("name", "").lower()
        mime_type = file.get("mimeType", "").lower()
        
        # Check by extension
        ext = Path(name).suffix.lower()
        if ext in self.supported_exts:
            return True
        
        # Check by MIME type
        image_mimes = {
            "image/jpeg",
            "image/png",
            "image/webp",
        }
        if mime_type in image_mimes:
            return True
        
        return False

    def _download_file(self, file: dict) -> Path:
        """Download a file from Google Drive."""
        file_id = file.get("id")
        file_name = file.get("name")
        
        if not file_id or not file_name:
            raise ValueError("Invalid file metadata")
        
        # Try direct download URL first
        web_content_link = file.get("webContentLink")
        if web_content_link:
            try:
                local_path = self.download_dir / file_name
                if local_path.exists():
                    return local_path
                
                urllib.request.urlretrieve(web_content_link, local_path)
                return local_path
            except Exception:
                pass
        
        # Fallback: use API export
        export_url = f"{self.DRIVE_API_URL}/files/{file_id}?alt=media"
        headers = self._get_auth_headers()
        req = urllib.request.Request(export_url, headers=headers)
        
        local_path = self.download_dir / file_name
        if local_path.exists():
            return local_path
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(local_path, "wb") as f:
                    f.write(response.read())
            return local_path
        except Exception as e:
            # Mask file ID in error for security
            safe_error = str(e).replace(str(file_id or ""), "***")
            raise RuntimeError(f"Failed to download file {file_name}: {safe_error}")

    def resolve(self, url: str) -> List[Path]:
        """Resolve Google Drive folder URL to list of local image paths.
        
        Args:
            url: Google Drive folder URL (e.g., https://drive.google.com/drive/folders/FOLDER_ID)
        
        Returns:
            List of Path objects to downloaded/locally cached images
        
        Raises:
            ValueError: Invalid URL or authentication error
            RuntimeError: API error or download failure
        """
        folder_id = self._extract_folder_id(url)
        files = self._list_files_in_folder(folder_id)
        
        resolved_paths = []
        for file in files:
            if not self._is_supported_image(file):
                continue
            try:
                local_path = self._download_file(file)
                resolved_paths.append(local_path)
            except Exception as e:
                # Skip files that fail to download, but continue with others
                logger.warning(f"Failed to download {file.get('name', 'unknown')}: {type(e).__name__}")
                continue
        
        if not resolved_paths:
            raise RuntimeError(
                f"No supported images found in Google Drive folder {folder_id}. "
                f"Found {len(files)} files total."
            )
        
        return resolved_paths
