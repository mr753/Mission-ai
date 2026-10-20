import os
import json
import logging
import time
import socket
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from mission_ai.models import ContentPackage
from mission_ai.upload.base import BasePlatformAdapter
from mission_ai.upload.models import UploadResult
from mission_ai.upload.registry import register_adapter

logger = logging.getLogger("mission_ai.upload.adapters")

class YouTubeShortsAdapter(BasePlatformAdapter):
    YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"

    @property
    def platform_name(self) -> str:
        return "youtube"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "video_path": package.video_path,
            "title": cap.title if cap and cap.title else "Mission Short",
            "description": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
            "tags": cap.hashtags if cap else [],
        }

    def _get_client_secrets(self, credentials: Optional[Dict[str, Any]] = None, token_file_data: Optional[Dict[str, Any]] = None) -> tuple[Optional[str], Optional[str], str]:
        client_id = None
        client_secret = None
        token_uri = "https://oauth2.googleapis.com/token"

        # 1. Credentials / config priority
        if credentials:
            client_id = credentials.get("client_id")
            client_secret = credentials.get("client_secret")
            if credentials.get("token_uri"):
                token_uri = credentials.get("token_uri")

        # 2. Token file data priority
        if token_file_data:
            client_id = client_id or token_file_data.get("client_id")
            client_secret = client_secret or token_file_data.get("client_secret")
            if not credentials or not credentials.get("token_uri"):
                token_uri = token_file_data.get("token_uri", token_uri)

        # 3. Environment variable / Client secret file fallback
        client_secret_path_str = os.getenv("YOUTUBE_CLIENT_SECRET_FILE")
        if client_secret_path_str and (not client_id or not client_secret):
            path = Path(client_secret_path_str).expanduser()
            if path.exists():
                try:
                    cdata = json.loads(path.read_text())
                    cfg = cdata.get("installed") or cdata.get("web") or cdata
                    client_id = client_id or cfg.get("client_id")
                    client_secret = client_secret or cfg.get("client_secret")
                except Exception:
                    pass

        env_token_uri = os.getenv("YOUTUBE_TOKEN_URI")
        if env_token_uri and (not credentials or not credentials.get("token_uri")) and (not token_file_data or not token_file_data.get("token_uri")):
            token_uri = env_token_uri

        return client_id, client_secret, token_uri

    def _get_access_token(self, credentials: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if credentials and credentials.get("access_token"):
            return credentials.get("access_token")

        token_path_str = os.getenv("YOUTUBE_TOKEN_FILE")
        if not token_path_str:
            return None

        token_path = Path(token_path_str).expanduser()
        if not token_path.exists():
            return None

        try:
            data = json.loads(token_path.read_text())
            access_token = data.get("token") or data.get("access_token")
            refresh_token = data.get("refresh_token")
            expires_at = float(data.get("expires_at", 0) or 0)

            # Check if access token is still valid (with 60s buffer)
            if access_token and expires_at > time.time() + 60:
                return access_token

            # Access token is expired or missing. Attempt refresh if refresh_token exists.
            if refresh_token:
                client_id, client_secret, token_uri = self._get_client_secrets(credentials, data)

                if not client_id or not client_secret:
                    logger.warning("YouTube OAuth refresh failed: missing client_id or client_secret.")
                    return None

                payload = urllib.parse.urlencode({
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }).encode("utf-8")

                req = urllib.request.Request(
                    token_uri,
                    data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        resp_bytes = resp.read()
                        try:
                            refreshed = json.loads(resp_bytes.decode("utf-8"))
                        except (ValueError, json.JSONDecodeError):
                            logger.warning("YouTube OAuth refresh returned invalid JSON response.")
                            return None
                except urllib.error.HTTPError as he:
                    logger.warning("YouTube OAuth refresh HTTP error status %s", he.code)
                    return None
                except (urllib.error.URLError, TimeoutError, socket.timeout) as ue:
                    logger.warning("YouTube OAuth refresh network error: %s", ue)
                    return None
                except Exception as e:
                    logger.warning("YouTube OAuth refresh unexpected error: %s", e)
                    return None

                new_token = refreshed.get("access_token")
                if not new_token:
                    logger.warning("YouTube OAuth refresh response did not contain access_token.")
                    return None

                expires_in = int(refreshed.get("expires_in", 3600))
                data["token"] = new_token
                data["access_token"] = new_token
                data["expires_at"] = time.time() + expires_in
                if refreshed.get("refresh_token"):
                    data["refresh_token"] = refreshed.get("refresh_token")

                self._write_token_atomically(token_path, data)
                return new_token

            # Expired and no refresh token available -> NEVER return expired token
            logger.warning("YouTube OAuth access token expired and no refresh token available.")
            return None
        except Exception as exc:
            logger.warning("Failed to load or refresh YouTube OAuth credentials: %s", type(exc).__name__)
            return None

    @staticmethod
    def _write_token_atomically(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.pop("client_secret", None)
        content = json.dumps(data, indent=2)
        tmp_path = path.parent / f"{path.name}.tmp"
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to write token file atomically: {e}") from e

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        now_ts = datetime.now(timezone.utc).isoformat()
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for YouTube Shorts upload",
                timestamp=now_ts
            )

        live_enabled = os.getenv("ENABLE_LIVE_UPLOAD", "false").lower() == "true"
        if not live_enabled:
            return UploadResult(
                platform=self.platform_name,
                status="stub_success",
                message="Stub upload executed: YouTube Shorts API/OAuth not configured yet. No real post made.",
                timestamp=now_ts
            )

        token = self._get_access_token(credentials)
        client_secret_env = os.getenv("YOUTUBE_CLIENT_SECRET_FILE")
        if not token and not (credentials and (credentials.get("client_secret_file") or credentials.get("access_token"))) and not client_secret_env:
            return UploadResult(
                platform=self.platform_name,
                status="not_configured",
                message="YouTube OAuth credentials not configured (YOUTUBE_TOKEN_FILE / YOUTUBE_CLIENT_SECRET_FILE missing).",
                timestamp=now_ts
            )

        payload = self.prepare_payload(package)
        video_path = payload["video_path"]
        title = payload["title"]
        description = payload["description"]
        tags = payload["tags"]

        try:
            metadata = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "22"
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False
                }
            }
            meta_bytes = json.dumps(metadata).encode("utf-8")
            headers = {
                "Authorization": f"Bearer {token or 'dummy_token'}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(Path(video_path).stat().st_size)
            }
            req = urllib.request.Request(self.YOUTUBE_UPLOAD_URL, data=meta_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                upload_url = resp.headers.get("Location")
                if not upload_url:
                    return UploadResult(
                        platform=self.platform_name,
                        status="failed",
                        error="Failed to obtain resumable upload session location header from YouTube API.",
                        timestamp=now_ts
                    )

            video_bytes = Path(video_path).read_bytes()
            upload_headers = {
                "Authorization": f"Bearer {token or 'dummy_token'}",
                "Content-Type": "video/mp4",
                "Content-Length": str(len(video_bytes))
            }
            upload_req = urllib.request.Request(upload_url, data=video_bytes, headers=upload_headers, method="PUT")
            with urllib.request.urlopen(upload_req, timeout=60) as upload_resp:
                resp_data = json.loads(upload_resp.read().decode("utf-8"))
                video_id = resp_data.get("id")
                if not video_id:
                    return UploadResult(
                        platform=self.platform_name,
                        status="failed",
                        error="YouTube upload response missing video id.",
                        timestamp=now_ts
                    )
                return UploadResult(
                    platform=self.platform_name,
                    status="success",
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    external_id=video_id,
                    message="YouTube Shorts uploaded successfully.",
                    timestamp=now_ts
                )

        except urllib.error.HTTPError as he:
            logger.warning("YouTube upload HTTP error status %s", he.code)
            if he.code == 429:
                return UploadResult(
                    platform=self.platform_name,
                    status="rate_limited",
                    error="YouTube API rate limit exceeded (HTTP 429).",
                    timestamp=now_ts
                )
            elif he.code in (401, 403):
                return UploadResult(
                    platform=self.platform_name,
                    status="failed",
                    error=f"YouTube API authentication/authorization failed (HTTP {he.code}).",
                    timestamp=now_ts
                )
            else:
                return UploadResult(
                    platform=self.platform_name,
                    status="failed",
                    error=f"YouTube API HTTP error {he.code}.",
                    timestamp=now_ts
                )
        except (TimeoutError, urllib.error.URLError, socket.timeout) as te:
            logger.warning("YouTube upload network error: %s", type(te).__name__)
            return UploadResult(
                platform=self.platform_name,
                status="timeout",
                error="YouTube API connection timeout or network error.",
                timestamp=now_ts
            )
        except Exception as e:
            logger.warning("YouTube upload unexpected error: %s", type(e).__name__)
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="YouTube upload failed due to an unexpected error.",
                timestamp=now_ts
            )


class InstagramAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "instagram"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "media_path": package.video_path,
            "caption": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for Instagram upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: Instagram API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class FacebookAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "facebook"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "video_path": package.video_path,
            "description": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for Facebook upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: Facebook API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class XAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "x"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        text = f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}"
        return {
            "platform": self.platform_name,
            "media_path": package.video_path,
            "text": text[:280],
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for X upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: X API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class ThreadsAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "x" if False else "threads"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "media_path": package.video_path,
            "text": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for Threads upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: Threads API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


# Register adapters
register_adapter("youtube", YouTubeShortsAdapter)
register_adapter("youtube_shorts", YouTubeShortsAdapter)
register_adapter("instagram", InstagramAdapter)
register_adapter("facebook", FacebookAdapter)
register_adapter("x", XAdapter)
register_adapter("twitter", XAdapter)
register_adapter("threads", ThreadsAdapter)
