from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import datetime

class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    PENDING = "pending"
    CONNECTED = "connected"
    EXPIRED = "expired"
    ERROR = "error"

@dataclass
class SocialAccount:
    platform: str
    account_id: str
    display_name: Optional[str] = None
    status: ConnectionState = ConnectionState.DISCONNECTED
    connected_at: Optional[str] = None
    credential_ref: Optional[str] = None  # Opaque reference ONLY. Never store tokens or secrets!
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # Ensure credentials/secrets are never leaked in repr
        return f"SocialAccount(platform={self.platform!r}, account_id={self.account_id!r}, display_name={self.display_name!r}, status={self.status!r}, connected_at={self.connected_at!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize account data securely without exposing any credential secrets."""
        return {
            "platform": self.platform,
            "account_id": self.account_id,
            "display_name": self.display_name,
            "status": str(self.status.value if isinstance(self.status, ConnectionState) else self.status),
            "connected_at": self.connected_at,
            "has_credential_ref": bool(self.credential_ref),
            "metadata": self.metadata,
        }

class AccountStore:
    def __init__(self):
        self._accounts: Dict[str, SocialAccount] = {}
        # Pre-initialize supported platforms as disconnected by default
        default_platforms = ["youtube", "instagram", "facebook", "x", "threads"]
        for p in default_platforms:
            self._accounts[p] = SocialAccount(platform=p, account_id="", status=ConnectionState.DISCONNECTED)

    def list_accounts(self) -> List[SocialAccount]:
        """List all accounts across supported platforms."""
        return list(self._accounts.values())

    def get_account(self, platform: str) -> Optional[SocialAccount]:
        """Get account record for a given platform."""
        key = platform.lower()
        if key == "twitter":
            key = "x"
        elif key == "youtube_shorts":
            key = "youtube"
        return self._accounts.get(key)

    def link_account(
        self,
        platform: str,
        account_id: str,
        display_name: Optional[str] = None,
        credential_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SocialAccount:
        """Link or add an account with connected status."""
        key = platform.lower()
        if key == "twitter":
            key = "x"
        elif key == "youtube_shorts":
            key = "youtube"

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        account = SocialAccount(
            platform=key,
            account_id=account_id,
            display_name=display_name,
            status=ConnectionState.CONNECTED,
            connected_at=now,
            credential_ref=credential_ref,
            metadata=metadata or {}
        )
        self._accounts[key] = account
        return account

    def update_account(
        self,
        platform: str,
        status: Optional[ConnectionState] = None,
        display_name: Optional[str] = None,
        credential_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SocialAccount:
        """Update an existing account record."""
        account = self.get_account(platform)
        if not account:
            raise ValueError(f"Account for platform {platform} not found.")

        if status is not None:
            account.status = status
            if status == ConnectionState.CONNECTED and not account.connected_at:
                account.connected_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if display_name is not None:
            account.display_name = display_name
        if credential_ref is not None:
            account.credential_ref = credential_ref
        if metadata is not None:
            account.metadata.update(metadata)

        self._accounts[account.platform] = account
        return account

    def disconnect_account(self, platform: str) -> bool:
        """Disconnect an account, resetting status to disconnected."""
        account = self.get_account(platform)
        if not account:
            return False
        account.status = ConnectionState.DISCONNECTED
        account.account_id = ""
        account.credential_ref = None
        account.connected_at = None
        return True

    def is_connected(self, platform: str) -> bool:
        """Check if platform account is connected."""
        account = self.get_account(platform)
        return bool(account and account.status == ConnectionState.CONNECTED)


class AccountStatusResolver:
    """Helper for UploadWorker and dashboard to check platform account availability."""
    def __init__(self, store: Optional[AccountStore] = None):
        self.store = store or AccountStore()

    def is_platform_available(self, platform: str) -> bool:
        return self.store.is_connected(platform)

    def get_platform_status(self, platform: str) -> str:
        account = self.store.get_account(platform)
        if not account:
            return str(ConnectionState.DISCONNECTED.value)
        return str(account.status.value if isinstance(account.status, ConnectionState) else account.status)
