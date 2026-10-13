import pytest
from mission_ai.upload import (
    AccountStore,
    AccountStatusResolver,
    ConnectionState,
    SocialAccount,
    UploadWorker,
    UploadJob,
)
from mission_ai.output_manager import OutputManager
from mission_ai.models import ContentPackage, ImageAnalysis, JobStatus, PlatformContent
from PIL import Image

def test_create_and_link_account():
    store = AccountStore()
    account = store.link_account("instagram", "ig_user_123", display_name="Insta User", credential_ref="ref_xyz_secret_token_123")
    assert account.platform == "instagram"
    assert account.account_id == "ig_user_123"
    assert account.display_name == "Insta User"
    assert account.status == ConnectionState.CONNECTED
    assert account.credential_ref == "ref_xyz_secret_token_123"

def test_list_accounts():
    store = AccountStore()
    accounts = store.list_accounts()
    # 5 default platforms initialized
    assert len(accounts) >= 5
    platforms = {a.platform for a in accounts}
    assert {"youtube", "instagram", "facebook", "x", "threads"}.issubset(platforms)

def test_get_account():
    store = AccountStore()
    store.link_account("youtube", "yt_channel_1")
    acc = store.get_account("youtube")
    assert acc is not None
    assert acc.account_id == "yt_channel_1"

    # Test alias resolution
    acc_alias = store.get_account("youtube_shorts")
    assert acc_alias is not None
    assert acc_alias.account_id == "yt_channel_1"

def test_update_account():
    store = AccountStore()
    store.link_account("facebook", "fb_page_1", display_name="Old Name")
    updated = store.update_account("facebook", display_name="New Name", status=ConnectionState.EXPIRED)
    assert updated.display_name == "New Name"
    assert updated.status == ConnectionState.EXPIRED

def test_disconnect_account():
    store = AccountStore()
    store.link_account("x", "x_user_1")
    assert store.is_connected("x") is True

    success = store.disconnect_account("x")
    assert success is True
    assert store.is_connected("x") is False
    acc = store.get_account("x")
    assert acc.status == ConnectionState.DISCONNECTED
    assert acc.account_id == ""

def test_all_five_platforms_represented():
    store = AccountStore()
    platforms = ["youtube", "instagram", "facebook", "x", "threads"]
    for p in platforms:
        store.link_account(p, f"id_{p}")

    for p in platforms:
        assert store.is_connected(p) is True

def test_connected_disconnected_states():
    store = AccountStore()
    store.link_account("youtube", "yt_1")
    # Threads left disconnected by default
    assert store.is_connected("youtube") is True
    assert store.is_connected("threads") is False

def test_platform_independence_connect_disconnect():
    store = AccountStore()
    store.link_account("youtube", "yt_1")
    store.link_account("instagram", "ig_1")

    # Disconnect YouTube while Instagram remains connected
    store.disconnect_account("youtube")

    assert store.is_connected("youtube") is False
    assert store.is_connected("instagram") is True

def test_no_credential_secret_in_serialization():
    store = AccountStore()
    acc = store.link_account("instagram", "ig_1", credential_ref="super_secret_token_abc_xyz")
    d = acc.to_dict()

    # Ensure token/secret is not exposed in serialized dict or repr
    assert "super_secret_token_abc_xyz" not in str(d)
    assert "super_secret_token_abc_xyz" not in repr(acc)
    assert d["has_credential_ref"] is True

def test_account_resolver_offline():
    store = AccountStore()
    store.link_account("threads", "threads_1")
    resolver = AccountStatusResolver(store)

    assert resolver.is_platform_available("threads") is True
    assert resolver.is_platform_available("facebook") is False
    assert resolver.get_platform_status("threads") == "connected"
    assert resolver.get_platform_status("facebook") == "disconnected"

def test_existing_upload_worker_and_manual_mode(tmp_path):
    # Ensure UploadWorker still works normally and manual artifacts untouched
    output_mgr = OutputManager(tmp_path, "MISSION-22D")
    img = tmp_path / "img.png"
    Image.new("RGB", (32, 32), color="green").save(img)
    vid = output_mgr.videos_dir / "vid.mp4"
    vid.write_bytes(b"mp4")

    pkg = ContentPackage(
        image_id="img_1",
        source_path=str(img),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="instagram", caption="cap", hashtags=["#tag"])],
        status=JobStatus.COMPLETED,
        video_path=str(vid)
    )
    output_mgr.save_package(pkg)

    worker = UploadWorker()
    batch = worker.process_package("MISSION-22D", pkg, ["instagram"])
    assert batch.simulated_count == 1
    assert batch.success_count == 0
    assert (output_mgr.metadata_dir / "img_1.json").exists()
    assert (output_mgr.captions_dir / "img_1_caption.txt").exists()
