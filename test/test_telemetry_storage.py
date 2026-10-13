# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""Real filesystem regressions for the shared telemetry storage protocol."""

import logging
import os
import sqlite3
import subprocess
from unittest.mock import patch

import pytest

from olive.telemetry.deviceid._store import Store
from olive.telemetry.offline_store import OfflineEventStore
from olive.telemetry.process_lock import ProcessDrainLock
from olive.telemetry.storage_paths import StorageDirectory


@pytest.fixture
def make_store():
    stores = []

    def create(path):
        store = OfflineEventStore(str(path))
        stores.append(store)
        return store

    yield create
    for store in stores:
        store.close()


def _link_directory(link, target):
    if os.name == "nt":
        quoted_link = str(link).replace("'", "''")
        quoted_target = str(target).replace("'", "''")
        command = f"New-Item -ItemType Junction -Path '{quoted_link}' -Target '{quoted_target}' | Out-Null"
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("nested", [False, True])
def test_directory_links_are_rejected_before_writing(tmp_path, make_store, nested):
    target = tmp_path / "outside"
    target.mkdir()
    link = tmp_path / "linked"
    _link_directory(link, target)
    directory = link / "nested" / "cache" if nested else link

    with (
        patch("olive.telemetry.deviceid._store.get_telemetry_base_dir", return_value=directory),
        pytest.raises(PermissionError),
    ):
        Store().store_id("00000000-0000-4000-8000-000000000000")
    with patch("olive.telemetry.offline_store.sqlite3.connect") as connect:
        store = make_store(directory / "queue.db")
    assert not store.is_open
    assert store._directory is None
    assert store._db_fd is None
    connect.assert_not_called()
    lock = ProcessDrainLock(str(directory / "queue.db.lock"))
    assert not lock.acquire()
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_hard_linked_database_files_are_rejected_before_connect(tmp_path, make_store, suffix):
    target = tmp_path / "outside"
    target.write_bytes(b"private data")
    directory = tmp_path / "storage"
    directory.mkdir()
    os.link(target, directory / ("queue.db" + suffix))
    with patch("olive.telemetry.offline_store.sqlite3.connect") as connect:
        store = make_store(directory / "queue.db")
    assert not store.is_open
    connect.assert_not_called()
    assert target.read_bytes() == b"private data"


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow file opens")
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_symlinked_database_files_are_rejected_before_connect(tmp_path, make_store, suffix):
    target = tmp_path / "outside"
    target.write_bytes(b"private data")
    directory = tmp_path / "storage"
    directory.mkdir()
    (directory / ("queue.db" + suffix)).symlink_to(target)
    with patch("olive.telemetry.offline_store.sqlite3.connect") as connect:
        store = make_store(directory / "queue.db")
    assert not store.is_open
    connect.assert_not_called()
    assert target.read_bytes() == b"private data"


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow file opens")
def test_device_id_file_replacement_cannot_follow_a_symlink(tmp_path):
    target = tmp_path / "outside"
    target.write_text("private data", encoding="utf-8")
    directory = tmp_path / "storage"
    directory.mkdir()
    device_file = directory / "deviceid"
    device_file.write_text("00000000-0000-4000-8000-000000000000", encoding="utf-8")
    original_open = os.open

    def swap_before_open(name, *args, **kwargs):
        if name == "deviceid":
            device_file.unlink()
            device_file.symlink_to(target)
        return original_open(name, *args, **kwargs)

    with (
        patch("olive.telemetry.deviceid._store.get_telemetry_base_dir", return_value=directory),
        patch("olive.telemetry.storage_paths.os.open", side_effect=swap_before_open),
        pytest.raises(OSError, match="symbolic link"),
    ):
        _ = Store().retrieve_id
    assert target.read_text(encoding="utf-8") == "private data"


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow file opens")
def test_device_id_and_lock_symlinks_cannot_modify_target(tmp_path):
    target = tmp_path / "outside"
    target.write_text("private data", encoding="utf-8")
    directory = tmp_path / "storage"
    directory.mkdir()
    (directory / "deviceid").symlink_to(target)
    (directory / "deviceid.lock").symlink_to(target)
    with (
        patch("olive.telemetry.deviceid._store.get_telemetry_base_dir", return_value=directory),
        pytest.raises(PermissionError),
    ):
        Store().store_id("00000000-0000-4000-8000-000000000000", replace_existing=True)
    lock = ProcessDrainLock(str(directory / "deviceid.lock"))
    assert not lock.acquire()
    assert target.read_text(encoding="utf-8") == "private data"


@pytest.mark.skipif(os.name == "nt", reason="Windows pins the open database against replacement")
@pytest.mark.parametrize("target_exists", [False, True])
def test_database_replacement_is_rejected_before_sql_writes(tmp_path, make_store, target_exists):
    target = tmp_path / "outside.db"
    if target_exists:
        with sqlite3.connect(target) as connection:
            connection.execute("CREATE TABLE sentinel(value TEXT)")
            connection.execute("INSERT INTO sentinel VALUES ('private data')")
        before = target.read_bytes()
    directory = tmp_path / "storage"
    directory.mkdir()
    database = directory / "queue.db"
    original_connect = sqlite3.connect

    def swap_before_connect(*args, **kwargs):
        database.unlink()
        database.symlink_to(target)
        return original_connect(*args, **kwargs)

    with patch("olive.telemetry.offline_store.sqlite3.connect", side_effect=swap_before_connect):
        store = make_store(database)
    assert not store.is_open
    assert store._directory is None
    assert store._db_fd is None
    if target_exists:
        assert target.read_bytes() == before
    else:
        assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership metadata")
def test_other_user_owned_files_are_rejected(tmp_path):
    filename = tmp_path / "queue.db"
    filename.touch()
    info = list(filename.stat())
    info[4] = os.geteuid() + 1
    with (
        StorageDirectory(tmp_path) as directory,
        patch("olive.telemetry.storage_paths.os.stat", return_value=os.stat_result(info)),
        pytest.raises(PermissionError),
    ):
        directory.validate_file(filename.name)


@pytest.mark.skipif(os.name == "nt", reason="POSIX private-directory permissions")
def test_permission_hardening_failure_disables_storage(tmp_path, make_store, caplog):
    with (
        caplog.at_level("DEBUG", logger="olive.telemetry.offline_store"),
        patch.object(logging.getLogger("olive"), "propagate", True),
        patch("olive.telemetry.storage_paths.os.fchmod", side_effect=PermissionError("cannot secure directory")),
        patch("olive.telemetry.offline_store.sqlite3.connect") as connect,
    ):
        store = make_store(tmp_path / "queue.db")
    assert not store.is_open
    assert store._directory is None
    assert store._db_fd is None
    connect.assert_not_called()
    assert "Telemetry storage initialization failed" in caplog.text


def test_storage_close_releases_pinned_handles(tmp_path, make_store):
    directory = tmp_path / "storage"
    directory.mkdir()
    store = make_store(directory / "queue.db")
    assert store.is_open
    store.close()
    store.close()
    assert store._directory is None
    assert store._db_fd is None
    directory.rename(tmp_path / "closed-storage")


@pytest.mark.skipif(os.name != "nt", reason="Windows file-handle delete sharing")
def test_open_file_cannot_be_replaced_on_windows(tmp_path):
    filename = tmp_path / "queue.db"
    with StorageDirectory(tmp_path) as directory:
        fd = directory.open_file(filename.name, os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(PermissionError):
                filename.unlink()
        finally:
            os.close(fd)
    filename.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-based ownership checks")
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_existing_files_require_owner_validation_on_windows(tmp_path, suffix):
    filename = tmp_path / ("queue.db" + suffix)
    filename.write_bytes(b"private data")
    with (
        StorageDirectory(tmp_path) as directory,
        patch("olive.telemetry.storage_paths._validate_windows_owner", side_effect=PermissionError),
        pytest.raises(PermissionError),
    ):
        directory.validate_file(filename.name)
    assert filename.read_bytes() == b"private data"
    filename.unlink()
