import os
import uuid
from contextlib import suppress
from pathlib import Path

from olive.telemetry.storage_paths import StorageDirectory
from olive.telemetry.utils import get_telemetry_base_dir

REGISTRY_PATH = r"SOFTWARE\Microsoft\DeveloperTools\.onnxruntime"
REGISTRY_KEY = "deviceid"
MAX_DEVICE_ID_FILE_SIZE = 256


class Store:
    """File persistence; the caller owns caching and publication coordination."""

    def __init__(self) -> None:
        self._file_path: Path = self._build_path

    @property
    def _build_path(self) -> Path:
        return get_telemetry_base_dir() / "deviceid"

    @property
    def retrieve_id(self) -> str:
        """Retrieve the device id from the store location.

        :return: The device id.
        :rtype: str
        """
        try:
            with StorageDirectory(self._file_path.parent) as directory:
                # Atomic publication briefly links two names; reads must not chmod that inode.
                fd = directory.open_file(self._file_path.name, os.O_RDONLY, allow_shared=True)
                with os.fdopen(fd, "rb") as device_id_file:
                    content = device_id_file.read(MAX_DEVICE_ID_FILE_SIZE + 1)
            if len(content) > MAX_DEVICE_ID_FILE_SIZE:
                raise ValueError(f"File {self._file_path.stem} is too large")
            return content.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise ValueError(f"File {self._file_path.stem} is not valid UTF-8") from None

    def store_id(self, device_id: str, replace_existing: bool = False) -> bool:
        """Store the device id in the store location.

        :param str device_id: The device id to store.
        :type device_id: str
        """
        with StorageDirectory(self._file_path.parent, create=True) as directory:
            directory.validate_file(self._file_path.name)
            temporary_name = f"deviceid.tmp.{uuid.uuid4().hex}"
            fd = directory.open_file(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                    temp_file.write(device_id)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                try:
                    directory.publish(temporary_name, self._file_path.name, replace_existing)
                except FileExistsError:
                    return False
                return True
            finally:
                with suppress(FileNotFoundError):
                    directory.unlink(temporary_name)


class WindowsStore:
    """Registry persistence; the caller coordinates the shared publication mutex."""

    @property
    def retrieve_id(self) -> str:
        """Retrieve the device id from the Windows registry."""
        import winreg

        with winreg.OpenKeyEx(
            winreg.HKEY_CURRENT_USER, REGISTRY_PATH, reserved=0, access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        ) as key_handle:
            device_id, value_type = winreg.QueryValueEx(key_handle, REGISTRY_KEY)
        if value_type != winreg.REG_SZ or not isinstance(device_id, str):
            raise ValueError(f"Registry value {REGISTRY_KEY} is not a string")
        return device_id.strip()

    def store_id(self, device_id: str, replace_existing: bool = False) -> bool:
        """Store the device id in the windows registry.

        :param str device_id: The device id to store.
        """
        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            REGISTRY_PATH,
            reserved=0,
            access=winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY | winreg.KEY_WOW64_64KEY,
        ) as key_handle:
            if not replace_existing:
                try:
                    winreg.QueryValueEx(key_handle, REGISTRY_KEY)
                    return False
                except FileNotFoundError:
                    pass
            winreg.SetValueEx(key_handle, REGISTRY_KEY, 0, winreg.REG_SZ, device_id)
        return True
