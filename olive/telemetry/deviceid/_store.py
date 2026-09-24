import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

from olive.telemetry.utils import get_telemetry_base_dir

REGISTRY_PATH = r"SOFTWARE\Microsoft\DeveloperTools\.onnxruntime"
REGISTRY_KEY = "deviceid"
MAX_DEVICE_ID_FILE_SIZE = 256


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        # Permission tightening is best-effort on filesystems that do not support chmod.
        pass


class Store:
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
        flags = os.O_RDONLY
        for optional_flag in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
            flags |= getattr(os, optional_flag, 0)
        try:
            fd = os.open(self._file_path, flags)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self._file_path.stem} does not exist") from None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise PermissionError(f"File {self._file_path.stem} is not a regular file")
            chunks = []
            remaining = MAX_DEVICE_ID_FILE_SIZE + 1
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            if len(content) > MAX_DEVICE_ID_FILE_SIZE:
                raise ValueError(f"File {self._file_path.stem} is too large")
            return content.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise ValueError(f"File {self._file_path.stem} is not valid UTF-8") from None
        finally:
            os.close(fd)

    def store_id(self, device_id: str, replace_existing: bool = False) -> bool:
        """Store the device id in the store location.

        :param str device_id: The device id to store.
        :type device_id: str
        """
        # create the folder location if it does not exist, owner-only (0700) so other users on the
        # machine cannot traverse into it to reach the device id.
        self._file_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod_best_effort(self._file_path.parent, 0o700)

        fd, temp_path = tempfile.mkstemp(prefix="deviceid.tmp.", dir=self._file_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(device_id)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            _chmod_best_effort(Path(temp_path), 0o600)
            if replace_existing:
                Path(temp_path).replace(self._file_path)
                temp_path = ""
                return True
            try:
                os.link(temp_path, self._file_path)
            except FileExistsError:
                return False
            return True
        finally:
            if temp_path:
                with suppress(OSError):
                    Path(temp_path).unlink()


class WindowsStore:
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
