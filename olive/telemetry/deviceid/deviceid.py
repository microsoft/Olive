import hashlib
import platform
import threading
import uuid
from contextlib import suppress
from enum import Enum
from typing import ClassVar

from olive.telemetry.deviceid._store import Store, WindowsStore
from olive.telemetry.process_lock import ProcessDrainLock
from olive.telemetry.utils import get_telemetry_base_dir


class DeviceIdStatus(Enum):
    NEW = "New"
    EXISTING = "Existing"
    CORRUPTED = "Corrupted"
    FAILED = "Failed"


_device_id_state = {"device_id": None, "status": DeviceIdStatus.NEW}
_device_id_lock = threading.RLock()


def _fnv1a_hex_bytes(value: bytes) -> str:
    """Hash the Windows SID only for the native-compatible mutex name."""
    hash_value = 14695981039346656037
    for byte in value:
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


class _WindowsDeviceIdMutex:
    """Named mutex compatible with the native device-id publication protocol."""

    def __init__(self) -> None:
        self._handle = None
        self._acquired = False
        self._kernel32 = None

    def acquire(self) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            class SidAndAttributes(ctypes.Structure):
                _fields_: ClassVar = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

            class TokenUser(ctypes.Structure):
                _fields_: ClassVar = [("user", SidAndAttributes)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            advapi32.OpenProcessToken.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.HANDLE),
            ]
            advapi32.GetTokenInformation.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            advapi32.IsValidSid.argtypes = [ctypes.c_void_p]
            advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
            advapi32.GetLengthSid.restype = wintypes.DWORD

            token = wintypes.HANDLE()
            if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
                return False
            try:
                size = wintypes.DWORD()
                advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
                if not size.value:
                    return False
                token_info = ctypes.create_string_buffer(size.value)
                if not advapi32.GetTokenInformation(token, 1, token_info, size.value, ctypes.byref(size)):
                    return False
                sid = ctypes.cast(token_info, ctypes.POINTER(TokenUser)).contents.user.sid
                if not sid or not advapi32.IsValidSid(sid):
                    return False
                sid_size = advapi32.GetLengthSid(sid)
                sid_hash = _fnv1a_hex_bytes(ctypes.string_at(sid, sid_size))
            finally:
                kernel32.CloseHandle(token)

            handle = kernel32.CreateMutexW(
                None,
                False,
                f"Global\\Microsoft.DeveloperTools.OnnxRuntime.DeviceId.{sid_hash}",
            )
            if not handle:
                return False
            self._handle = handle
            self._kernel32 = kernel32
            wait_result = kernel32.WaitForSingleObject(handle, 1000)
            self._acquired = wait_result in (0x00000000, 0x00000080)
            return self._acquired
        except Exception:
            self.release()
            return False

    def release(self) -> None:
        if self._handle is None or self._kernel32 is None:
            return
        if self._acquired:
            with suppress(Exception):
                self._kernel32.ReleaseMutex(self._handle)
        with suppress(Exception):
            self._kernel32.CloseHandle(self._handle)
        self._handle = None
        self._acquired = False
        self._kernel32 = None


def _is_valid_device_id(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 36:
        return False
    hyphens = {8, 13, 18, 23}
    return all(
        char == "-" if index in hyphens else char.lower() in "0123456789abcdef" for index, char in enumerate(value)
    )


def _initialize_device_id() -> str:
    r"""Get the device id from the store or create one if it does not exist.

    An empty string is returned if an error occurs during saving or retrieval of the device id.

    Linux id location: $XDG_CACHE_HOME/Microsoft/DeveloperTools/.onnxruntime/deviceid if defined
        else $HOME/.cache/Microsoft/DeveloperTools/.onnxruntime/deviceid
    MacOS id location: $HOME/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/deviceid
    Windows id location: HKEY_CURRENT_USER\SOFTWARE\Microsoft\DeveloperTools\.onnxruntime\deviceid

    :return: The device id.
    :rtype: str
    """
    system = platform.system()
    if system == "Windows":
        store = WindowsStore()
    elif system in ("Linux", "Darwin"):
        try:
            store = Store()
        except Exception:
            generated = str(uuid.uuid4()).lower()
            _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
            return generated
    else:
        generated = str(uuid.uuid4()).lower()
        _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
        return generated

    def read_existing() -> tuple[str, str]:
        try:
            existing = store.retrieve_id
        except (FileExistsError, FileNotFoundError):
            return ("missing", "")
        except ValueError:
            return ("invalid", "")
        except Exception:
            return ("failed", "")
        return ("valid", existing) if _is_valid_device_id(existing) else ("invalid", "")

    initial_state, existing = read_existing()
    if initial_state == "valid":
        _device_id_state.update({"status": DeviceIdStatus.EXISTING, "device_id": existing})
        return existing
    if initial_state == "failed":
        generated = str(uuid.uuid4()).lower()
        _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
        return generated

    lock = None
    acquired = True
    if system == "Windows":
        lock = _WindowsDeviceIdMutex()
        acquired = lock.acquire()
    elif initial_state == "invalid":
        lock = ProcessDrainLock(str(get_telemetry_base_dir() / "deviceid.lock"))
        acquired = lock.acquire(1.0)

    try:
        if not acquired:
            winner_state, winner = read_existing()
            generated = winner if winner_state == "valid" else str(uuid.uuid4()).lower()
            status = DeviceIdStatus.EXISTING if winner_state == "valid" else DeviceIdStatus.FAILED
            _device_id_state.update({"status": status, "device_id": generated})
            return generated

        current_state, current = read_existing()
        if current_state == "valid":
            _device_id_state.update({"status": DeviceIdStatus.EXISTING, "device_id": current})
            return current
        if current_state == "failed":
            generated = str(uuid.uuid4()).lower()
            _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
            return generated

        corrupted = initial_state == "invalid" or current_state == "invalid"
        generated = str(uuid.uuid4()).lower()
        try:
            stored = store.store_id(generated, replace_existing=corrupted)
        except Exception:
            stored = False
        if stored:
            status = DeviceIdStatus.CORRUPTED if corrupted else DeviceIdStatus.NEW
            _device_id_state.update({"status": status, "device_id": generated})
            return generated

        winner_state, winner = read_existing()
        if winner_state == "valid":
            _device_id_state.update({"status": DeviceIdStatus.EXISTING, "device_id": winner})
            return winner
        _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
        return generated
    finally:
        if lock is not None:
            lock.release()


def get_device_id() -> str:
    """Get the process-cached persistent device ID, initializing it once."""
    with _device_id_lock:
        if _device_id_state["device_id"] is None:
            return _initialize_device_id()
        return _device_id_state["device_id"]


def get_hashed_device_id_and_status() -> tuple[str, DeviceIdStatus]:
    """Get the canonical shared SHA-256 device ID and its status."""
    with _device_id_lock:
        device_id = get_device_id()
        hashed = hashlib.sha256(device_id.encode("utf-8")).hexdigest() if device_id else ""
        return f"c:{hashed}" if hashed else "", _device_id_state["status"]
