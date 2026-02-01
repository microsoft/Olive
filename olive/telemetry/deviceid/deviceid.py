import logging
import platform
import uuid
from typing import Union

from olive.telemetry.deviceid._store import Store, WindowsStore


def get_device_id(*, full_trace: bool = False) -> str:
    r"""Get the device id from the store or create one if it does not exist.

    An empty string is returned if an error occurs during saving or retrieval of the device id.

    Linux id location: $XDG_CACHE_HOME/deviceid if defined else $HOME/.cache/deviceid
    MacOS id location: $HOME/Library/Application Support/Microsoft/DeveloperTools/deviceid
    Windows id location: HKEY_CURRENT_USER\SOFTWARE\Microsoft\DeveloperTools\deviceid

    :keyword full_trace: If True, the full stack trace is logged. Default is False.
    :return: The device id.
    :rtype: str
    """
    logger = logging.getLogger(__name__)
    device_id: str = ""
    store: Union[Store, WindowsStore]

    try:
        if platform.system() == "Windows":
            store = WindowsStore()
        elif platform.system() in ("Linux", "Darwin"):
            store = Store()
        else:
            return device_id
        return store.retrieve_id()
    except (PermissionError, ValueError, NotImplementedError):
        if full_trace:
            logger.exception("Failed to retrieve stored device id.")
        return device_id
    except Exception:
        if full_trace:
            logger.exception("Failed to retrieve stored device id.")

    device_id = str(uuid.uuid4()).lower()

    try:
        store.store_id(device_id)
    except Exception:
        if full_trace:
            logger.exception("Failed to store device id.")
        device_id = ""

    return device_id
